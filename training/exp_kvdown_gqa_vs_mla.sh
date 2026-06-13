#!/usr/bin/env bash
# =============================================================================
# kv_down INT8 脆弱性: MLA vs GQA の切り分け実験
# =============================================================================
# 背景:
#   phase5 (MLA) で「recurrent.block.attn.kv_down だけ fp32 に残すと INT8 化の
#   品質劣化を大きく回復」と判明した。kv_down は MLA の低ランク圧縮(K/V を latent に
#   圧縮)の射影。この感度が
#     H1: MLA の圧縮ボトルネック固有なのか
#     H2: KV 経路一般の性質(圧縮なしの GQA wk/wv でも起きる)なのか
#   を、MLA と GQA を**同一条件**で学習して切り分ける。
#
# フェアネス要件:
#   比較可能な GQA checkpoint が無いため、同じ base 初期化・同じデータ・同じ steps・
#   同じ seed で MLA と GQA を学習する(--attn_type だけを変える)。絶対 PPL は
#   短時間学習なので最終品質ではないが、「KV 下流射影の量子化感度」という*相対*比較は
#   両者を同一条件で学習する限り妥当。
#
# 読み方(exp_attn_ablation.py の出力):
#   MLA 側: 'INT8 except attn.kv_down (MLA compress)' の回復が大きい(既知)。
#   GQA 側: 'INT8 except attn.wk / attn.wv' の回復を見る。
#     - 回復が小さい  → kv_down 感度は圧縮固有(H1 支持)
#     - 回復も大きい  → KV 経路一般の性質(H2 支持)
#
# 使い方:
#   # 本番(Colab GPU 推奨)。STEPS は短め既定。data cache は事前に用意済みのこと。
#   bash training/exp_kvdown_gqa_vs_mla.sh
#   # ステップ数や出力先を変える:
#   STEPS=3000 OUTDIR=checkpoints/kvcmp PHASE=1 bash training/exp_kvdown_gqa_vs_mla.sh
# =============================================================================
set -euo pipefail

PY="${PY:-python3}"
OUTDIR="${OUTDIR:-checkpoints/kvcmp}"
STEPS="${STEPS:-3000}"          # 各アーキの学習ステップ数(短い同一条件で相対比較)
PHASE="${PHASE:-1}"             # 学習フェーズ。1=WikiText(短い同一条件の既定)。
                                # 注: finance_pretrain の step 予算は累積式のため、短い
                                # 単発ランがクリーンなのは PHASE=1。finance 分布で再現
                                # したいなら full curriculum を別途回すこと。
EVAL_SET="${EVAL_SET:-wikitext}"  # PHASE=1(WikiText 学習)なら in-distribution の wikitext
SEQ_LEN="${SEQ_LEN:-1024}"
BATCH="${BATCH:-4}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-200}"
LOOP_SEED="${LOOP_SEED:-0}"
EVAL_CHUNKS="${EVAL_CHUNKS:-30}"
CACHE_DIR="${CACHE_DIR:-.cache}"

mkdir -p "$OUTDIR"
echo "OUTDIR=$OUTDIR  STEPS=$STEPS  PHASE=$PHASE  seq=$SEQ_LEN  batch=$BATCH  lr=$LR"

# finance_pretrain.py の step 予算は累積式(p_total = phase1+...+phaseN steps)。
# このランナーは --phase1_steps だけを指定するため、短い単発ランがクリーンなのは
# PHASE=1 のみ。PHASE>=2 だと意図しない長さ(直前フェーズ分も込み)で学習される。
if [ "$PHASE" != "1" ]; then
  echo "[error] このランナーは PHASE=1(WikiText 短期同一条件)専用です。" >&2
  echo "        PHASE=$PHASE は finance_pretrain の累積 step 設計と噛み合いません。" >&2
  echo "        finance 分布で再現したい場合は full curriculum を別途実行してください。" >&2
  exit 1
fi

run_one () {
  local attn="$1"
  local base="$OUTDIR/base_${attn}.pt"
  local ckdir="$OUTDIR/train_${attn}"

  echo
  echo "############################################################"
  echo "#  [$attn]  base 作成 -> 学習 -> ablation"
  echo "############################################################"

  # 1) base checkpoint(--attn_type だけ変える。それ以外は同一既定)
  "$PY" -u training/make_base_ckpt.py --attn_type "$attn" --out "$base"

  # 2) 同一条件で学習(seed/データ/steps/seq/batch/lr/warmup を共有)
  "$PY" -u training/finance_pretrain.py \
      --phase "$PHASE" \
      --phase1_steps "$STEPS" \
      --base_ckpt "$base" \
      --ckpt_dir "$ckdir" \
      --seq_len "$SEQ_LEN" \
      --batch_size "$BATCH" \
      --lr "$LR" \
      --warmup_steps "$WARMUP" \
      --loop_seed "$LOOP_SEED" \
      --cache_dir "$CACHE_DIR" \
      --save_every 100000   # 中間保存を抑制(ディスク節約)

  # 3) 同じ ablation(MLA は kv_down, GQA は wk/wv のサブグループが効く)
  echo
  echo "=== [$attn] attention ablation (eval_set=$EVAL_SET) ==="
  "$PY" -u training/exp_attn_ablation.py \
      --ckpt "$ckdir/phase${PHASE}_final.pt" \
      --eval_max_chunks "$EVAL_CHUNKS" \
      --eval_set "$EVAL_SET" \
      --cache_dir "$CACHE_DIR" \
      --allow_unsafe_checkpoint
}

run_one mla
run_one gqa

echo
echo "完了。MLA の 'INT8 except attn.kv_down' と GQA の 'INT8 except attn.wk/wv' の"
echo "回復幅を比べて H1(圧縮固有)/ H2(KV経路一般)を判定すること。"
