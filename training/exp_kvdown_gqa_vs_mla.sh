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
#   # PHASE=1(WikiText 一般言語、軽い同一条件比較):
#   STEPS=3000 PHASE=1 bash training/exp_kvdown_gqa_vs_mla.sh
#   # PHASE=3(finance domain を重めに。kv_down 現象が再現するかの中規模確認):
#   STEPS=6000 PHASE=3 OUTDIR=checkpoints/kvcmp_p3 bash training/exp_kvdown_gqa_vs_mla.sh
# =============================================================================
set -euo pipefail

PY="${PY:-python3}"
OUTDIR="${OUTDIR:-checkpoints/kvcmp}"
STEPS="${STEPS:-3000}"          # 各アーキの学習ステップ数(短い同一条件で相対比較)。
                                # PHASE=3 で kv_down 現象を狙うなら 6000 程度を推奨。
PHASE="${PHASE:-1}"             # 学習フェーズ。1=WikiText(一般言語)/ 3=finance domain。
                                # 累積 step 設計に噛み合うのはこの2つ(下の case で
                                # 対象外フェーズの steps を 0 にして実 N step に揃える)。
# EVAL_SET 既定はフェーズ依存(下で確定): PHASE=1→wikitext / PHASE=3→finance
EVAL_SET="${EVAL_SET:-}"
SEQ_LEN="${SEQ_LEN:-1024}"
BATCH="${BATCH:-4}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-200}"
LOOP_SEED="${LOOP_SEED:-0}"
EVAL_CHUNKS="${EVAL_CHUNKS:-30}"
CACHE_DIR="${CACHE_DIR:-.cache}"

# finance_pretrain.py の step 予算は累積式(p_total = phase1+...+phaseN steps)、
# LR cosine は p5_total 基準。実 N step の単発フェーズにするには、対象フェーズ以外の
# *_steps を 0 にして p5_total=N に揃える(LR が N で正しく減衰する)。
# 対応フェーズ: 1(WikiText 一般言語)/ 3(finance domain mix)。
case "$PHASE" in
  1)
    PHASE_STEP_ARGS=(--phase 1 --phase1_steps "$STEPS"
                     --phase2_steps 0 --phase3_steps 0 --phase4_steps 0 --phase5_steps 0)
    EVAL_SET="${EVAL_SET:-wikitext}"   # WikiText 学習 → in-distribution は wikitext
    ;;
  3)
    PHASE_STEP_ARGS=(--phase 3 --phase1_steps 0 --phase2_steps 0 --phase3_steps "$STEPS"
                     --phase4_steps 0 --phase5_steps 0)
    EVAL_SET="${EVAL_SET:-finance}"    # finance domain 学習 → 元発見と同じ finance 評価
    ;;
  *)
    echo "[error] PHASE=$PHASE は未対応。1(WikiText)か 3(finance domain)を指定。" >&2
    echo "        累積 step 設計と噛み合うのはこの2つ(他フェーズは別フェーズ分も込みで学習される)。" >&2
    exit 1
    ;;
esac

mkdir -p "$OUTDIR"
echo "OUTDIR=$OUTDIR  STEPS=$STEPS  PHASE=$PHASE  EVAL_SET=$EVAL_SET  seq=$SEQ_LEN  batch=$BATCH  lr=$LR"

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
      "${PHASE_STEP_ARGS[@]}" \
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
