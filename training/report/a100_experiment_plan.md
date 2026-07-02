# A100 実験計画 — Loop Curriculum の本評価とスケールアップ

ローカル pilot（16GB Mac）でハーネスの動作と計算削減（実時間 15〜25%）は確認済み。
本計画は A100 で **(1) loop curriculum が test-time scaling を生むか** を検証し、
そのうえで **(2) dim=2048 へのスケールアップ** を行う。

> 関連: ハーネスは `--loop_schedule {off,fixed,curriculum}`（`training/finance_pretrain.py`）、
> base 生成は `training/make_base_ckpt.py`（`--dim/--n_heads/--expert_dim/--max_loop_iters`）。
> 評価は `training/eval_perplexity.py`・`training/eval_finance_behavior.py`。

---

## 検証したい仮説

1. **速度**: curriculum は同ステップで fixed より速い（平均ループ数が低い分）。
2. **品質維持**: 同ステップ・同データで最終 loss / PPL が fixed と同等以上。
3. **test-time scaling（本命）**: curriculum 学習モデルは推論ループを増やす（8→12→16）と
   性能が伸びる。fixed 学習モデルは頭打ち or 悪化する。

---

## Stage 1: dim=768 で test-time scaling を安価に検証（A100 数時間）

既存の base checkpoint（dim=768, ~99M）とパイプラインをそのまま使い、低コストで仮説3を判定する。
**まずここで効果が出ることを確認してから Stage 2 に進む**（A100 数日の投資を de-risk）。

### 1-1. 2 本の学習（同条件・同ステップ）

```bash
# A: baseline（固定8）
python training/finance_pretrain.py --phase 1 --phase1_steps 20000 \
  --loop_schedule fixed --batch_size 32 --seq_len 1024 --dtype auto \
  --base_ckpt checkpoints/a100_v2_gpt2vocab/final.pt \
  --ckpt_dir checkpoints/exp_fixed --auto_resume --cache_dir /content/cache

# B: curriculum（フェーズ別レンジ + Phase2+ で裾≤12）
python training/finance_pretrain.py --phase 1 --phase1_steps 20000 \
  --loop_schedule curriculum --batch_size 32 --seq_len 1024 --dtype auto \
  --base_ckpt checkpoints/a100_v2_gpt2vocab/final.pt \
  --ckpt_dir checkpoints/exp_curr --auto_resume --cache_dir /content/cache
```

注: dim=768 の base は `max_loop_iters=8`。Stage 1 は裾が 8 を超えない範囲（Phase1 のみ or
裾を 8 に制限）で十分。test-time scaling の評価は推論側でループを増やして見る。

### 1-2. test-time scaling ablation（核心）

両モデルを **複数の推論ループ数**で評価し、深さに対する性能曲線を比較する。

```bash
for L in 4 8 12 16; do
  echo "=== fixed @ loops=$L ==="
  python training/eval_perplexity.py --ckpt checkpoints/exp_fixed/phase1_final.pt \
    --n_loops $L --split validation --device cpu
  echo "=== curriculum @ loops=$L ==="
  python training/eval_perplexity.py --ckpt checkpoints/exp_curr/phase1_final.pt \
    --n_loops $L --split validation --device cpu
done
```

挙動側も同様に:
```bash
python training/eval_finance_behavior.py --device cpu \
  --ckpts checkpoints/exp_fixed/phase1_final.pt checkpoints/exp_curr/phase1_final.pt \
  --loops 12   # 4 / 8 / 12 / 16 を変えて複数回
```

### 1-3. 成功条件

| 指標 | 合格ライン |
|---|---|
| 速度 | curriculum の wall-clock < fixed（log の `loops≈` と実時間で確認） |
| 品質 | loops=8 で PPL(curr) ≲ PPL(fixed) |
| **test-time scaling** | **PPL(curr) が loops 4→8→12→16 で単調改善 or 飽和。fixed は早期に頭打ち/悪化** |

→ Stage 1 で test-time scaling が出れば、Stage 2 に進む価値が確定する。

---

## Stage 2: dim=2048 へスケールアップ（A100 数日〜）

### 2-1. base checkpoint 生成（dim=2048, max_loop_iters=12）

```bash
python training/make_base_ckpt.py \
  --out checkpoints/base_dim2048/final.pt \
  --dim 2048 --n_heads 16 --expert_dim 2048 --max_loop_iters 12 --no_gpt2_init
```

- `max_loop_iters=12`: 裾（9〜12）に固有の深度 LoRA を割り当てるため（8 のままだと裾が
  ループ8 の LoRA を使い回し、extrapolation が薄れる）。
- `--no_gpt2_init`: dim≠768 は GPT-2 埋め込み流用不可（自動スキップもされる）。
- 規模: 総 ~520M params / アクティブ ~193M（routed 28→top2）。

### 2-2. トークン予算（ここが本質的な制約）

| 基準 | 必要トークン |
|---|---|
| アクティブ params 基準（下限） | ~3.9B |
| 総 params 基準（推奨） | ~10.4B |
| **実用目標** | **6〜10B** |

- 現行 Phase1（WikiText-103 ~115M tokens）は**2 桁足りない**。
- dim=2048 は GPT-2 init 不可 → **ゼロから一般事前学習が必須**。
  `training/3b_fine_web_edu.py`（FineWeb-Edu、数B〜10B規模）で汎用事前学習 →
  その後に 5 フェーズの金融適応を乗せる。
- ループモデルは Parcae 則で「再帰↑ならトークン↓でも同FLOPsで低loss」に振れる可能性あり。

### 2-3. メモリ・時間・運用

| 項目 | 見積り |
|---|---|
| 学習静的メモリ（混合精度+AdamW, 520M） | ~8.3GB + 活性化 → A100 40GB で余裕 |
| 活性化対策 | `--grad_checkpoint`（8〜12 ループ分の活性化を O(1) 化） |
| 速度 | 6B tokens で単一 A100 **~1 週間規模** |
| セッション制限 | Colab は切れる → `--auto_resume` + Drive 保存で分割実行 |
| 高速化 | bf16 + batch 最大化（`--dtype auto`）、余裕あれば SDPA 化 |

### 2-4. 本実験（dim=2048）

Stage 1 と同じ A/B（fixed vs curriculum）+ test-time scaling ablation を、
dim=2048 / max_loop_iters=12 / 6〜10B tokens で実施。裾（9〜12）は Phase2 以降で発火。

---

## 計測・記録

- `train.log` の `loops≈`（平均ループ）と step time → 速度効果。
- `eval_perplexity.py --n_loops {4,8,12,16}` → PPL の depth 曲線。
- `eval_finance_behavior.py --loops {4,8,12,16}` → structured/risk/uncertainty の depth 曲線。
- 結果は `training/report/` に追記し、ブログ（Qiita）へ反映。

## リスクと留意

- **品質はトークン量に律速**。dim だけ上げてデータ不足だと undertrained で「大きいのに賢くない」。
- **loop curriculum の C（単純削減）は不可**。可変サンプリング（A）+ 裾、必要なら truncated BPTT（B）。
- Stage 1 で test-time scaling が出なければ、Stage 2 のスケールアップは見送り/再設計。
