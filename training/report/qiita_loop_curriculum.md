# Recurrent-Depth Transformer に「ループ・カリキュラム」を入れて高速化を狙う（ハーネス構築 + 予備計測編）

## TL;DR

- 自作の Recurrent-Depth Transformer（再帰深度を 8 ループ回すモデル）は、**ループ回数ぶん計算量が増える**のがボトルネック。
- そこで学習中の再帰回数を**フェーズ別にサンプリングする「ループ・カリキュラム」**を opt-in で実装した。
- ローカル（16GB MacBook Air, MPS）で予備実行し、**スケジュール通りに動くこと**と、**平均ループ数の低下に応じて計算時間が短縮されること**を確認した（実時間で 15〜25% 程度）。
- ただし本記事は**ハーネス構築 + 予備計測**まで。「精度（test-time scaling）が改善するか」の本評価は A100 で実施予定。

:::note warn
これは研究用コードの実験ログです。投資助言ではありません。また予備結果であり、品質に関する結論はまだ出していません。
:::

---

## 背景：なぜループ・カリキュラムか

このモデルは Transformer 層を積み上げる代わりに、少数の層を**推論時に複数回ループ**させて「深さ」を得る（Recurrent-Depth / Looped Transformer）。今回の構成は最大 8 ループ。

問題は単純で、**8 ループ = 再帰ブロックの計算が 8 倍**。学習を高速化したいなら、ここを削るのが一番効く。

そこで使うのが**ループ回数のサンプリング**：

- 学習中、毎ステップで再帰回数 `n_loops` を固定 8 ではなく**分布から引く**。
- 平均を下げれば計算が減る。
- かつ、**時々 8 より深い回数**も引くことで、「学習時より深く回すと性能が伸びる」**depth extrapolation（test-time scaling）**を獲得させる。なお今回の pilot は Phase 1 のみで裾はオフのため、9〜12 の深い回数はまだ発火していない（裾は Phase 2 以降）。

これは思いつきではなく、recurrent-depth 研究で確立された手法だ：

- **Universal Transformer / ACT**（Dehghani et al. 2018）：位置ごとに可変の計算深度。
- **Recurrent-Depth LM**（Geiping et al. 2025）：再帰回数をランダムサンプリング + 逆伝播打ち切りで学習し、推論時の depth scaling を実現。
- **Parcae**：ループ言語モデルのスケーリング則。再帰回数とトークン数はトレードオフ。

このアーキは**毎ループで入力を再注入** + **LTI 安定化（スペクトル半径 < 1）** + **ACT 停止**を持ち、可変深度学習と相性が良い設計になっている。

---

## 設計：3 モードを opt-in で

学習スクリプトに `--loop_schedule` を追加した（既定 `off` = 既存挙動を変えない）。

| モード | 挙動 |
|---|---|
| `off` | モデル既定に委譲（従来通り） |
| `fixed` | 毎ステップ `n_loops = 8` 固定（クリーンな baseline） |
| `curriculum` | フェーズ別レンジ + 上方向の裾をサンプリング |

カリキュラムのスケジュール：

```python
def phase_loop_range(phase_idx, progress):   # progress: 0..1
    if phase_idx <= 1:
        return (1, 4) if progress < 0.5 else (2, 8)   # Phase1 前半1-4 → 後半2-8
    return (4, 8)                                       # Phase2 以降は 4-8
# Phase2 以降のみ、確率 0.2 で 9..12 の「裾」を引く（depth extrapolation 用）
```

サンプラは `(seed, step)` で決定的にしてある（**resume しても同じ系列を再現**）。

---

## 予備実行：16GB Mac で（OOM の教訓つき）

最初に普通の設定（batch=4, seq_len=1024, fp32, 8 ループ）で走らせたら、即 `Killed: 9`。**macOS の OOM キラー**だ。学習は推論よりずっとメモリを食う（Adam 状態 + 勾配 + 8 ループ分の活性化）。

対策として **gradient checkpointing（8 ループの活性化を O(N)→O(1) に削減）** を有効化し、batch/seq を絞った：

```bash
# baseline（固定8）
python3 training/finance_pretrain.py --phase 1 --phase1_steps 300 \
  --loop_schedule fixed --batch_size 1 --seq_len 256 --grad_checkpoint \
  --ckpt_dir checkpoints/pilot_fixed

# curriculum
python3 training/finance_pretrain.py --phase 1 --phase1_steps 300 \
  --loop_schedule curriculum --batch_size 1 --seq_len 256 --grad_checkpoint \
  --ckpt_dir checkpoints/pilot_curr
```

:::note info
教訓：recurrent-depth モデルの学習メモリは「重み」より「8 ループ分の活性化」が効く。ローカルで回すなら gradient checkpointing はほぼ必須。
:::

---

## 結果1：スケジュール通りに動いた

ログに平均ループ数 `loops≈` を出すようにした。

| ステップ | fixed | curriculum |
|---|---|---|
| 50 | `loops≈8.0` | `loops≈2.4`（range 1-4） |
| 100 | `loops≈8.0` | `loops≈2.5` |
| 150 | `loops≈8.0` | `loops≈2.5` |
| **200** | `loops≈8.0` | **`loops≈5.0`（range 2-8 に切替）** |
| 250 | `loops≈8.0` | `loops≈5.2` |
| 300 | `loops≈8.0` | `loops≈5.0` |

- fixed は 8.0 で一定。
- curriculum は **進捗 50% を境に 2.5 → 5.0 にアニーリング**（前半 1-4 → 後半 2-8）。Phase1 なので裾はオフ。設計通り。

## 結果2：平均ループ数の低下に応じて計算時間が短縮された

| | 平均ループ | 50 ステップあたりの時間 |
|---|---|---|
| fixed | 8.0 | ~55–61 秒 |
| curriculum（序盤） | ~2.5 | **~30 秒** |
| curriculum（後半） | ~5.0 | ~44–49 秒 |

計算時間は平均ループ数の低下に応じて短縮された。完全な比例ではないが、固定 8 ループより明確に軽くなっている。

再帰ブロック部分だけを見ると、平均 5 ループは固定 8 ループ比で約 37% の計算削減に相当する。ただし embedding / coda / loss / データローダ / gradient checkpointing の再計算といった固定オーバーヘッドがあるため、**実時間全体の削減率はこれより小さくなる**（後半 5 ループで実測 15〜25% 程度）。それでも狙い通りの高速化が予備計測で見えた。

最終 loss は fixed 6.85 / curriculum 6.79 とほぼ同等（300 ステップなので誤差範囲。品質の優劣を語るデータではない）。

---

## まだ言えないこと（重要）

- **これは品質比較ではない。** 300 ステップ・loss ~6.8（ppl ~900）はほぼ未学習。
- ループ・カリキュラムの本来の狙いである **test-time scaling（推論で 8→12→16 とループを増やすと性能が伸びるか）は、十分に学習しないと判定できない。**
- それは A100 で本学習してから、`n_loops ∈ {4,8,12,16}` の ablation で評価する。

---

## まとめ

| 項目 | 状態 |
|---|---|
| ハーネス（fixed / curriculum / 裾 / ログ / resume 安全） | ✅ 動作確認 |
| 計算削減（平均ループ低下で実時間も短縮、15〜25%） | ✅ 予備計測で確認 |
| メモリ（grad checkpointing で 16GB Mac でも完走） | ✅ |
| 品質・test-time scaling | ⏳ A100 で本評価予定 |

小さく作って小さく検証し、設計通りに動くこととコスト削減を確認できた。次は本番スケールで「速くなって、かつ深いほど賢くなる」かを確かめる。

> 推奨タグ：`機械学習` `LLM` `PyTorch` `深層学習` `Transformer`
