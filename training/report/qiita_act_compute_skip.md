# 適応計算(ACT)を推論速度に変換する — 出力 bit 一致のまま chat decode 2.45 倍

## TL;DR

- 自作 Recurrent-Depth Transformer(98.6M)は **ACT(Adaptive Computation Time)** で入力ごとに再帰ループ数を変える。しかし **KV キャッシュ推論では halt 後もフルのループ実行が必要**だった — 将来のトークンが全ループ深度のキャッシュを参照するため。
- 観察: 全位置が halt した後のループでは、ブロック入力が **halt 時点の凍結値に固定**される。つまり残り深度で本当に必要なのは「その凍結値の K/V 射影をキャッシュに詰めること」だけ。
- 実装: 全 halt 後の残り深度を **K/V projection + cache append のみに縮退**(query・アテンション・MoE・LoRA・injection・ACT predictor を省略)。キャッシュ内容はフル実行と**バイト一致**なので、これは近似ではなく**厳密に同じ計算の省略**。
- A100 実測(phase5 checkpoint、batch 1 chat、8 loops): **decode 25.60 → 62.61 tok/s(2.446 倍)**、ループスロットの **74.6% を cache-only 化**、出力トークン ID 完全一致、peak VRAM 同一。
- 「簡単なトークンは浅く、難しいトークンは深く」という ACT の設計思想が、**品質を 1 bit も変えずに実速度へ変換された**。

:::note warn
研究用コードの実験ログです。対象は自作モデル(Recurrent-Depth Transformer + MoE + ACT、金融特化学習)。数値はこの構成での実測であり、一般化はできません。
:::

---

## 背景: ACT はあるのに、推論は速くならなかった

自作モデル [BushidoMythos](https://github.com/tedorigawa001/BushidoMythos) は再帰ブロックを最大 12 回ループさせ、ACT が位置ごとに「もう十分考えた」と判断したら halt する。学習時は halt した位置の hidden 更新が止まり、ponder cost で平均ループ数も制御できる。

ところが**推論(KV キャッシュあり)では、この適応性が速度に変換できていなかった**。理由はキャッシュの構造にある:

- 各ループ深度 t は独自のキャッシュキー `recurrent_loop_t` を持つ
- 将来のトークンを decode するとき、その全深度のキャッシュエントリに過去トークンの K/V が**揃っている必要がある**
- よって「全位置が halt したからループを打ち切る」と、**将来の decode が参照するキャッシュに穴が開く**

従来実装はこれを正しく認識していて、キャッシュありのときは halt 後も全ループをフル実行していた。安全だが、halt の意味がない。

## 観察: halt 後のループは「同じ入力」を処理している

鍵は halt 処理の既存実装にあった。halt した位置のブロック入力は、halt 時点の値 `combined_frozen` に**凍結**される(ループ индекс埋め込みの影響も含めて固定):

```python
# 毎ループ: halt 済み位置は凍結値で置換
combined = torch.where(halted.unsqueeze(-1), combined_frozen, combined_new)
```

つまり**全位置が halt した後のループでは、ブロックへの入力が毎回まったく同じ**になる。フル実行がそこでやっている仕事を分解すると:

| 計算 | 出力はどこへ行くか |
|---|---|
| K/V projection → cache append | **将来の decode が参照する(必要)** |
| Q projection、アテンション、出力射影 | halt 済みなので h に反映されない(捨てられる) |
| MoE FFN、depth-LoRA、LTI injection | 同上(捨てられる) |
| ACT predictor | 全 halt 済みなので無意味 |

必要なのは1行目だけだ。しかも入力は `combined_frozen` に固定されているのだから、**残りの全深度に対して「凍結値の K/V」を先に詰めてループを抜けてよい** — フル実行と結果は完全に同じになる。

## 実装: cache-only loop 充填

アテンション層に「K/V 射影とキャッシュ追記だけ」を行うメソッドを追加した(GQA は `wk`/`wv`、MLA は `kv_down` の圧縮 latent のみ — ブロック全体のコストのごく一部):

```python
# 全 halt 検出時(no-grad 推論 + KV cache のみ)
if halted.all() and kv_cache is not None:
    if self.act_compute_skip and not self.training and not torch.is_grad_enabled():
        for future_t in range(t + 1, n_loops):
            self.block.append_attention_kv_cache(
                combined_frozen, freqs_cis, kv_cache, f"recurrent_loop_{future_t}"
            )
        break
```

安全設計:

- **学習・勾配有効時は絶対に発動しない**(二重ガード + 専用テスト)
- `compute_stats()` で executed / cache-only ループ数と skip 率を観測可能
- chat は既定有効、`--disable_act_compute_skip` で legacy に退避可能

そして最重要のテスト — GQA/MLA 両方で、**logits・全キャッシュ tensor・skip 充填キャッシュを使った次 decode ステップまで `torch.equal`(bit 一致)**を assert した。数値プローブ(許容誤差つき比較)ですらなく、恒等の証明である。

## 結果: 2.45 倍、出力は 1 トークンも変わらない

A100 BF16、学習済み phase5 checkpoint、batch 1 chat(prompt 64、生成 32、n_loops 8):

| 指標 | legacy | cache-only skip |
|---|---|---|
| decode tok/s | 25.60 | **62.61(2.446×)** |
| 実行ループ / 全スロット | 1280 / 1280 | **325 / 1280(74.6% を skip)** |
| 平均フル実行ループ数 | 8.0 / token | **2.03 / token** |
| peak VRAM | 459.4 MiB | 459.4 MiB(同一) |
| 出力トークン ID | — | **完全一致** |

`compute_skip_fraction=74.6%` の意味は重い。**decode 中、このモデルは平均 2 ループで「考え終わって」いた**。8 ループ分の計算を払い続けていた従来推論は、6 ループ分をドブに捨てていたことになる。ACT が学習で獲得した「入力の難易度に応じた計算配分」が、初めてそのまま実速度になった。

なお同じモデルの chat では、学習で 1.99 倍を出した MoE grouped GEMM が**逆に 0.80 倍**だった(小さい routing 行数ではカーネル起動を償却できない)。ワークロードごとに勝つ最適化は違う — 学習は grouped GEMM、decode は ACT skip、と適材適所になった。

## 学び

1. **「品質リスクのある最適化」は、問題を作り替えると無リスク化できることがある**。halt 済み位置の計算スキップは普通は近似(品質影響あり)だが、「出力に影響しないと証明できる計算だけを省く」形に再定式化すれば、検証は数値プローブではなく**恒等テスト**で済む。
2. **キャッシュの整合要件を「フル実行」で満たす必要はない**。必要なのはキャッシュの中身であって、それを生成した計算過程ではない。入力が凍結されるという既存実装の性質が、最小コストの充填方法を与えてくれた。
3. **適応計算は「実装しただけ」では速くならない**。ACT・early-exit・MoD 系の機構は、推論経路(特にキャッシュ)がその適応性を速度に変換できる形になっているかまで含めて設計する必要がある。

## まとめ

| 項目 | 結果 |
|---|---|
| decode 速度 | ◎ 2.446×(25.60 → 62.61 tok/s) |
| 計算削減 | ◎ ループスロットの 74.6% を cache-only 化 |
| 品質 | ◎ 出力 bit 一致(logits・キャッシュ・次ステップまで torch.equal) |
| メモリ | ◎ 増加なし |
| 適用範囲 | no-grad 推論 + KV cache のみ(学習経路は不変) |

実装・ベンチハーネス・計測 JSON: [BushidoMythos](https://github.com/tedorigawa001/BushidoMythos)(`bushido_mythos/main.py`, `training/bench_chat_act_skip.py`, `docs/performance_roadmap.md`)
