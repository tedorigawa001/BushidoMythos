# 有名最適化カーネルが自分のモデルで速いとは限らない — Liger Fused CE を実測して不採用にした話

## TL;DR

- LLM 学習の定番メモリ最適化 **Liger Kernel の Fused Linear Cross Entropy(FLCE)** を自作モデル(98.6M、dim=768、vocab 50k)に実装し、A100 で実測した。
- 数値は完全に合格(loss / hidden 勾配 / tied weight 勾配のプローブ delta ≤ 0.002、bf16 丸め水準)。
- しかし速度は **full-logits CE 比 -57.3%**(75,903 → 32,435 tok/s)。自前実装した**依存ゼロの chunked CE にすら -46%** で完敗し、VRAM 削減幅も chunked と 82 MiB しか違わなかった。
- 結論: **本番不採用**。通常 run は full logits、OOM 時は chunked CE、Liger はモデル拡大時に再評価する実験経路として残す。
- 教訓は「Liger が悪い」ではない。**大規模モデル向けに設計された融合カーネルは、小さい形状では素の cuBLAS GEMM に勝てない**ことがあり、しかも**その損益分岐は自分のモデルで測るまで分からない**。

:::note warn
研究用コードの実験ログです。数値は自作 Recurrent-Depth Transformer(98.6M)+ A100 BF16 の特定構成での実測であり、Liger Kernel の一般的な性能評価ではありません。大規模モデル・巨大 vocab では Liger が大きな効果を出す報告が多数あります。
:::

---

## 背景: LM head の logits がメモリを支配する

学習時、LM head は `(B, T, vocab)` の logits を生成してから cross entropy を計算する。B=16, T=256, vocab=50,257 なら bf16 で約 400MB — 勾配や CE の fp32 中間も含めると、実測でこのテンソル群が **peak VRAM の約 57%** を占めていた。

これを消す定番が2つある:

1. **chunked CE(自前・依存なし)**: トークン軸をチャンク分割し、各チャンクの LM head + CE を activation checkpoint する。forward で logits を捨て、backward で再計算する。
2. **Liger Kernel の FLCE**: Triton の融合カーネルで、logits を実体化せずに loss と勾配を直接計算する。LLaMA 系のファインチューニングで広く使われている。

自作モデルにはまず 1 を実装済みで(-57% VRAM / -11〜21% 速度、OOM 用 fallback として採用)、「速度とメモリを同時に取る本命」として 2 を実装・計測した。

## 実装: 検証は同じ規律で

これまでの最適化(grouped GEMM 等)で確立した規律をそのまま適用した:

- `--liger_fused_ce` は **fail-fast**(CUDA / liger-kernel がなければ理由付き即エラー、silent fallback なし)
- SFT の loss mask は `ignore_index=-100` へ変換(mean 縮約の意味論が既存経路と一致することを確認)
- 全 mask バッチは autograd 接続を保った厳密な zero loss
- ベンチは計測前に**数値プローブ**: 部分 mask 条件で loss・hidden 勾配・tied weight 勾配を full-logits CE と突き合わせ、delta を JSON に保存
- 結果 JSON に `liger_fused_ce_active` / `request_valid` / `steady_state_valid` を記録

## 結果: 数値は合格、速度は完敗

A100 BF16、本番相当条件(batch 16、seq 256、8 loops、gradient checkpointing、grouped MoE、compile、steps=warmup=100)。3経路を同一条件で比較:

| CE 経路 | compile tok/s | peak VRAM | 追加依存 |
|---|---|---|---|
| full logits(既定) | **75,903** | 3,137 MiB | なし |
| chunked ce1024(自前) | 60,244(-20.6%) | 1,385 MiB | なし |
| **Liger FLCE** | **32,435(-57.3%)** | 1,303 MiB | liger-kernel |

数値プローブは全項目合格(loss 0.0020 / hidden 勾配 0.0020 / weight 勾配 0.0010 — bf16 丸め水準)、first loss も legacy と一致。**壊れているから遅いのではなく、このワークロードに合わないから遅い**。

- Liger は速度で最下位。メモリ最小だが、依存なしの chunked との差は **82 MiB** しかない
- つまりこのモデルでは、Liger を選ぶ理由が**速度にもメモリにも存在しない**

## なぜ負けたのか(推定)

Liger の FLCE は、LLaMA 級(dim 4096+、vocab 32k〜128k、長系列)で logits が数 GB 級になる状況を想定した設計だ。内部でチャンク化と再計算を行い、Triton カーネルで融合する。一方この自作モデルは:

- **dim=768**: GEMM が小さく、cuBLAS の素の行列積が十分速い領域。融合カーネルの起動・チャンク管理のオーバーヘッドが相対的に重い
- **4,096 トークン/step**: チャンク化の並列度を活かすには総トークン数が小さい
- **grouped MoE 導入済み**: ステップ全体が既に半分に縮んでおり、CE の再計算コストの相対比重が倍増していた(実際、自前 chunked CE のペナルティも grouped 化前の -11% から -21% へ拡大していた)

同じ現象は逆向きにも観測済みだ — MoE grouped GEMM は学習(4,096 行)で 1.99 倍だったが、batch-1 chat(数行)では 0.80 倍だった。**カーネルの損益分岐は形状で決まり、形状はモデルとワークロードで決まる**。

## 学び

1. **「定番だから入れる」は最適化では通用しない**。コミュニティで実績のあるカーネルでも、想定形状の外では逆効果になる。導入判断は必ず自分のモデル・自分のワークロード・自分の GPU での実測で行う。
2. **不採用も成果として記録する**。「Liger: 計測済み・数値合格・速度不合格・モデル拡大時に再評価」とロードマップに残せば、将来 dim や vocab を増やしたときに再評価すべき候補が明確になる。フラグと数値プローブは残してあるので、再評価は1コマンドで済む。
3. **代替案の存在が判断を明確にする**。依存ゼロの chunked CE が既にあったから、「メモリ目的でも Liger は不要」と言い切れた。外部依存を入れる前に、素朴な自前実装との差分を測る価値は大きい。
4. **最適化同士は相互作用する**。grouped MoE がステップを半減させた結果、CE 系最適化のペナルティは倍増した。ベンチは「今の本番構成」に揃えて取り直す必要がある。

## まとめ

| 項目 | 結果 |
|---|---|
| 数値正しさ | ◎ プローブ delta ≤ 0.002(bf16 水準)、意味論一致 |
| 速度 | ✗ full logits 比 -57.3%、自前 chunked 比 -46% |
| メモリ | △ 最小だが chunked との差 82 MiB |
| 採否 | **不採用**(通常: full logits / OOM: chunked / Liger: 拡大時に再評価) |

実装・ベンチハーネス・計測 JSON: [BushidoMythos](https://github.com/tedorigawa001/BushidoMythos)(`training/bench_act_compile.py`, `docs/performance_roadmap.md`)
