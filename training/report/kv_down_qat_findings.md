# kv_down ターゲット QAT — 成果レポート

## 要約

BushidoMythos(MLA)の単一層 `recurrent.block.attn.kv_down` を狙った
量子化対応学習(QAT)により、**kv_down が INT8 量子化のボトルネックである状態を解消**した。

- **発見(再現)**: フル INT8 量子化で finance PPL が **+45.4%** 悪化し、その劣化の
  **約 63% が kv_down 単一層**に起因(`recurrent.block.attn.kv_down` を fp32 に残すと
  +45.4% → +16.8% に回復)。
- **QAT の効果(本成果)**: kv_down だけを fake-quant 込みで再学習した結果、
  **QAT モデル内では full-INT8 と mixed-INT8(kv_down=fp32)がほぼ同値**になった
  (kv_down の量子化ペナルティ **+28.6pt → ≈0**)。すなわち **kv_down を INT8 にしても
  劣化が出ない重み**を獲得した。

---

## 実験設定

- **モデル**: `checkpoints/finance_a100_v2/phase5_final.pt`(MLA, 98.6M, max_loop_iters=8)
- **評価**: finance ドメイン(`financial_news_gpt2` cache)、非重複チャンク PPL、
  `eval_max_chunks=30`、`n_loops ∈ {1,2,4,8}`、INT8 dynamic(CPU 専用)
- **量子化**: `torch.quantization.quantize_dynamic({nn.Linear}, qint8)`
  (per-tensor 対称・zero_point=0・scale=max|w|/127)
- **QAT**: `training/qat_kv_down.py`
  - 対象: `kv_down`(weight/bias)のみ学習、他層は凍結
  - Fake-Quant: 評価器と同一(per-tensor 対称・scale=max|w|/127・STE)
  - quant_strength を 0→1 へ段階的に上げる(前半でフル量子化へ)
  - 損失: CE のみ(後述のとおり「ループ増幅」が観測されないため loop-aware 項は不採用)
  - ハイパラ: steps=1000, n_loops=8, batch_size=4, seq_len=1024, lr=2e-5,
    train=`finance_domain_mix_gpt2`
- **比較ツール**: `training/eval_qat_compare.py`(4 条件を同一プロトコルで PPL 比較)

---

## 結果

### 1) ベースモデル(QAT 前)

| 条件 | n_loops=1 | n_loops=2 | n_loops=4 | n_loops=8 | fp32比(代表 n=8) |
|---|---|---|---|---|---|
| fp32 (基準) | 51.53 | 46.09 | 43.80 | 43.53 | — |
| full-INT8 | 87.76 | 71.31 | 63.90 | 63.29 | **+45.4%** |
| mixed-INT8 (kv_down=fp32) | 61.50 | 55.28 | 51.57 | 50.85 | **+16.8%** |

- kv_down 単独の寄与 = 63.29 − 50.85 = **+12.44 PPL = 全 INT8 劣化の 63%**(1 層で 2/3)。

### 2) QAT モデル(QAT 後)— 自己基準

`--base_ckpt phase5_qat.pt` で QAT モデルの fp32 を基準に取り直したもの。

| 条件 | n_loops=1 | n_loops=2 | n_loops=4 | n_loops=8 | fp32比(代表 n=8) |
|---|---|---|---|---|---|
| fp32 (QAT モデル) | 54.01 | 41.12 | 38.10 | 37.78 | — |
| full-INT8 | 66.79 | 50.92 | 44.86 | 44.28 | **+17.2%** |
| mixed-INT8 (kv_down=fp32) | 64.76 | 48.89 | 45.21 | 44.88 | **+18.8%** |

### 3) kv_down の量子化ペナルティ(full-INT8 と mixed-INT8 の差)

| モデル | full-INT8 | mixed-INT8 | kv_down ペナルティ |
|---|---|---|---|
| ベース | +45.4% | +16.8% | **+28.6pt** |
| **QAT 後** | +17.2% | +18.8% | **≈ 0(−1.6pt, ノイズ内)** |

→ **QAT は kv_down の量子化ペナルティ(+28.6pt)をほぼ完全に消去**。QAT モデルでは
kv_down を INT8 にしても full-INT8 ≦ mixed-INT8 であり、ボトルネックが解消されている。

---

## 解釈と注意(交絡の分離)

QAT 仕上げには **2 つの独立した効果**が含まれることを明示する:

1. **量子化頑健性(本成果・成功)**
   同一モデル内で full-INT8 と mixed-INT8 が同値になった事実は、kv_down が
   「INT8 に丸めても出力が崩れない重み」へ移ったことを意味する。これは追加学習の
   有無に依らず、量子化器(quantize_dynamic)を学習ループに入れた直接の結果。

2. **fp32 PPL の低下(交絡・別物)**
   QAT の 1000 step 追加学習で QAT モデルの fp32 PPL が 43.53 → 37.78 に低下した。
   - このため「ベース fp32(43.53)」と「QAT 後 full-INT8(44.28)」を比べた当初の
     **「+1.7%」は見かけの値**(下がった QAT-fp32 ではなく古い基準と比較していた)。
     正しい量子化劣化は QAT モデル基準で **+17.2%**。
   - fp32 低下が汎化か overfit かは本レポートの量子化の主張とは独立。評価
     (`financial_news`)と学習(`finance_domain_mix`)の分布重なりがあり得るため、
     held-out finance での確認は今後の課題。

---

## 残課題

- **full-INT8 の残存劣化 +17.2%** は **他層の量子化**に由来(ベースの mixed 床 +16.8% と
  ほぼ一致)。kv_down のみ QAT したため当然の残差。フル INT8 を fp32 ロスレスへ
  近づけるには、次に効く層(先の ablation で示唆された attn.q / attn.wo など)へ
  QAT を拡張する必要がある。
- QAT による fp32 改善の真偽(汎化 vs overfit)を held-out finance で検証。

---

## 結論

**kv_down ターゲット QAT は、その設計目的「kv_down を INT8 量子化のボトルネックでなくする」
を達成した。** ベースで +28.6pt あった kv_down の量子化ペナルティは QAT 後にほぼ 0 となり、
kv_down を INT8 にしても劣化しない重みを獲得した。フル INT8 全体の劣化(+17.2%)は
他層に残るため、次段は他層への QAT 拡張で床を下げる。

### 再現コマンド

```bash
# QAT 仕上げ(GPU 推奨)
python3 training/qat_kv_down.py \
  --base_ckpt checkpoints/finance_a100_v2/phase5_final.pt \
  --train_cache finance_domain_mix_gpt2 \
  --steps 1000 --n_loops 8 --batch_size 4 --seq_len 1024 \
  --device cuda --out checkpoints/finance_a100_v2/phase5_qat.pt

# 4 条件比較(CPU — INT8 dynamic は CPU 専用)
python3 training/eval_qat_compare.py \
  --base_ckpt checkpoints/finance_a100_v2/phase5_final.pt \
  --qat_ckpt  checkpoints/finance_a100_v2/phase5_qat.pt \
  --eval_set finance --n_loops 1,2,4,8 --eval_max_chunks 30 --device cpu

# 交絡確認: QAT モデル自身の fp32 を基準にした量子化劣化
python3 training/eval_qat_compare.py \
  --base_ckpt checkpoints/finance_a100_v2/phase5_qat.pt \
  --eval_set finance --n_loops 1,2,4,8 --eval_max_chunks 30 --device cpu
```

---

# 第2段: 他層への拡張と第2ボトルネック

kv_down を直した後の残存量子化劣化(QAT モデル基準で約 +16〜17%)を下げるべく、QAT を
他層へ拡張した。結論を先に: **第2ボトルネックも「1層集中」型(shared_experts)** であり、
**整合正則化で recurrent 多層 QAT の破壊を防げる**。ただし床の低下は逓減する。

## 群別感度 ablation（次の犯人を特定）

`exp_quantize_ablation.py`（phase5 / finance）で `INT8 except G`（G だけ fp32 温存）の
回復幅を測り、量子化に弱い群をランク付けした:

| group | 回復幅 | 全劣化に占める |
|---|---|---|
| attn | 大 | **70%**（うち kv_down 63%）|
| experts (MoE) | 中 | 27% |
| ffn_dense | 中 | 20% |
| head | 小 | 4% |
| router | 小 | 1% |

さらに experts を射影タイプ別へ細分化（INT8 except subgroup, 12 chunks, INT8 all=100.43）:

| keep fp32 する subgroup | 回復幅 | 解釈 |
|---|---|---|
| **shared_experts（3層）** | **8.67** | MoE 劣化の大半が 3 層に集中 |
| ffn_dense（prelude/coda 6層）| 6.38 | 非ループの dense FFN |
| experts.down（28層）| 1.03 | ほぼ無害 |
| experts.gate（28層）| 0.77 | ほぼ無害 |
| experts.up（28層）| −0.05 | 無害 |

→ **routed experts 84層（53M）は INT8 に頑健**。第2ボトルネックは **shared_experts（3層）+
ffn_dense（6層）= 計9層・約8M** に集中。kv_down と同じ「サイズ≠感度・1層集中」構図。

## 実験と結果（full-INT8 vs 各モデル自身の fp32, n_loops=8）

| 実験 | QAT 対象 | lr / 正則化 | fp32 が depth で | full-INT8 | 判定 |
|---|---|---|---|---|---|
| base | なし | — | 改善（51.5→43.5）| +45.4% | — |
| kv_down | kv_down 1層 | 2e-5 / なし | 改善 | +17.2% | 成功 |
| Exp A | recurrent.block.attn 6層 | 2e-5 / なし | 改善（45.9→30.0）| +16.4% | 健全 |
| Exp B | attn+experts+ffn ~84層/60M | 2e-5 / なし | **悪化（59.9→77.4）** | — | **破壊** |
| Exp C | attn+shared_experts+pre/coda ffn 15層/9M | 1e-5 / **λ=1.0** | 改善（42.1→30.9）| **+13.5%** | 健全 |

- **Exp B（wide, 正則化なし）は破壊**: fp32 が depth で悪化＝ループ不動点の崩壊。
  量子化に頑健な 53M の routed experts まで lr=2e-5 で学習したため。
- **Exp C（ターゲット9層 + 整合正則化 λ=1.0）は健全**: 同じ recurrent 多層（shared_experts
  含む）を触っても、凍結 base のループ出力への KL 整合でループが保持された。床も
  +16.4% → +13.5% へ低下（量子化ギャップ 4.92 → 4.15 PPL）。
- すべての QAT モデルで **full-INT8 ≦ mixed-INT8(kv_down=fp32)**: kv_down は量子化
  ネイティブになり、fp32 に戻す方がむしろ悪い。

## 交絡（fp32 改善）の検証 — overfit ではない

Exp A の fp32 PPL は finance で 43.5→30.0 と大きく改善したが、**held-out の WikiText でも
~370→258.79 と同程度に改善**。finance 固有の overfit ではなく汎化（`finance_domain_mix` に
一般テキストも含まれ、追加学習が attention を全般的に改善したと解釈）。量子化頑健性の
結論（full ≦ mixed）はこの交絡とは独立。

## 第2段の結論

1. **第2ボトルネックも 1 層集中**（shared_experts 3層）。routed experts 84層は量子化頑健で、
   QAT 対象にすべきでない（触ると壊れるだけ＝Exp B）。
2. **整合正則化（--consistency_lambda）が recurrent 多層 QAT の破壊を防ぐ**。Exp B（破壊）
   → Exp C（健全）の差はこの一点。
3. **床の低下は逓減**: +45.4%(base) → +17.2%(kv_down) → +16.4%(attn) → +13.5%(9層)。
   残る +13.5% は頑健・拡散した層（routed experts ~0% / head 4% / router 1%）由来で、
   これらの QAT は head 38.6M・routed 53M と高コスト低リターン。**+13.5% を実用上の床**とする。

### 再現コマンド（第2段）

```bash
# 群別 ablation（次の犯人の特定）
python3 training/exp_quantize_ablation.py \
  --ckpt checkpoints/finance_a100_v2/phase5_final.pt --eval_max_chunks 15

# Exp C: 第2ボトルネックも含めた 15 層ターゲット QAT（整合正則化つき）
python3 training/qat_kv_down.py \
  --base_ckpt checkpoints/finance_a100_v2/phase5_final.pt \
  --targets recurrent.block.attn,shared_experts,prelude.0.ffn,coda.0.ffn \
  --consistency_lambda 1.0 --lr 1e-5 \
  --steps 1500 --n_loops 8 --batch_size 4 --seq_len 1024 --device cuda \
  --out checkpoints/finance_a100_v2/phase5_qat_floor.pt

# 破壊チェック（fp32 が depth で改善するか）+ 量子化劣化
python3 training/eval_qat_compare.py \
  --base_ckpt checkpoints/finance_a100_v2/phase5_qat_floor.pt \
  --eval_set finance --n_loops 1,2,4,8 --eval_max_chunks 30 --device cpu
```
