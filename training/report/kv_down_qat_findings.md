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
- **QAT**: `training/experiments/qat_kv_down.py`
  - 対象: `kv_down`(weight/bias)のみ学習、他層は凍結
  - Fake-Quant: 評価器と同一(per-tensor 対称・scale=max|w|/127・STE)
  - quant_strength を 0→1 へ段階的に上げる(前半でフル量子化へ)
  - 損失: CE のみ(後述のとおり「ループ増幅」が観測されないため loop-aware 項は不採用)
  - ハイパラ: steps=1000, n_loops=8, batch_size=4, seq_len=1024, lr=2e-5,
    train=`finance_domain_mix_gpt2`
- **比較ツール**: `training/experiments/eval_qat_compare.py`(4 条件を同一プロトコルで PPL 比較)

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
python3 training/experiments/qat_kv_down.py \
  --base_ckpt checkpoints/finance_a100_v2/phase5_final.pt \
  --train_cache finance_domain_mix_gpt2 \
  --steps 1000 --n_loops 8 --batch_size 4 --seq_len 1024 \
  --device cuda --out checkpoints/finance_a100_v2/phase5_qat.pt

# 4 条件比較(CPU — INT8 dynamic は CPU 専用)
python3 training/experiments/eval_qat_compare.py \
  --base_ckpt checkpoints/finance_a100_v2/phase5_final.pt \
  --qat_ckpt  checkpoints/finance_a100_v2/phase5_qat.pt \
  --eval_set finance --n_loops 1,2,4,8 --eval_max_chunks 30 --device cpu

# 交絡確認: QAT モデル自身の fp32 を基準にした量子化劣化
python3 training/experiments/eval_qat_compare.py \
  --base_ckpt checkpoints/finance_a100_v2/phase5_qat.pt \
  --eval_set finance --n_loops 1,2,4,8 --eval_max_chunks 30 --device cpu
```
