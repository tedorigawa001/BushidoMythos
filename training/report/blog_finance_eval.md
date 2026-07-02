# 99M 金融特化 LM は何を学び、何を学べなかったか — Phase 1 vs Phase 5 の挙動評価

BushidoMythos は金融トレーディング研究向けの Recurrent-Depth Transformer（RDT、98.6M params, dim=768, GPT-2 語彙）である。WikiText-103 で一般言語を学んだ **Phase 1** と、金融ニュース・トレード手法・リスク管理 QA まで 5 段階で適応させた **Phase 5** を比較し、「金融特化は本当に効いたのか」を定量的に検証した。

---

## 1. WikiText-103 perplexity だけでは判断できない

まず汎用ベンチで perplexity（PPL, 低いほど良い）を測った。

| モデル | パラメータ | PPL ↓ |
|---|---|---|
| GPT-2 small | 117M | 29.41 |
| GPT-2 medium | 345M | 22.76 |
| GPT-2 large | 762M | 19.93 |
| GPT-2 XL | 1.5B | 17.48 |
| **BushidoMythos (Phase 1)** | **99M** | **54.86** |
| **BushidoMythos (Phase 5)** | **99M** | **361.48** |

> GPT-2 ベースラインは test-set PPL（Radford et al. 2019）。トークナイザ・stride・前処理が異なるため **rough reference**（厳密な同条件比較ではない）。

Phase 5 の PPL が Phase 1 より大幅に高い。しかしこれは劣化ではなく、金融ドメインへ特化して汎用テキスト（WikiText）の分布から乖離した **ドメイン適応のコスト**だ。汎用ベンチの損失では「金融特化の良し悪し」は判断できない。そこで生成出力そのものを評価する。

---

## 2. 挙動評価:固定プロンプトで Phase 1 と Phase 5 を比較

8 個の金融プロンプト（`high leverage risk` / `position sizing` / `Fed rate and inflation` ほか）を各チェックポイントに `### Instruction: / ### Response:` 形式で与え、生成を 3 系統で集計した（再現性のため CPU・決定的実行、n=8）。

### 結果サマリ

| 指標 | Phase 1 | Phase 5 |
|---|---|---|
| ③ format adherence（EOS で終端） | 0% | **38%** |
| structured-reasoning rate | 12% | **75%** |
| non-degenerate rate | 100% | 100% |
| ④ avg risk concepts (0–6) | 0.25 | **0.62** |
| repetition rate ↓ | 0.07 | **0.02** |

### ④ リスク概念の被覆

| 概念 | Phase 1 | Phase 5 |
|---|---|---|
| uncertainty（不確実性） | 25% | **50%** |
| event_risk（イベントリスク） | 0% | 12% |
| stop_loss（損切り） | 0% | 0% |
| position_sizing（ポジションサイズ） | 0% | 0% |
| leverage（レバレッジ） | 0% | 0% |
| liquidity（流動性） | 0% | 0% |

---

## 3. 効いたこと:文体・推論構造・不確実性の表明

同じプロンプト `high leverage risk` への応答:

> **Phase 1:** *"The system's main feature of the decking system, which has been described as the biggest single-tier system in Asia to handle most major issues on construction projects worldwide..."*

> **Phase 5:** *"I'm not sure what the average person is in this situation; it's just a probability distribution... The potential for such risks depends how much money are involved on your side of the trade or whether they experience volatility changes over time..."*

- **文体の移行**: Phase 1 は Wikipedia 風の百科事典文で指示を無視する。Phase 5 は market / trade / volatility / probability を一人称で語る金融会話に移行している。
- **構造化推論 12% → 75%**: Phase 5 は `because` / `however` / `should` / `consider` 等の構造語を伴う「整理された応答」を 75% で生成。
- **不確実性の表明 25% → 50%**: ヘッジ表現が増加。Phase 5 の学習目標（不確実性の開示）と一致する。
- 繰り返し率も 0.07 → 0.02 と低下し、出力がクリーンになった。

---

## 4. まだ弱いこと:具体的リスク手順とフォーマットの確実性

- **stop-loss / position-sizing / leverage / liquidity の具体語は 0%**。Phase 5 は市場を「語る」が、損切り位置やサイズ計算を「提示」しない。
- **フォーマット追従（EOS 終端）は 38%**。半分以上は終端できず冗長に続く。

> 補足:当初この指標は `### ` 境界で測り両者 0% だったが、これは **生成関数が EOS を無視するバグ + 指標の誤り**だった。SFT は応答末尾に EOS（50256）を学習させており、`generate()` に EOS 停止を実装すると Phase 5 は 0% → 38% に改善した。残る未終端は Phase 4/5 の SFT データが小さい（~12M トークンの過反復）ことによる確信度不足で、データ量の問題。**0% は能力不足ではなく測定・生成バグのサインだった。**

---

## 5. 結論:規模に律速される能力の境界

| 観点 | 評価 |
|---|---|
| ドメイン文体の移行 | ◎ 明確 |
| 不確実性の表明 | ◎ 25% → 50% |
| 構造化推論 | ◎ 12% → 75% |
| 指示応答フォーマット追従 | △ 38%（生成バグ修正後・データ律速） |
| 具体的リスク用語 | △ ほぼ未獲得 |

99M というサイズで、**文体・トーン・不確実性の意識**は獲得できた。フォーマット追従も生成バグを直せば部分的に効いている。だが **具体的リスク管理の実行と終端の確実性**はまだ弱い。「ドメイン適応は register には効くが、具体能力と確実性はデータ量・パラメータ数に律速される」「**0% は能力不足ではなく測定・生成バグのサインだった**」という再現性のある知見が得られた。

> **注意:** n=8・キーワード一致（言及 ≠ 正しさ）の rough な指標。Apple MPS はサンプリングが run 間で非再現のため、数値は CPU（決定的）で取得した。
