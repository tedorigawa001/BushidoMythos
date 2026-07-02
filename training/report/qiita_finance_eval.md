# 99Mパラメータの金融特化LMは「何を学び、何を学べなかったか」を定量評価してみた

## TL;DR

- 自作の小型 Recurrent-Depth Transformer（**98.6M params**）を、汎用言語フェーズ（Phase 1）と金融特化フェーズ（Phase 5）で比較した。
- **WikiText-103 perplexity だけでは「金融特化が効いたか」は判断できない**（むしろ PPL は悪化する）。
- そこで固定プロンプトへの生成出力を 3 系統（フォーマット追従・推論構造・リスク言及）で定量化した。
- 結果：**文体・トーン・不確実性の表明は獲得（structured-reasoning 12%→75%、uncertainty 25%→50%）。一方で具体的なリスク手順はまだ弱い**。
- **「フォーマット追従 0%」は誤りだった** — 生成関数が EOS を無視するバグ＋指標の誤りで、直すと Phase 5 は 38% に（後述）。
- 教訓：**0% という極端な数値は、能力不足より測定・生成バグを疑え。** ドメイン適応は register には効くが、具体能力と確実性はデータ量に律速される。

:::note info
対象読者：小規模 LM のドメイン適応・ファインチューニングの「効き目」をどう測るか興味がある方。モデルは研究用で投資助言ではありません。
:::

---

## 背景：評価対象のモデル

| 項目 | 値 |
|---|---|
| アーキテクチャ | Recurrent-Depth Transformer（Prelude → 再帰ブロック×ループ → Coda、MoE FFN、ACT 停止） |
| パラメータ数 | 98.6M |
| 隠れ次元 dim | 768 |
| 語彙 | GPT-2 互換（50,257） |
| 推論ループ数 | 8 |

5 段階で学習した：

| フェーズ | データ | 目的 |
|---|---|---|
| Phase 1 | WikiText-103 | 一般言語 |
| Phase 2 | OpenWebMath + Orca Math | 定量推論 |
| Phase 3 | 金融ニュース + finance-alpaca | 金融語彙・指示形式 |
| Phase 4 | FinGPT forecaster / sentiment | トレード手法 SFT |
| Phase 5 | FinGPT FIQA QA | リスク管理 QA |

本記事では **Phase 1（汎用）** と **Phase 5（金融特化の最終形）** を比較する。

---

## 評価1：WikiText-103 perplexity の限界

まず汎用ベンチで perplexity（PPL、低いほど良い）を測った。

| モデル | パラメータ | PPL ↓ |
|---|---|---|
| GPT-2 small | 117M | 29.41 |
| GPT-2 medium | 345M | 22.76 |
| GPT-2 large | 762M | 19.93 |
| GPT-2 XL | 1.5B | 17.48 |
| **本モデル（Phase 1）** | **99M** | **54.86** |
| **本モデル（Phase 5）** | **99M** | **361.48** |

:::note warn
GPT-2 ベースラインは test-set PPL（Radford et al. 2019）。トークナイザ・stride・前処理が異なるため **rough reference**（厳密な同条件比較ではない）として扱うこと。
:::

Phase 5 の PPL が大幅に高いが、これは**劣化ではなくドメイン適応のコスト**。金融に特化して汎用テキスト（WikiText）の分布から離れただけだ。つまり**汎用ベンチの損失では金融特化の良し悪しは測れない**。挙動そのものを見る必要がある。

---

## 評価2：固定プロンプトで挙動を定量化

8 個の金融プロンプト（`high leverage risk` / `position sizing` / `Fed rate and inflation` など）を `### Instruction: / ### Response:` 形式で与え、生成を 3 系統で集計した。

```bash
# 再現コマンド（再現性のため CPU・決定的実行）
python3 training/eval_finance_behavior.py --device cpu \
  --ckpts checkpoints/finance_a100_v2/phase1_final.pt \
          checkpoints/finance_a100_v2/phase5_final.pt
```

### 結果サマリ

| 指標 | Phase 1 | Phase 5 |
|---|---|---|
| フォーマット追従（EOS で終端） | 0% | **38%** |
| structured-reasoning rate | 12% | **75%** |
| non-degenerate rate | 100% | 100% |
| avg risk concepts (0–6) | 0.25 | **0.62** |
| repetition rate ↓ | 0.07 | **0.02** |

> ※ 当初「フォーマット追従」は `### ` 境界で測り両者 0% だったが、これは**生成側のバグ**（後述）で、実際には Phase 5 は 38% で正しく EOS 終端していた。上表は修正版。

### リスク概念の被覆

| 概念 | Phase 1 | Phase 5 |
|---|---|---|
| uncertainty（不確実性） | 25% | **50%** |
| event_risk（イベントリスク） | 0% | 12% |
| stop_loss（損切り） | 0% | 0% |
| position_sizing（ポジションサイズ） | 0% | 0% |
| leverage（レバレッジ） | 0% | 0% |
| liquidity（流動性） | 0% | 0% |

---

## 効いたこと：文体・推論構造・不確実性

同じプロンプト `high leverage risk` への応答：

> **Phase 1:** *"The system's main feature of the decking system, which has been described as the biggest single-tier system in Asia to handle most major issues on construction projects worldwide..."*

> **Phase 5:** *"I'm not sure what the average person is in this situation; it's just a probability distribution... The potential for such risks depends how much money are involved on your side of the trade or whether they experience volatility changes over time..."*

- **文体の移行**：Phase 1 は Wikipedia 風の百科事典文で指示を無視。Phase 5 は market / trade / volatility / probability を一人称で語る金融会話に移行。
- **構造化推論 12% → 75%**：`because` / `however` / `should` 等の構造語を伴う「整理された応答」が激増。
- **不確実性の表明 25% → 50%**：ヘッジ表現が増加。Phase 5 の学習目標（不確実性の開示）と一致。
- 繰り返し率も 0.07 → 0.02 と低下し、出力がクリーンに。

## まだ弱いこと：具体的リスク手順とフォーマットの確実性

- **stop-loss / position-sizing / leverage / liquidity の具体語は 0%**。市場を「語る」が、損切り位置やサイズ計算を「提示」しない。
- **フォーマット追従（EOS 終端）は 38%**。半分以上は終端できず冗長に続く。「未学習」ではなく「確信度が弱い」状態。

---

## 追記：「フォーマット追従 0%」の正体は生成バグだった

当初この評価で**フォーマット追従が両者 0%** と出て「形式を学習できていない」と結論しかけた。だが原因を追うと、モデル側ではなく**評価・生成側の問題**だった。

1. **SFT データは正しかった**：各応答の末尾に `<|endoftext|>`（EOS, 単一トークン 50256）を付け、loss マスクも EOS にかかっていた。つまりモデルは「終端を出す」よう学習していた。
2. **生成関数が EOS を無視していた**：`generate()` が `for step in range(max_new_tokens)` で固定回数回し、**EOS が出ても停止しなかった**。だから全出力が上限まで冗長に続いた。
3. **指標も誤っていた**：`### ` 境界を探していたが、モデルは `### ` でなく EOS で終端するよう学習していた（しかも EOS は `skip_special_tokens` で消える）。

→ `generate()` に EOS 停止を実装したところ、**Phase 5 のフォーマット追従は 0% → 38%** に。早期終端で平均語数も 84 → 77 に短縮した。

残る 62% の未終端は、Phase 4/5 の SFT データが極端に小さい（合計 ~12M トークンを多数エポック反復）ことによる**終端確信度の弱さ**で、これは**データ量の問題**。スケールではなくデータ拡充で押し上げる対象だ。

> 教訓：「モデルが学習できていない」と断じる前に、**評価系と生成系のバグを疑う**。0% という極端な数値は、能力不足よりも測定ミス・実装ミスのサインであることが多い。

---

## 結論：規模に律速される能力の境界

| 観点 | 評価 |
|---|---|
| ドメイン文体の移行 | ◎ 明確 |
| 不確実性の表明 | ◎ 25% → 50% |
| 構造化推論 | ◎ 12% → 75% |
| 指示応答フォーマット追従 | △ 38%（生成バグ修正後・データ律速） |
| 具体的リスク用語 | △ ほぼ未獲得 |

99M というサイズで、**文体・トーン・不確実性の意識**は獲得できた。フォーマット追従も（生成バグを直せば）部分的に効いている。だが**具体的リスク管理の実行**と**終端の確実性**は、まだ弱い。

**ドメイン適応は安価な側面（register）には効くが、具体能力と確実性はデータ量とパラメータ数に律速される** —— これが再現性のある知見だ。そして **0% という数値は、能力不足ではなく測定・生成バグのサインだった**。

:::note alert
注意：n=8・キーワード一致（言及 ≠ 正しさ）の rough な指標。また Apple MPS はサンプリングが run 間で非再現のため、数値は CPU（決定的）で取得した。
:::

---

## 環境・補足

- 評価はローカル（Apple Silicon, 16GB）で実行。perplexity は CPU で約13分、挙動評価は数分。
- 評価スクリプトはチェックポイントを `weights_only=True` で安全にロードし、出力 token ID を `[0, vocab_size-1]` に clamp する。
- 指標の詳細・全プロンプト出力はリポジトリのレポートに保存。

---

> 推奨タグ：`機械学習` `LLM` `PyTorch` `自然言語処理` `深層学習`
