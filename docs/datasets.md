# Dataset Plan for Financial-Trading Specialization

BushidoMythos should be trained in stages. General language data gives the model fluency; financial and trading data gives it domain grounding; curated decision data teaches discipline, risk framing, and post-trade reasoning.

This document is a data strategy, not an endorsement of any live trading decision. Every dataset used for market work should be checked for licensing, timestamp integrity, survivorship bias, look-ahead leakage, and redistribution rights.

---

## Training Stages

| Stage | Purpose | Example data | Notes |
|---|---|---|---|
| General language pretraining | Basic language, instruction following, broad world knowledge | FineWeb-Edu, high-quality instruction data, math/reasoning text | Useful foundation, but not sufficient for trading specialization |
| Financial language adaptation | Market vocabulary and document structure | filings, earnings-call transcripts, annual reports, macro commentary, central-bank statements | Preserve timestamps so the model does not learn future information as past context |
| Market reasoning specialization | Trading-specific analysis and scenario planning | strategy notes, trade theses, risk/reward writeups, setup reviews, market-regime labels | Should emphasize uncertainty, invalidation, and position sizing rather than prediction certainty |
| Behavioral discipline tuning | Bushido-inspired decision quality | post-trade journals, rule violations, revenge-trading examples, risk checklists | Teaches restraint, patience, loss acceptance, and process consistency |
| Evaluation-only data | Honest measurement | held-out market periods, unseen instruments, unseen regimes | Never mix into training |

---

## Recommended Data Categories

### Market Text

- financial news and market commentary
- earnings summaries and guidance updates
- macroeconomic release explanations
- central-bank speeches and policy statements
- sector and asset-class research notes

Use these to teach the model how traders describe catalysts, flows, volatility, and regime changes.

### Structured Financial Documents

- company filings and annual reports
- earnings-call transcripts
- analyst-style financial summaries
- economic calendars and release descriptions

These help the model parse dense financial language. Keep source dates and publication times attached to each sample.

### Trading Process Data

- trade plans with entry, stop, target, invalidation, and sizing
- post-trade reviews and journal entries
- examples of good and bad risk management
- scenario trees and pre-mortems
- strategy rules and failure cases

This is the most important layer for the intended Bushido concept. The goal is not just market knowledge; it is disciplined decision reasoning.

### Numeric and Time-Series Context

The current model is a language model, so raw OHLCV should usually be converted into text or compact structured prompts before training.

Useful representations:

- trend, range, volatility, and breakout summaries
- indicator states described as text
- multi-timeframe context summaries
- event-window summaries around earnings, CPI, FOMC, or market open/close

Avoid training on future-derived labels unless the prompt makes the prediction horizon explicit.

---

## Leakage and Bias Controls

Financial datasets are unusually easy to contaminate. Before training, check:

- **Look-ahead leakage:** the sample must not include information unavailable at the decision time.
- **Timestamp leakage:** publication time matters, not just publication date.
- **Survivorship bias:** delisted assets and failed strategies should not disappear from the corpus.
- **Selection bias:** only collecting winning trades teaches unrealistic confidence.
- **Label leakage:** labels such as "big winner" or "crash before earnings" should not appear in the prompt when predicting.
- **Regime balance:** include trend, range, crisis, low-volatility, high-volatility, and liquidity-stressed periods.

---

## Token Budget Guidance

Start small and evaluate before scaling. For laptop development, use `mythos_tiny` and short sequence lengths. For serious domain adaptation, prefer a staged budget:

| Stage | Pilot | Serious run |
|---|---:|---:|
| General language | 10M-100M tokens | 1B+ tokens |
| Financial language | 5M-50M tokens | 500M+ tokens |
| Trading process / journals | 100k-5M tokens | 10M+ curated tokens |
| Evaluation holdout | fixed, never trained | fixed, never trained |

Quality matters more than size for the trading-process stage. A smaller set of clean, timestamped, well-labeled trade reviews is more valuable than a large noisy corpus.

---

## Evaluation Sets

A trading-specialized model should be evaluated on more than language loss.

Recommended eval buckets:

- market regime classification from text summaries
- catalyst extraction from news or filings
- risk/reward critique of a proposed trade
- detection of missing invalidation or oversized risk
- scenario planning under bullish, bearish, and neutral paths
- post-trade review quality
- refusal or caution when information is stale, insufficient, or outside the model's context

For live market use, pair the model with retrieval or tools for current prices, calendar events, and filings. The model itself should not be treated as a source of current market truth.

---

## 武士道 SFT データセット設計書

Phase 4 / Phase 5 に組み込む独自 SFT データの仕様。教師モデル蒸留・手動作成・トレード日誌変換の3経路で収集する。

### 基本フォーマット

すべてのサンプルは Phase 3 以降の学習形式に統一する。

```
### Instruction:
[マーケットコンテキスト（任意）]
[具体的な質問・シナリオ]

### Response:
[構造化された回答]
```

**Response の内部構造（推奨）：**

```
[状況認識] 現在の相場環境・セットアップの客観的評価
[リスク評価] 損切り水準・最大損失・無効化条件
[行動提案] エントリー可否・ポジションサイズ・優先シナリオ
[不確実性] 判断の根拠が弱い箇所・見落としうるリスク
```

不確実性の明示は必須。「確実に上がる」「必ず損切りできる」などの断定表現は品質違反とする。

---

### シナリオ分類と件数目標

#### カテゴリ A — エントリー判断（go / no-go）

**目的:** セットアップの強度評価と「やらない理由」の言語化。ルール違反のエントリーを断るサンプルを多く含める。

| サブカテゴリ | 内容 | 目標件数 |
|---|---|---|
| A-1 順張りセットアップ | トレンドフォロー・ブレイクアウト判断 | 400 |
| A-2 逆張りセットアップ | 支持線・抵抗線・RSI 逆張り判断 | 300 |
| A-3 スキップ判断 | 条件不足・低確率でエントリーしない | 400 |
| A-4 イベント前後 | 決算・FOMC・雇用統計前後の判断 | 200 |

**A-3 スキップ判断は全体の 30% 以上を占めること。** 「やる理由」だけ学んだモデルはオーバートレードする。

サンプル:
```
### Instruction:
USD/JPY 日足: 157.30 付近に強い抵抗線。現在 157.25。RSI 72（過熱）。
明日 FOMC 議事録公開予定。ロングエントリーを検討中。

### Response:
[状況認識] 抵抗線直下かつ RSI 過熱域。イベントリスクが翌日に控えている。
[リスク評価] 抵抗線上抜け失敗なら 156.50 まで下落余地あり。FOMC でボラティリティ急増の可能性。
[行動提案] エントリー見送り。FOMC 通過後に RSI が落ち着いてから改めてセットアップを確認する。
[不確実性] 抵抗線を強くブレイクするシナリオは排除できないが、現時点でリスクリワードが成立しない。
```

---

#### カテゴリ B — リスク管理・ポジションサイジング

**目的:** 感情ではなくルールに基づくサイズ計算と損切り設定の徹底。

| サブカテゴリ | 内容 | 目標件数 |
|---|---|---|
| B-1 ポジションサイズ計算 | 資産・リスク許容度・損切り幅からサイズを導出 | 300 |
| B-2 損切り水準の設定 | 技術的根拠のある損切り位置の選択 | 300 |
| B-3 部分利確・スケールアウト | 目標到達前の部分決済判断 | 200 |
| B-4 ナンピン拒否 | 含み損への追加ポジションを断る | 150 |

サンプル（B-4）:
```
### Instruction:
NI225 先物 ロング 38,000 で 2 枚保有。現在 37,600 (-400)。
「平均単価を下げるためにもう 2 枚買い増したい」という考えが浮かんでいる。

### Response:
[状況認識] 現在 400 円の含み損。当初の想定外の動きをしている。
[リスク評価] 買い増しにより損失が倍増するリスク。相場の方向性が変わった可能性を無視している。
[行動提案] 買い増し禁止。当初の損切り水準（例: 37,400）を守る。損切り後にセットアップを再評価する。
[不確実性] 一時的な押しである可能性はあるが、ナンピンは損失を管理不能にする行為であり、武士道の規律に反する。
```

---

#### カテゴリ C — 相場環境分析（マクロ・レジーム）

**目的:** RAG で注入される外部データを解釈する能力の育成。

| サブカテゴリ | 内容 | 目標件数 |
|---|---|---|
| C-1 トレンド vs レンジ判断 | 現在のレジームを複数の根拠で分類 | 200 |
| C-2 マクロ指標の解釈 | CPI・雇用・GDP などの数値から市場への影響を推論 | 200 |
| C-3 センチメント評価 | VIX・Fear&Greed・Put/Call Ratio の読み方 | 150 |
| C-4 クロスアセット分析 | 株・債券・為替・商品の相互関係 | 150 |

---

#### カテゴリ D — トレード日誌レビュー

**目的:** 事後評価の習慣化。良い結果と悪いプロセスを分離する。

| サブカテゴリ | 内容 | 目標件数 |
|---|---|---|
| D-1 勝ちトレードのレビュー | プロセスの正当性評価（結果だけで判断しない） | 200 |
| D-2 負けトレードのレビュー | ルール遵守 vs ルール違反の分類と改善案 | 300 |
| D-3 ルール違反の分析 | 感情的判断・衝動的オーバートレードの解剖 | 200 |

**D-1「勝ちトレードのレビュー」にはワーストケース指摘を含めること。** 結果が良くてもプロセスが悪いサンプルは最重要。

サンプル（D-3）:
```
### Instruction:
今日 3 連敗した後、4 回目のエントリーを衝動的に行い、さらに損失を出した。
計画にないトレードだった。この行動を振り返ってほしい。

### Response:
[状況認識] 連続損失後の感情的報復トレード（リベンジトレード）。計画外のエントリーは戦略の外側にある。
[リスク評価] 感情が高ぶった状態では判断精度が下がる。損失の累積が加速するパターン。
[行動提案] 3 連敗後は当日のトレードを終了するルールを設定する。翌日、冷静な状態で原因を分析する。
[不確実性] 4 回目が利益になる可能性はあったが、それは結果論。プロセスの問題は変わらない。武士道は結果ではなく行動の規律を問う。
```

---

#### カテゴリ E — シナリオ計画・プリモータム

**目的:** 最悪ケースを事前に言語化し、感情的な意思決定を防ぐ。

| サブカテゴリ | 内容 | 目標件数 |
|---|---|---|
| E-1 プリモータム | 「このトレードが失敗した理由は何か」を事前に列挙 | 200 |
| E-2 複数シナリオ計画 | 強気・弱気・中立シナリオと各々の対応 | 200 |
| E-3 ブラックスワン対応 | 急激なギャップや流動性危機への対処 | 100 |

---

### 件数サマリと導入フェーズ

| カテゴリ | パイロット | 本番目標 | 導入フェーズ |
|---|---:|---:|---|
| A エントリー判断 | 130 | 1,300 | Phase 4 |
| B リスク管理 | 95 | 950 | Phase 4 |
| C 相場環境分析 | 70 | 700 | Phase 4 |
| D トレード日誌 | 70 | 700 | Phase 5 |
| E シナリオ計画 | 50 | 500 | Phase 5 |
| **合計** | **415** | **4,150** | |
| 評価用 holdout（学習に使わない） | 85 | 850 | — |

パイロット 500 件で Phase 4/5 を実行し、PPL と人手評価で品質を確認してから本番スケールに移行する。

**holdout の分割ルール（重要）：** トレードデータは同一期間・同一銘柄・同一シナリオテンプレートが train/eval に混在すると評価が甘くなる。以下の3軸で分割する。

| 分割軸 | ルール | 理由 |
|---|---|---|
| 時間軸 (time-based) | `published_at` で昇順ソートし、最も新しい 20% を holdout に固定する | 同一相場環境で評価すると汎化を過大評価する |
| 銘柄軸 (instrument-based) | ゴールド・原油・EUR/USD など 2〜3 銘柄を holdout 専用に予約する | train で見た銘柄だけ正解するモデルを検出する |
| テンプレート軸 (scenario-template) | `scenario_template_id` を管理し、holdout 用テンプレート ID は train に流用しない | 問い文のパラフレーズを暗記しているだけのモデルを検出する |

実装: JSONL の `split` / `holdout_reason` / `scenario_template_id` フィールドで管理する。

---

### データ収集の3経路

#### 経路 1 — 教師モデル蒸留（主力）

GPT-4o / Claude Sonnet にシナリオを渡して回答を生成し、フォーマットに変換する。

```python
# 生成スクリプトのイメージ
system_prompt = """
あなたはプロのトレーダーです。以下のルールで回答してください：
- 状況認識 / リスク評価 / 行動提案 / 不確実性 の4セクション構成
- 断定表現を使わない。「可能性がある」「想定される」を使う
- 損切りとポジションサイズは常に明示する
- リベンジトレードとナンピンは常に否定する
"""
```

生成後に人手レビューで品質フィルタリング（不確実性明示なし・断定表現あり → 除外）。

#### 経路 2 — 手動作成（高品質・少量）

実際のトレード経験や書籍（『マーケットの魔術師』等）から典型シナリオを設計。
1 件あたりの品質が最も高い。D カテゴリに向いている。

#### 経路 3 — 既存文書変換

許諾済み・自作・ライセンスが明確なもののみ使用する。具体的には以下の条件をすべて満たすこと。

- **使用許諾:** 著者から明示的な許諾を得ているか、CC BY 等の再利用可能ライセンスが付与されている
- **個人情報の除去:** 口座番号・証券会社名・実名・スクリーンショット由来の固有取引情報は変換前に削除する
- **商業利用の確認:** note・Substack の有料コンテンツは無許諾でのスクレイピング禁止
- **ライセンスが不明な場合:** 経路 1（教師モデル蒸留）で同等のシナリオを代替生成し、元文書は使用しない

---

### 品質基準（除外条件）

以下に該当するサンプルはデータセットから除外する。

| 違反パターン | 例 |
|---|---|
| 断定表現 | 「必ず上がる」「確実に損切りできる」 |
| 不確実性セクションの欠落 | 4 セクション構成が崩れている |
| 損切り水準の未記載 | リスク管理サンプルで損切りに言及なし |
| ナンピン・リベンジを肯定 | 「平均単価を下げれば大丈夫」 |
| 未来情報の混入 | 「この後 XX 円まで上がった」など事後情報 |
| 特定銘柄への投資推奨 | 「〇〇株を買うべき」 |

---

### ファイル構成（予定）

```
data/
├── bushido_sft/
│   ├── raw/            # 生成・収集した原文（未フィルタ）
│   ├── reviewed/       # 人手レビュー済み JSONL
│   │   ├── phase4_entry.jsonl
│   │   ├── phase4_risk.jsonl
│   │   ├── phase4_regime.jsonl
│   │   ├── phase5_journal.jsonl
│   │   └── phase5_scenario.jsonl
│   └── holdout/        # 評価専用（学習に使わない）
│       └── eval.jsonl
```

JSONL フォーマット（1 行 1 サンプル）：

```json
{
  "instruction": "USD/JPY 日足: 157.30 付近に強い抵抗線...",
  "input": "",
  "output": "[状況認識] ...\n[リスク評価] ...\n[行動提案] ...\n[不確実性] ...",
  "category": "A-3",
  "scenario_template_id": "A3-usdjpy-resistance-skip-v1",
  "source": "distilled_gpt4o",
  "source_ref": "https://... または 'manual' または 'distilled'",
  "reviewed": true,
  "reviewer": "author_handle_or_anonymous",
  "license": "cc-by-4.0",
  "instrument": "USD/JPY",
  "published_at": "2024-03-15T09:00:00Z",
  "decision_time": "2024-03-15T08:55:00Z",
  "split": "train",
  "holdout_reason": null
}
```

**主要フィールドの用途：**

| フィールド | 用途 |
|---|---|
| `scenario_template_id` | template-based holdout の分割管理。同一テンプレートの派生サンプルが train/eval に混在しないよう管理する |
| `instrument` | instrument-based holdout の分割管理 |
| `published_at` | 元データの公開日時（ISO8601）。time-based holdout の分割基準 |
| `decision_time` | シナリオ中の意思決定時点（ISO8601）。これより未来の情報がプロンプトに混入していないか確認する |
| `source_ref` | 元データの URL または識別子。ライセンス確認・削除要求への対応に使用。経路 1 は `"distilled"`、手動作成は `"manual"` |
| `license` | 経路 3 データの権利確認証跡。経路 1 は `"distilled"`、手動作成は `"proprietary"` |
| `reviewer` | 人手レビュー担当の記録。未レビューは `null` |
| `split` | `"train"` / `"eval"` — ロード時にフィルタリングに使う |
| `holdout_reason` | `"time"` / `"instrument"` / `"template"` / `null` |

`category` と `split` フィールドで `_extract_sft_pairs()` によるフィルタリングが可能。

---

### training/finance_pretrain.py への接続方針

設計書の JSONL を Phase 4/5 に組み込むには以下の追加が必要。実装は別タスクだが、方針をここに記載しておく。

**1. `build_bushido_sft()` 関数の追加**

`training/finance_pretrain.py` に `build_bushido_sft(vocab_size, seq_len, batch_size, device, sft_dir, phase, cache_dir)` を追加する。`sft_dir` 配下の `phase{N}_*.jsonl` を読み込み、`split == "train"` かつ `reviewed == true` のみを返す。

```python
def build_bushido_sft(vocab_size, seq_len, batch_size, device, sft_dir, phase, cache_dir):
    import glob, json
    pattern = str(Path(sft_dir) / f"phase{phase}_*.jsonl")
    pairs = []
    for path in glob.glob(pattern):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("split", "train") != "train":
                    continue
                if not r.get("reviewed", False):
                    continue
                pairs.append((r["instruction"], r["output"], r.get("input", "")))
    # SFTDataset は cache_path.exists() を呼ぶため None を渡すと落ちる。
    # 他の build_* 関数と同様に cache_dir からパスを生成する。
    cache_path = Path(cache_dir) / f"bushido_phase{phase}_sft_{vocab_size}_{_CACHE_VERSION}.pt"
    return SFTDataset(pairs, vocab_size, seq_len, batch_size, device, cache_path)
```

**2. `--bushido_sft_dir` フラグの追加**

```bash
python training/finance_pretrain.py \
  --phase 4 \
  --bushido_sft_dir data/bushido_sft/reviewed \
  --resume checkpoints/finance_a100_v2/phase3_final.pt
```

`--bushido_sft_dir` が指定されている場合は既存データセット（forecaster + sentiment）と結合して Phase 4 データセットを構成する。指定なしなら従来通り。

**3. Phase 5 も同様**

`phase5_*.jsonl`（journal + scenario）を `build_trading_qa()` と結合して Phase 5 データセットとする。
