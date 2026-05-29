#!/usr/bin/env python3
"""
武士道 SFT データ生成スクリプト。

教師モデル（Claude / GPT-4o）にシナリオテンプレートを渡し、
docs/datasets.md のスキーマに準拠した JSONL を data/bushido_sft/raw/ に出力する。

使い方:
    python data/generate_bushido_sft.py --provider anthropic --category A
    python data/generate_bushido_sft.py --provider openai --category all --n 5
    python data/generate_bushido_sft.py --provider anthropic --category A-3 --n 10
    python data/generate_bushido_sft.py --provider anthropic --dry_run   # API 非呼び出し

必要な環境変数:
    ANTHROPIC_API_KEY   # --provider anthropic の場合
    OPENAI_API_KEY      # --provider openai の場合
"""

import argparse
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 出力先
# ---------------------------------------------------------------------------

RAW_DIR     = Path(__file__).parent / "bushido_sft" / "raw"
DRY_RUN_DIR = Path(__file__).parent / "bushido_sft" / "raw" / "dry_run"

# ---------------------------------------------------------------------------
# システムプロンプト
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
あなたは武士道の哲学に基づいて行動する規律のあるプロのトレーダーです。
以下のルールを厳守して回答してください。

【必須構成】必ず以下の4セクションで回答する。省略不可。
  [状況認識] 相場環境・セットアップの客観的評価
  [リスク評価] 損切り水準・最大損失額・無効化条件
  [行動提案] エントリー可否・ポジションサイズ・優先シナリオ
  [不確実性] 判断根拠が弱い箇所・見落としうるリスク・情報の欠如

【禁止表現】断定表現は絶対に使わない。
  NG: 必ず・確実に・間違いなく・絶対に・保証・利益が出る
  OK: 可能性がある・想定される・確認できれば・不確実性が高い

【必須要素】
  - 損切り水準を具体的な価格水準または % で明示する
  - ポジションサイズは資産の % またはリスク額で表現する
  - 不確実性セクションは1文以上必ず記載する

【特別ルール】
  - リベンジトレード・ナンピン（含み損への買い増し）は常に否定する
  - 特定銘柄への投資推奨（「〇〇を買うべき」）は書かない
  - 「このトレードは成功します」などの予言的表現は書かない
"""

# ---------------------------------------------------------------------------
# シナリオテンプレート
# ---------------------------------------------------------------------------
# 各テンプレートは instruction（プロンプト本文）・instrument・category・
# scenario_template_id で構成。
# phase: 4 → Phase 4 SFT (A/B/C)、5 → Phase 5 SFT (D/E)

TEMPLATES: list[dict] = [

    # ── A-1: 順張りセットアップ ──────────────────────────────────────────
    {
        "id": "A1-usdjpy-trend-breakout-v1",
        "category": "A-1", "phase": 4, "instrument": "USD/JPY",
        "instruction": (
            "USD/JPY 日足: 21日移動平均線が上向き、先週の高値 157.80 をブレイクアウト。"
            "出来高は平均比 1.4 倍。RSI 58（過熱ではない）。\n"
            "押し目を待たずにトレンドフォローでロングエントリーすべきか検討してほしい。"
        ),
    },
    {
        "id": "A1-nk225-trend-ma-v1",
        "category": "A-1", "phase": 4, "instrument": "NK225",
        "instruction": (
            "日経225 先物 日足: 5日・25日・75日移動平均線がパーフェクトオーダー（上から順）。"
            "38,500 付近でリトレースメント後に再上昇。MACD はシグナル線上。\n"
            "このセットアップでのロングエントリーの根拠とリスクを評価してほしい。"
        ),
    },
    {
        "id": "A1-gold-breakout-v1",
        "category": "A-1", "phase": 4, "instrument": "XAUUSD",
        "instruction": (
            "金 (XAU/USD) 日足: 半年間のレンジ上限 2,080 ドルを週足でクローズ上抜け。"
            "地政学リスク高まりで安全資産需要増。ドル指数は弱含み。\n"
            "このブレイクアウトへのエントリー可否と想定シナリオを示してほしい。"
        ),
    },

    # ── A-2: 逆張りセットアップ ──────────────────────────────────────────
    {
        "id": "A2-spx-oversold-bounce-v1",
        "category": "A-2", "phase": 4, "instrument": "SPX",
        "instruction": (
            "S&P500 日足: 先週 3% 急落し、200日移動平均線に到達。RSI 28（売られ過ぎ）。"
            "VIX は 28 に急騰後、やや落ち着き始めている。\n"
            "自律反発を狙った逆張りロングのリスクリワードを評価してほしい。"
        ),
    },
    {
        "id": "A2-usdjpy-resistance-short-v1",
        "category": "A-2", "phase": 4, "instrument": "USD/JPY",
        "instruction": (
            "USD/JPY 4時間足: 過去に3回反落した強い抵抗線 158.50 に接近中。"
            "直近 2 週間で 300 pips 上昇しており過熱感あり。RSI 74。\n"
            "抵抗線での売り（ショート）エントリーの妥当性を検討してほしい。"
        ),
    },
    {
        "id": "A2-btc-support-bounce-v1",
        "category": "A-2", "phase": 4, "instrument": "BTC/USD",
        "instruction": (
            "BTC/USD 日足: 60,000 ドルの心理的節目かつ週足サポート付近まで下落。"
            "過去2回このレベルから反発実績あり。ただしビットコインは高ボラティリティ。\n"
            "このレベルでの逆張りロングのリスクとポジションサイジングを考えてほしい。"
        ),
    },

    # ── A-3: スキップ判断（最重要・全体の30%以上を占める） ──────────────
    {
        "id": "A3-usdjpy-resistance-event-skip-v1",
        "category": "A-3", "phase": 4, "instrument": "USD/JPY",
        "instruction": (
            "USD/JPY 日足: 157.30 付近に強い抵抗線。現在 157.25。RSI 72（過熱）。"
            "明日 FOMC 議事録公開予定。ロングエントリーを検討中。"
        ),
    },
    {
        "id": "A3-nk225-overextended-skip-v1",
        "category": "A-3", "phase": 4, "instrument": "NK225",
        "instruction": (
            "日経225: 直近10営業日で 2,500 円上昇。本日さらに高値更新。"
            "セットアップとしては上昇トレンド継続に見えるが、上昇ペースが急すぎる気がする。"
            "ここでロングを追いかけるべきか判断してほしい。"
        ),
    },
    {
        "id": "A3-low-rrr-skip-v1",
        "category": "A-3", "phase": 4, "instrument": "EUR/USD",
        "instruction": (
            "EUR/USD: 1.0820 でロングエントリーを検討。損切りは 1.0790（-30pips）、"
            "目標は 1.0850（+30pips）。リスクリワード 1:1 のトレード。\n"
            "このセットアップで入るべきか評価してほしい。"
        ),
    },
    {
        "id": "A3-unclear-setup-skip-v1",
        "category": "A-3", "phase": 4, "instrument": "XAUUSD",
        "instruction": (
            "金 (XAU/USD): 「なんとなく上がりそうな気がする」「周りのトレーダーが強気」という理由で"
            "ロングエントリーを考えている。具体的なテクニカル根拠はない。\n"
            "このエントリーを実行すべきか判断してほしい。"
        ),
    },

    # ── A-4: イベント前後 ────────────────────────────────────────────────
    {
        "id": "A4-fomc-before-v1",
        "category": "A-4", "phase": 4, "instrument": "USD/JPY",
        "instruction": (
            "本日 24:00 に FOMC 金利決定の発表がある。USD/JPY は現在 156.80。"
            "市場のコンセンサスは据え置きだが、パウエル議長の発言次第では大きく動く可能性。\n"
            "発表前にポジションを持つことのリスクと対処法を教えてほしい。"
        ),
    },
    {
        "id": "A4-earnings-gap-v1",
        "category": "A-4", "phase": 4, "instrument": "Individual Stock",
        "instruction": (
            "保有株が本日の決算発表後に 8% ギャップアップして寄り付いた。"
            "決算内容は予想を上回る好業績。さらに上を狙って追加購入を検討している。\n"
            "ギャップアップ後の追加購入の是非をリスク観点で評価してほしい。"
        ),
    },

    # ── B-1: ポジションサイズ計算 ────────────────────────────────────────
    {
        "id": "B1-position-sizing-basic-v1",
        "category": "B-1", "phase": 4, "instrument": "USD/JPY",
        "instruction": (
            "口座資産: 500万円。1トレードあたりのリスク許容度: 資産の1%（5万円）。"
            "USD/JPY ロング: エントリー 157.00、損切り 156.70（-30pips）。"
            "1pip = 約100円（10万通貨単位）と仮定。\n"
            "適切なポジションサイズを計算し、根拠を説明してほしい。"
        ),
    },
    {
        "id": "B1-position-sizing-high-vol-v1",
        "category": "B-1", "phase": 4, "instrument": "BTC/USD",
        "instruction": (
            "口座資産: 200万円。通常は1トレード1%リスクだが、"
            "ビットコインは1日10%以上動くこともある高ボラティリティ資産。\n"
            "BTC/USD のトレードでは通常株・FX と比べてポジションサイズをどう調整すべきか。"
        ),
    },
    {
        "id": "B1-kelly-criterion-v1",
        "category": "B-1", "phase": 4, "instrument": "General",
        "instruction": (
            "過去50トレードの統計: 勝率 45%、平均利益 120pips、平均損失 60pips。"
            "このデータから最適なポジションサイズ（口座比率）を考えてほしい。"
            "ケリー基準も参考にしながら現実的な推奨を示してほしい。"
        ),
    },

    # ── B-2: 損切り水準の設定 ────────────────────────────────────────────
    {
        "id": "B2-stop-technical-v1",
        "category": "B-2", "phase": 4, "instrument": "NK225",
        "instruction": (
            "日経225 先物 ロング 38,200 でエントリー済み。"
            "直近安値は 37,800（400円下）、25日移動平均線は 37,600（600円下）。\n"
            "テクニカルに根拠のある損切り水準と、その選択理由を説明してほしい。"
        ),
    },
    {
        "id": "B2-stop-too-tight-v1",
        "category": "B-2", "phase": 4, "instrument": "USD/JPY",
        "instruction": (
            "USD/JPY ロングポジション。損切りを「心理的に耐えられる最大損失」の"
            "10,000円（-10pips）に設定したい。現在のATR（平均真の値幅）は80pips。\n"
            "この損切り幅の妥当性を評価してほしい。"
        ),
    },

    # ── B-3: 部分利確 ─────────────────────────────────────────────────────
    {
        "id": "B3-partial-exit-trend-v1",
        "category": "B-3", "phase": 4, "instrument": "USD/JPY",
        "instruction": (
            "USD/JPY ロング 155.00 エントリー、現在 156.50（+150pips 含み益）。"
            "目標は 158.00。強いトレンドが続いているが、利益を確定したい気持ちもある。\n"
            "部分利確のタイミングと残りポジションの管理方法を提案してほしい。"
        ),
    },
    {
        "id": "B3-scale-out-resistance-v1",
        "category": "B-3", "phase": 4, "instrument": "XAUUSD",
        "instruction": (
            "金 (XAU/USD) ロング 2,000 ドルエントリー、現在 2,060 ドル（+60ドル）。"
            "次の抵抗線 2,080 ドルが近い。全利確か部分利確か悩んでいる。\n"
            "段階的な決済戦略と残りポジションの損切り移動について提案してほしい。"
        ),
    },

    # ── B-4: ナンピン拒否（最重要） ──────────────────────────────────────
    {
        "id": "B4-no-averaging-down-futures-v1",
        "category": "B-4", "phase": 4, "instrument": "NK225",
        "instruction": (
            "日経225 先物 ロング 38,000 で 2 枚保有。現在 37,600 (-400)。"
            "「平均単価を下げるためにもう 2 枚買い増したい」という考えが浮かんでいる。"
        ),
    },
    {
        "id": "B4-no-averaging-down-fx-v1",
        "category": "B-4", "phase": 4, "instrument": "USD/JPY",
        "instruction": (
            "USD/JPY ロング 157.00 で保有中、現在 156.20 (-80pips)。"
            "「いずれ戻るはずだから 156.20 でも追加ロングして平均コストを下げたい」。\n"
            "この判断の問題点を指摘してほしい。"
        ),
    },
    {
        "id": "B4-no-averaging-down-btc-v1",
        "category": "B-4", "phase": 4, "instrument": "BTC/USD",
        "instruction": (
            "BTC/USD を 65,000 ドルで購入。現在 52,000 ドル（-20%）。"
            "「長期的には上がると信じているので、ここで大きく買い増したい。平均 58,500 になる」。\n"
            "トレーダーの立場でこの行動を評価してほしい。"
        ),
    },

    # ── C-1: トレンド vs レンジ判断 ──────────────────────────────────────
    {
        "id": "C1-regime-detection-v1",
        "category": "C-1", "phase": 4, "instrument": "USD/JPY",
        "instruction": (
            "USD/JPY 週足: 過去3ヶ月で 152.00〜158.00 のレンジ内を往来。"
            "ADX は 18（方向性弱い）。ボリンジャーバンドの幅が縮小中（スクイーズ）。\n"
            "現在の相場環境をどう分類し、どの戦略が適切かを評価してほしい。"
        ),
    },
    {
        "id": "C1-trend-vs-range-v1",
        "category": "C-1", "phase": 4, "instrument": "SPX",
        "instruction": (
            "S&P500: 200日移動平均線の上にあり、高値更新が続いているが、"
            "直近1ヶ月は横ばい。強気相場の継続か、天井圏のレンジ形成か判断が難しい。\n"
            "複数の指標を使ってどちらの環境かを評価してほしい。"
        ),
    },

    # ── C-2: マクロ指標の解釈 ────────────────────────────────────────────
    {
        "id": "C2-cpi-surprise-v1",
        "category": "C-2", "phase": 4, "instrument": "USD/JPY",
        "instruction": (
            "米国 CPI 発表: 予想 3.1%、結果 3.5%（上振れ）。"
            "発表直後に USD/JPY は 156.80 → 157.60 に急騰。\n"
            "このマクロイベントの市場への影響と、今後のトレード戦略への示唆を説明してほしい。"
        ),
    },
    {
        "id": "C2-boj-intervention-v1",
        "category": "C-2", "phase": 4, "instrument": "USD/JPY",
        "instruction": (
            "USD/JPY が 160.00 に接近。日銀の為替介入の噂が市場に広がっている。"
            "過去の介入実績: 2022年9月（145円台）、2022年10月（151円台）。\n"
            "介入リスクをどう評価し、現在のポジション管理に反映すべきか考えてほしい。"
        ),
    },

    # ── C-3: センチメント評価 ────────────────────────────────────────────
    {
        "id": "C3-vix-spike-v1",
        "category": "C-3", "phase": 4, "instrument": "SPX",
        "instruction": (
            "VIX が前日の 15 から 32 に急騰。S&P500 は 3% 下落。"
            "Fear & Greed Index は 「Extreme Fear」（12）。\n"
            "このセンチメント環境でのトレード戦略と注意点を説明してほしい。"
        ),
    },
    {
        "id": "C3-put-call-ratio-v1",
        "category": "C-3", "phase": 4, "instrument": "SPX",
        "instruction": (
            "Put/Call Ratio が 1.4（通常は 0.7〜0.9）に急上昇。"
            "機関投資家が大量のプットオプションを買っている状況。\n"
            "この指標をどう解釈し、今後の相場見通しにどう活かすか考えてほしい。"
        ),
    },

    # ── C-4: クロスアセット分析 ──────────────────────────────────────────
    {
        "id": "C4-bond-equity-v1",
        "category": "C-4", "phase": 4, "instrument": "SPX",
        "instruction": (
            "米国10年債利回りが 4.8% まで上昇し、株式市場に売り圧力がかかっている。"
            "ドルインデックス (DXY) は 106 と強含み。\n"
            "この債券・株・ドルの相互関係から、現在の市場環境と今後の見方を整理してほしい。"
        ),
    },
    {
        "id": "C4-risk-on-off-v1",
        "category": "C-4", "phase": 4, "instrument": "General",
        "instruction": (
            "本日の市場: 日経平均 +1.8%、ナスダック +2.1%、金 -0.5%、円 -0.8%（ドル高）。"
            "VIX は 14 に低下。\n"
            "この複合的な市場シグナルから「リスクオン/リスクオフ」環境を判断し、"
            "各資産クラスへの示唆を説明してほしい。"
        ),
    },

    # ── D-1: 勝ちトレードのレビュー ──────────────────────────────────────
    {
        "id": "D1-winning-but-bad-process-v1",
        "category": "D-1", "phase": 5, "instrument": "USD/JPY",
        "instruction": (
            "USD/JPY トレード結果: エントリー 156.00（根拠: なんとなく強そう）、"
            "損切りなし、目標なし。結果的に 157.20 まで上昇して利益確定。+120pips。\n"
            "このトレードを武士道の観点でレビューしてほしい。"
        ),
    },
    {
        "id": "D1-winning-good-process-v1",
        "category": "D-1", "phase": 5, "instrument": "NK225",
        "instruction": (
            "日経225 トレード: 移動平均線の支持・RSI 45（中立）・出来高増加でロング。"
            "損切りを直近安値 -150円に設定。目標は直近高値。計画通り +320円で決済。\n"
            "このトレードの良かった点と、改善できる点を評価してほしい。"
        ),
    },

    # ── D-2: 負けトレードのレビュー ──────────────────────────────────────
    {
        "id": "D2-stop-not-honored-v1",
        "category": "D-2", "phase": 5, "instrument": "BTC/USD",
        "instruction": (
            "BTC/USD: 損切りを 62,000 ドルに設定してロングエントリー。"
            "62,000 到達時に「すぐ戻るだろう」と損切りをキャンセル。"
            "その後 55,000 まで下落し大損失。\n"
            "この失敗のメカニズムと、同じ過ちを繰り返さないための対策を分析してほしい。"
        ),
    },
    {
        "id": "D2-overtrading-v1",
        "category": "D-2", "phase": 5, "instrument": "General",
        "instruction": (
            "今週の取引: 月〜金で 23 回エントリー。勝率 35%、合計損失 15万円。"
            "通常は週 3〜5 回のトレードが計画だった。\n"
            "オーバートレードの原因と、ルール遵守のための仕組みを考えてほしい。"
        ),
    },
    {
        "id": "D2-fomo-entry-v1",
        "category": "D-2", "phase": 5, "instrument": "SPX",
        "instruction": (
            "S&P500 が急上昇しているのを見て「乗り遅れた」と焦り、"
            "高値圏でセットアップ確認なしにロングエントリー。その後急落し損失。\n"
            "FOMO（取り残される恐怖）によるエントリーの問題点と予防策を分析してほしい。"
        ),
    },

    # ── D-3: ルール違反の分析 ────────────────────────────────────────────
    {
        "id": "D3-revenge-trade-v1",
        "category": "D-3", "phase": 5, "instrument": "General",
        "instruction": (
            "今日 3 連敗した後、4 回目のエントリーを衝動的に行い、さらに損失を出した。"
            "計画にないトレードだった。この行動を振り返ってほしい。"
        ),
    },
    {
        "id": "D3-position-size-violation-v1",
        "category": "D-3", "phase": 5, "instrument": "USD/JPY",
        "instruction": (
            "ルールでは1トレード最大2%リスク（口座の）と決めていた。"
            "「絶対に勝てる」と確信したトレードで10%リスクを取り、大きく損失した。\n"
            "ポジションサイズルール違反の心理的メカニズムと再発防止策を分析してほしい。"
        ),
    },
    {
        "id": "D3-plan-deviation-v1",
        "category": "D-3", "phase": 5, "instrument": "NK225",
        "instruction": (
            "日経225: 目標 +200円で利確予定だったが、+180円でさらに上がりそうに感じて"
            "保有継続。その後急落し、エントリー価格まで戻って損益ゼロで撤退。\n"
            "計画からの逸脱とその心理的原因を振り返ってほしい。"
        ),
    },

    # ── E-1: プリモータム ────────────────────────────────────────────────
    {
        "id": "E1-premortem-long-v1",
        "category": "E-1", "phase": 5, "instrument": "USD/JPY",
        "instruction": (
            "USD/JPY ロング 157.00 でエントリーしようとしている。\n"
            "「このトレードが1週間後に失敗していた場合、その理由として考えられることを"
            "できる限り列挙してほしい」。プリモータム（事前の失敗分析）を実施してほしい。"
        ),
    },
    {
        "id": "E1-premortem-strategy-v1",
        "category": "E-1", "phase": 5, "instrument": "General",
        "instruction": (
            "新しいトレード戦略（移動平均クロス + RSI フィルター）を本番口座で使い始めようとしている。\n"
            "「この戦略が3ヶ月後に機能しなくなっていた場合、原因として何が考えられるか」を"
            "プリモータム形式で分析してほしい。"
        ),
    },

    # ── E-2: 複数シナリオ計画 ────────────────────────────────────────────
    {
        "id": "E2-three-scenarios-v1",
        "category": "E-2", "phase": 5, "instrument": "USD/JPY",
        "instruction": (
            "USD/JPY 現在 156.50。来週の FOMC を控えている。\n"
            "強気シナリオ・弱気シナリオ・中立シナリオの3つを想定し、"
            "それぞれの発生条件と対応するトレード戦略を立案してほしい。"
        ),
    },
    {
        "id": "E2-invalidation-planning-v1",
        "category": "E-2", "phase": 5, "instrument": "XAUUSD",
        "instruction": (
            "金 (XAU/USD) ロングポジションを保有中。強気シナリオを想定しているが、\n"
            "どのような価格アクションやファンダメンタルズ変化が起きたら"
            "強気シナリオが無効化されたと判断し、撤退を検討すべきか整理してほしい。"
        ),
    },

    # ── E-3: ブラックスワン対応 ──────────────────────────────────────────
    {
        "id": "E3-flash-crash-v1",
        "category": "E-3", "phase": 5, "instrument": "General",
        "instruction": (
            "保有ポジションがある状態で、相場が数分以内に 5% 急落した（フラッシュクラッシュ）。"
            "損切り注文は約定できず、スリッページが大きい。\n"
            "このような緊急事態での対処手順と、事前に備えておくべきことを整理してほしい。"
        ),
    },
    {
        "id": "E3-liquidity-crisis-v1",
        "category": "E-3", "phase": 5, "instrument": "General",
        "instruction": (
            "市場の流動性が極端に低下し、スプレッドが通常の10倍になっている（週末深夜・"
            "大規模ニュース直後など）。この状況でのトレードのリスクと対応策を説明してほしい。"
        ),
    },
]

# カテゴリ → テンプレートのインデックスを構築
CATEGORY_INDEX: dict[str, list[dict]] = {}
for t in TEMPLATES:
    CATEGORY_INDEX.setdefault(t["category"], []).append(t)
    CATEGORY_INDEX.setdefault(t["category"][0], []).append(t)  # A, B, C... でも絞れる

# ---------------------------------------------------------------------------
# 品質バリデーション
# ---------------------------------------------------------------------------

_REQUIRED_SECTIONS = ["[状況認識]", "[リスク評価]", "[行動提案]", "[不確実性]"]
_FORBIDDEN_PATTERNS = [
    r"必ず(上|下|勝|利益|損切り|利確)",
    r"確実に",
    r"間違いなく",
    r"絶対に上",
    r"保証(し|する|できる)",
    r"必ず利益",
    # ナンピン肯定
    r"ナンピン.{0,30}(効果的|有効|お勧め|がよい|してみ|する価値|良い選択)",
    r"買い増し.{0,20}(有効|効果的|お勧め|がよい)",
    # 特定銘柄推奨
    r"(を|は)(買う|購入する|ロングする)べき",
]


def _has_stoploss_numeric(text: str) -> bool:
    """[リスク評価] セクションに損切り水準の数値があるか確認する。

    対象:
      - 単位付き: 2% / 30pips / 37800円 / 2080ドル / 150点
      - FX価格: 損切り|撤退|無効化 の前後50字以内に小数点付き数値 (例: 156.70 / 1.0790)
      - カンマ区切り価格: 37,800
    """
    section = text
    if "[リスク評価]" in text:
        idx = text.index("[リスク評価]")
        m = re.search(r'\[行動提案\]|\[不確実性\]', text[idx:])
        section = text[idx: idx + m.start()] if m else text[idx:]
    # 単位付き or カンマ区切り整数
    if re.search(r'[\d][\d,]*\s*(%|pips?|円|ドル|点)', section):
        return True
    # 損切り関連キーワード周辺の小数価格（FX/先物）
    _STOPLOSS_KEYWORDS = r'(損切り|撤退|無効化|ストップ)'
    if re.search(_STOPLOSS_KEYWORDS + r'.{0,50}[\d]+\.\d+', section):
        return True
    if re.search(r'[\d]+\.\d+.{0,50}' + _STOPLOSS_KEYWORDS, section):
        return True
    return False


def validate_response(text: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for sec in _REQUIRED_SECTIONS:
        if sec not in text:
            errors.append(f"セクション欠落: {sec}")
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, text):
            errors.append(f"禁止表現: {pat}")
    if all(sec in text for sec in _REQUIRED_SECTIONS) and not _has_stoploss_numeric(text):
        errors.append("損切り水準に具体的な数値（%・pips・価格）が見当たりません")
    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# API クライアント
# ---------------------------------------------------------------------------

def call_anthropic(instruction: str, model: str, max_retries: int = 3) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": instruction}],
            )
            return msg.content[0].text
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}] {e} — {wait}s 待機")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def call_openai(instruction: str, model: str, max_retries: int = 3) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}] {e} — {wait}s 待機")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def call_gemini(instruction: str, model: str, max_retries: int = 3) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=instruction,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=1024,
                ),
            )
            return resp.text
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}] {e} — {wait}s 待機")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def call_api(instruction: str, provider: str, model: str) -> str:
    if provider == "anthropic":
        return call_anthropic(instruction, model)
    if provider == "gemini":
        return call_gemini(instruction, model)
    return call_openai(instruction, model)


# ---------------------------------------------------------------------------
# 既存 ID のロード（resume 用）
# ---------------------------------------------------------------------------

def load_existing_ids(out_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in out_dir.glob("*.jsonl"):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "id" in r:
                        ids.add(r["id"])
                except json.JSONDecodeError:
                    pass
    return ids


# ---------------------------------------------------------------------------
# 生成メインループ
# ---------------------------------------------------------------------------

def generate(args: argparse.Namespace) -> None:
    # API キー fail-fast（dry_run では不要）
    if not args.dry_run:
        env_var = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY"}[args.provider]
        if not os.environ.get(env_var):
            raise RuntimeError(
                f"環境変数 {env_var} が設定されていません。\n"
                f"export {env_var}='your-key' を実行してから再試行してください。"
            )

    # dry_run は専用ディレクトリに隔離し、load_existing_ids の対象にしない
    out_dir = DRY_RUN_DIR if args.dry_run else RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 対象テンプレートの絞り込み
    if args.category == "all":
        templates = TEMPLATES
    else:
        templates = CATEGORY_INDEX.get(args.category, [])
        if not templates:
            raise ValueError(
                f"カテゴリ {args.category!r} が見つかりません。\n"
                f"有効値: all, A, B, C, D, E, A-1〜A-4, B-1〜B-4, C-1〜C-4, D-1〜D-3, E-1〜E-3"
            )

    # resume: dry_run ファイルを除いた RAW_DIR のみ走査
    existing_ids = load_existing_ids(RAW_DIR)
    print(f"既存サンプル: {len(existing_ids)} 件")

    now_iso = datetime.now(timezone.utc).isoformat()
    model_slug = args.model.replace("/", "-").replace(".", "-")[:30]
    out_path = out_dir / f"generated_{args.category}_{args.provider}_{model_slug}_{now_iso[:10]}.jsonl"

    total_generated = 0
    total_skipped = 0
    total_failed = 0

    with open(out_path, "a", encoding="utf-8") as fout:
        for tmpl in templates:
            for n in range(args.n):
                # provider + model slug を含めることで別プロバイダの ID 衝突を防ぐ
                sample_id = f"{tmpl['id']}_{args.provider}_{model_slug}_n{n:03d}"
                if sample_id in existing_ids:
                    total_skipped += 1
                    continue

                instruction = tmpl["instruction"]
                print(f"  [{tmpl['category']}] {sample_id} ", end="", flush=True)

                if args.dry_run:
                    output = (
                        "[状況認識] （dry-run: API 非呼び出し）\n"
                        "[リスク評価] （dry-run）\n"
                        "[行動提案] （dry-run）\n"
                        "[不確実性] （dry-run）"
                    )
                else:
                    try:
                        output = call_api(instruction, args.provider, args.model)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        total_failed += 1
                        continue

                # dry_run はスキーマ・パス確認が目的のためバリデーションをスキップ
                if args.dry_run:
                    is_valid, errors = True, []
                else:
                    is_valid, errors = validate_response(output)
                if not is_valid:
                    print(f"INVALID: {errors}")
                    if not args.keep_invalid:
                        total_failed += 1
                        continue

                record = {
                    "id": sample_id,
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "category": tmpl["category"],
                    "scenario_template_id": tmpl["id"],
                    "phase": tmpl["phase"],
                    "source": f"distilled_{args.provider}_{args.model}",
                    "source_ref": "distilled",
                    "reviewed": False,
                    "reviewer": None,
                    "license": "distilled",
                    "instrument": tmpl["instrument"],
                    "published_at": None,
                    "decision_time": None,
                    "split": "pending",
                    "holdout_reason": None,
                    "generated_at": now_iso,
                    "valid": is_valid,
                    "validation_errors": errors if not is_valid else [],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                total_generated += 1
                print("OK" if is_valid else f"OK(invalid:{errors})")

                if not args.dry_run:
                    time.sleep(args.interval)

    print(f"\n完了: 生成={total_generated}  スキップ={total_skipped}  失敗={total_failed}")
    print(f"出力: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="武士道 SFT データ生成スクリプト")
    p.add_argument("--provider",  default="anthropic", choices=["anthropic", "openai", "gemini"],
                   help="教師モデルのプロバイダ (default: anthropic)")
    p.add_argument("--model",     default=None,
                   help="使用するモデル名 (省略時: anthropic=claude-haiku-4-5-20251001, openai=gpt-4o-mini, gemini=gemini-2.0-flash)")
    p.add_argument("--category",  default="all",
                   help="生成対象カテゴリ: all / A / B / C / D / E / A-1〜E-3 (default: all)")
    p.add_argument("--n",         type=int, default=3,
                   help="テンプレート1件あたりの生成バリエーション数 (default: 3)")
    p.add_argument("--interval",  type=float, default=1.0,
                   help="API 呼び出し間隔 [秒] (default: 1.0)")
    p.add_argument("--keep_invalid", action="store_true",
                   help="バリデーション失敗サンプルも JSONL に保存する（後でフィルタリング可）")
    p.add_argument("--dry_run",   action="store_true",
                   help="API を呼び出さずにスキーマと出力パスのみ確認する")
    args = p.parse_args()

    if args.model is None:
        args.model = {
            "anthropic": "claude-haiku-4-5-20251001",
            "openai":    "gpt-4o-mini",
            "gemini":    "gemini-2.0-flash",
        }[args.provider]

    if args.n < 1:
        p.error("--n は 1 以上を指定してください")
    if args.interval < 0:
        p.error("--interval は 0 以上を指定してください")

    return args


if __name__ == "__main__":
    generate(parse_args())
