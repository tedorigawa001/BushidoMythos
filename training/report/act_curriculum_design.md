# ACT カリキュラム（動的 act_threshold / ponder cost）設計メモ

## 背景・狙い

Recurrent Depth Transformer は ACT(Adaptive Computation Time, Graves 2016)で
トークンごとに停止確率を学習し、易しいトークンは早く停止・難しいトークンは深く
回す。停止は累積停止確率が `act_threshold` を超えた時点で確定する。

従来は `act_threshold=0.99` 固定・`act_aux_loss_weight=0.0`(ponder cost オフ)。
問題意識:

- 学習初期からいきなり深いループ(高い閾値=停止しにくい)で回すと、**ループ勾配が
  不安定**になりやすい(深さ方向に誤差が積み上がる。Exp B のループ破綻と同根)。
- 逆に常に浅いと、難解な金融タスクで必要な深い推論を獲得しにくい。

→ **浅→深カリキュラム**: 初期は浅いループで基礎的な推論を固め、学習が進むにつれて
深い推論を解禁する。`act_threshold` を低→高にランプ(停止しやすい→しにくい)し、
任意で ponder cost を大→小に減衰(初期は余分ループを強く抑制→後半で緩める)。

## 設計

### モデルコード非改変
`RecurrentBlock.forward` は毎イテレーション `self.cfg.act_threshold` /
`act_aux_loss_weight` を**実行時に読む**([main.py:1167](../../bushido_mythos/main.py),
[main.py:1404](../../bushido_mythos/main.py))。`cfg` は model→RecurrentBlock で共有
される単一オブジェクトなので、**学習ループ側で `model.cfg` を書き換えるだけ**で次の
forward に反映される。モデル本体は一切変更しない。

### グローバル進捗で測る
ランプ基準は**全フェーズ(1〜5)合計ステップ**。フェーズ境界に依存せず、学習全体で
一貫した浅→深カリキュラムになる。

```
progress = step / max(grand_total - 1, 1)        # 最終ステップで 1.0
value    = ramp(progress, start, end, warmup_frac) # warmup_frac までに start→end、以降 end 固定
```

`warmup_frac` は「全学習のうち何割でランプを完了するか」。既定 0.5(前半で end へ
到達し、後半は深いまま安定学習)。`warmup_frac<=0` はランプ無し=即 end(無効と同義)。

### ACT remainder trick との整合
閾値 < 1 でも ACT は数値的に安全。停止ステップで残り確率質量を最終重みに割り当てる
remainder trick が `still_running` ゲートで `threshold<1` を正しく処理する
([main.py:1160-1171](../../bushido_mythos/main.py))。低い開始閾値(例 0.5)でも
重み和が破綻しない。

## CLI

| フラグ | 既定 | 説明 |
|---|---|---|
| `--act_curriculum` | off | カリキュラム有効化。未指定なら従来通り cfg 固定値 |
| `--act_threshold_start` | 0.5 | 開始閾値(浅い=早期停止) |
| `--act_threshold_end` | -1 | 終了閾値。-1 は base ckpt の `cfg.act_threshold`(通常 0.99)を採用 |
| `--act_warmup_frac` | 0.5 | start→end をランプし切る割合 |
| `--ponder_weight_start` | 0.0 | 開始 ponder cost。初期に余分ループを抑えたいとき >0 |
| `--ponder_weight_end` | 0.0 | 終了 ponder cost。start>end で後半ほど緩める |

既定では `--act_curriculum` 未指定 = 完全後方互換。ponder は既定 0→0(無効)で、
**閾値が主レバー、ponder は補助**。

### 実行例

```bash
python3 training/finance_pretrain.py \
  --base_ckpt checkpoints/finance_a100_v2/phase1_final.pt \
  --act_curriculum \
  --act_threshold_start 0.5 --act_warmup_frac 0.5 \
  --ponder_weight_start 0.02 --ponder_weight_end 0.0 \
  ...（既存の phase/steps/seed 等）
```

学習ログに `act_thr=`(と ponder>0 時は `ponder=`)が出力され、進捗どおり閾値が
上昇しているか確認できる。

## 挙動（start=0.5→end=0.99, warmup_frac=0.5, ponder 0.02→0）

| 進捗 | act_threshold | ponder | 解釈 |
|---|---|---|---|
| 0% | 0.500 | 0.020 | 浅いループで早期停止＋余分ループに強ペナルティ |
| 25% | 0.745 | 0.010 | 線形に深さを解禁 |
| 50%（warmup 完了） | 0.990 | 0.000 | 深い推論を全面解禁・ペナルティ解除 |
| 以降 | 0.990 | 0.000 | hold |

## 実装箇所

すべて [training/finance_pretrain.py](../finance_pretrain.py):

- `_curriculum_ramp()` — 線形ランプ→hold（純関数）
- `apply_act_curriculum()` — グローバル進捗で `model.cfg` を in-place 更新（返り値は現在値）
- `run_phase()` — 全フェーズ合計から `act_grand_total` を算出し、毎 micro-step で適用。
  フェーズ頭にスケジュール表示、ログ行に `act_thr=` を追加
- `train()` — `--act_threshold_end` の既定 -1 を `cfg.act_threshold` に解決

## テスト

[tests/test_finance_pretrain.py](../../tests/test_finance_pretrain.py):
`TestCurriculumRamp`（ランプの数値挙動）/ `TestApplyACTCurriculum`（共有 cfg の
in-place 更新・閾値単調増・ponder 単調減・warmup 後 hold・既定 no-op）。計 14 ケース。

## 留意点・今後

- 検証は CPU でのスケジュール単体テストまで。**実効果(finance PPL・ループ安定性)は
  GPU/Colab での学習で要計測**。比較は「`--act_curriculum` あり vs なし(同 seed・同
  steps)」で行う。
- ramp は線形のみ。必要なら cosine 等への拡張は `_curriculum_ramp` 差し替えで対応可能。
- ponder cost を強くかけると停止が早まりすぎる恐れ。まずは閾値ランプ単独 → 効果を見て
  ponder を少量(0.01〜0.02)から併用する運用を推奨。
