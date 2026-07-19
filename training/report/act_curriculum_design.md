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

### 「今回学習する区間」を起点に進捗を測る（anchor 方式）
ランプ基準は**全フェーズ合計ステップ(grand_total)**だが、進捗は `anchor_step`
(今回の学習が始まる step)を起点に測る:

```
span     = max(grand_total - 1 - anchor_step, 1)
progress = (step - anchor_step) / span             # 今回学習区間の末尾で 1.0
value    = ramp(progress, start, end, warmup_frac)  # warmup_frac までに start→end、以降 end 固定
```

`anchor_step` は resume 時の開始 step(`--resume` で読んだ step、無ければ 0)。

- **フルラン(phase1 から / 新規)**: `anchor_step=0` → 全フェーズ通しでランプ。
- **phase2-5 のみ再学習**(`--resume phase1_final.pt`, step=30000 起点): `anchor_step=30000`
  → **phase2-5 の区間で start→end をランプ**。

> ⚠️ もし anchor を使わず全フェーズ通しの進捗(step/grand_total)で測ると、phase1_final
> (step≈30000)から resume した時点で進捗が既に warmup_frac を超え、**閾値が end 固定=
> カリキュラムが無効化**される。anchor 方式はこれを回避する設計(回帰テストあり)。

`warmup_frac` は「今回学習区間のうち何割でランプを完了するか」。既定 0.5。
`warmup_frac<=0` はランプ無し=即 end(無効と同義)。

### フェーズを別プロセスで分割実行する場合は `--act_anchor_step` を固定する
Colab notebook は phase2/3/4/5 を**別プロセス**(各 `--phase N --resume 前フェーズ`)で
回す。anchor を自動(=各プロセスの resume step)にすると、**フェーズ頭ごとに anchor が
取り直され、閾値が毎回 start にリセットされる**(ノコギリ波)。

→ 全フェーズに **`--act_anchor_step <phase1 合計 step>`(例 30000)を明示**で渡すと、
別プロセスをまたいで grand_total 区間の連続ランプになる。`grand_total` は phaseN_steps
から計算され全セルで同一なので、anchor を揃えるだけで進捗が一致する。

| 起点 | phase2 別プロセス | phase3 別プロセス | … |
|---|---|---|---|
| 自動(resume step) | 0.5→0.86 | **0.5**→…(リセット) | 毎回リセット |
| `--act_anchor_step 30000` 固定 | 0.5→0.86 | 0.86→…(連続) | 連続 |

Colab 切断後の再 resume でも同じ値を渡せばランプが再現する(自動だと再 resume 点で
取り直されるため、明示指定を推奨)。

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

### 実行例（phase2-5 を phase1_final から再学習）

phase は累積境界で管理されるため、phase1 の続きは `--resume phase1_final.pt`
(step を p1_total に設定)+ `--phase 2` で起動する(base_ckpt は元の素体)。
`--resume` を使わず `--base_ckpt phase1_final.pt` だけにすると step=0 から始まり
phase 境界がずれる(phase1 を再走する)ので注意。

```bash
python3 training/finance_pretrain.py \
  --base_ckpt checkpoints/a100_v2_gpt2vocab/final.pt \
  --resume    checkpoints/finance_a100_v2/phase1_final.pt \
  --phase 2 \
  --phase1_steps 30000 --phase2_steps 8000 ... \
  --act_curriculum --act_anchor_step 30000 \
  --act_threshold_start 0.5 --act_warmup_frac 0.5 \
  --ponder_weight_start 0.02 --ponder_weight_end 0.0 \
  ...（既存の batch/seq_len/seed 等）
```

`--act_anchor_step 30000`(phase1 合計 step)を全フェーズセルに渡すことで、別プロセス
でも phase2-5 の区間で閾値が 0.5→0.99 に連続ランプする(指定しないと各フェーズ頭で
start にリセットされる。上の「別プロセス分割実行」を参照)。
フェーズ開始行に `ramp over steps 30000→…`、ログ行に `act_thr=`(ponder>0 時は
`ponder=`)が出力され、進捗どおり閾値が上昇しているか確認できる。

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
- `apply_act_curriculum()` — `anchor_step` 起点の進捗で `model.cfg` を in-place 更新（返り値は現在値）
- `run_phase()` — 全フェーズ合計から `act_grand_total` を算出し、`act_anchor_step` 起点で
  毎 micro-step に適用。フェーズ頭にスケジュール表示(ramp 区間)、ログ行に `act_thr=` を追加
- `train()` — `--act_anchor_step`(既定 -1=resume step 自動)を解決して全 `run_phase`
  に伝搬。`--act_threshold_end` の既定 -1 を `cfg.act_threshold` に解決

## テスト

[tests/test_finance_pretrain.py](../../tests/test_finance_pretrain.py):
`TestCurriculumRamp`（ランプの数値挙動）/ `TestApplyACTCurriculum`（共有 cfg の
in-place 更新・閾値単調増・ponder 単調減・warmup 後 hold・既定 no-op・**anchor 起点の
区間ランプ**・**anchor=0 のグローバル進捗回帰**）。計 16 ケース。

## 実測: 新 phase5(カリキュラム適用)finance PPL

phase1 から ACT カリキュラム込み(anchor=0 / threshold 0.5→0.99 / warmup_frac=0.73 /
ponder 0.03→0.0)で全 5 フェーズを通し学習(step 52000、`max_seq_len=256`、`attn_type=mla`、
`optimizer_type=AdamW8bit`)。train.log 上で `act_thr` が phase1 開始 0.501 → phase3 で
0.990 に滑らかにランプし、finance フェーズ(3-5)は深いまま回る、という設計どおりの挙動を確認。

finance 評価(sliding-window、`--seq_len 256 --stride 128 --eval_max_chunks 30`、CPU):

| 条件 | n_loops=1 | =2 | =4 | =8 | fp32比 |
|---|---|---|---|---|---|
| fp32(基準) | 47.02 | 43.26 | 43.27 | 43.27 | — |
| full-INT8 | 55.16 | 49.13 | 49.13 | 49.13 | +13.5% |
| mixed-INT8(kv_down=fp32) | 52.03 | 46.93 | 46.92 | 46.92 | +8.4% |

所見:
- **ループ動態は健全**: fp32 が深さで悪化しない(47.0→43.3 で安定、n_loops 2 以降は平坦)。
  カリキュラムの主目的「深いループで壊れない」を達成。1→2 で改善後にサチるのは ACT が
  早めに停止し再帰精製が finance で速く飽和するため。
- **INT8 ボトルネックは従来どおり**: mixed(kv_down を fp32 維持)で +8.4% ≤ full +13.5%。
  kv_down が INT8 の主犯という過去知見と整合。

⚠️ **これは「カリキュラムで性能が向上した」ことを意味しない**:
1. **seq_len が違う** — 本モデルは 256 学習(GPU ティアの都合)。過去レポートは 1024。
   短文脈ほど PPL は高く出るため、43.3 を過去の 30〜36 と直接比較してはいけない。
2. **対照が無い** — 旧 no-curriculum phase5 は削除済み。性能効果の分離には
   同 seed・同 steps・**同 seq_len(256)**・`--act_curriculum` 無しの対照ランが必要。

→ 現時点の結論は「カリキュラムは正しく学習でき、深さで壊れない健全なモデルを作った」まで。
向上主張には対照ランが要る(下記 今後)。

## 留意点・今後

- **`--compile` との両立は実装済み**: `act_threshold`とponder weightを非永続の0次元
  tensor bufferへ移し、in-place更新する。Python float guardを作らないため、Colab notebookは
  `ACT_CURRICULUM=True`のまま対応GPUで`USE_COMPILE=True`を利用できる。Dynamo compile counter
  ではbuffer更新による追加compileなしを確認済み。A100/batch 16/seq 256/8 loops/gradient
  checkpointingの定常計測3回では、計測区間の追加graph/breakはすべて0、中央値はeager
  35,360 tokens/secに対してcompile 38,054 tokens/sec（1.075倍）、peak VRAMは3292から
  3137 MiB、最大loss差は0.00154114だった。総wall-clock効果はphaseログで別途確認する。
  **恒久対応**は `act_threshold` / `act_aux_loss_weight` を `persistent=False` の tensor buffer
  にして in-place 更新する形(buffer 読みなら値変更で再コンパイルされず、state_dict にも
  載らないので checkpoint 互換も保てる)。これは model 本体(main.py)の変更になるため別対応。
- **CLI 値域は `validate_act_curriculum_args` で検証**: threshold ∈ (0,1]、warmup_frac ∈ [0,1]、
  ponder_weight ≥ 0。特に ponder<0 は補助損失が負になり余分ループを報酬化するため弾く。
- 検証は CPU でのスケジュール単体テストまで。**実効果(finance PPL・ループ安定性)は
  GPU/Colab での学習で要計測**。比較は「`--act_curriculum` あり vs なし(同 seed・同
  steps・同 seq_len)」で行う。
- ramp は線形のみ。必要なら cosine 等への拡張は `_curriculum_ramp` 差し替えで対応可能。
- ponder cost を強くかけると停止が早まりすぎる恐れ。まずは閾値ランプ単独 → 効果を見て
  ponder を少量(0.01〜0.02)から併用する運用を推奨。
