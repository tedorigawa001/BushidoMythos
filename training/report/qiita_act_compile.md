# 毎ステップ変わるハイパーパラメータと torch.compile を両立させる — 0次元 tensor buffer で「compile 2.2倍遅い」が 1.075倍速に

## TL;DR

- 自作 LLM の **ACT カリキュラム**(halt 閾値と ponder 重みを学習中に毎ステップ変える)は、`torch.compile` と両立しなかった。Python float の `cfg.act_threshold` を書き換えるたびに **guard が再評価され再コンパイルが多発**するため、compile を自動無効化していた。
- 対策は素直で、動的な値を **`persistent=False` の 0 次元 tensor buffer** に移して `fill_()` で in-place 更新するだけ。**guard 対象から外れ、checkpoint スキーマも変わらない**。
- ところが A100 での最初のベンチマークは **compile が 2.2 倍遅い**(0.451×)という結果に。原因は測定の罠で、**ACT の実行ループ数が変わるたびに新しいグラフがコンパイルされ、そのコンパイル時間が計測区間に混入**していた。
- ハーネスを改修し、**Dynamo のグラフ生成数を warmup 区間と計測区間に分離**。「計測区間の新規グラフ = 0」を定常状態の判定条件にした。
- 本番相当条件(batch 16)の定常計測 3 回の中央値は **compile 1.075 倍速 + peak VRAM 155 MiB 減、loss 差 0.0015**。2 日の学習 run で 3〜4 時間の短縮に相当する。

:::note warn
研究用コードの実験ログです。対象は自作の Recurrent-Depth Transformer(98.6M、金融特化学習)。数値はこのモデル・この構成での実測であり、一般化はできません。
:::

---

## 背景: ACT カリキュラムは compile と両立しなかった

自作モデル [BushidoMythos](https://github.com/tedorigawa001/BushidoMythos) は、再帰ブロックを最大 12 回ループさせる Recurrent-Depth Transformer で、**ACT(Adaptive Computation Time)** により入力ごとにループ数を可変にしている。さらに学習では **ACT カリキュラム** — halt 閾値を 0.5 → 0.99 へ、ponder 重みを 0.03 → 0.0 へ、学習の 73% をかけて連続的にランプさせる — を使う。

問題はこの「毎ステップ値が変わる」性質だった。当初の実装はカリキュラム更新をこう書いていた:

```python
# 毎ステップ呼ばれる
model.cfg.act_threshold = thr        # Python float の書き換え
model.cfg.act_aux_loss_weight = pon
```

forward 内で `self.cfg.act_threshold` を読むため、`torch.compile` はこの float を **guard**(再コンパイル判定の条件)に取り込む。値が変わるたびに guard が破れて再コンパイル — 52,000 ステップの学習では論外なので、`--act_curriculum` 指定時は **compile を自動無効化**していた。「適応計算を取るか、コンパイル高速化を取るか」の二択になっていた。

---

## 対策: 0 次元 tensor buffer + in-place 更新

Dynamo の guard は **Python の属性値**に反応する。値を **tensor の中身**として持てば、グラフは「この buffer を読む」という操作としてトレースされ、**中身が変わっても再コンパイルは起きない**。

```python
class RecurrentBlock(nn.Module):
    def __init__(self, cfg):
        ...
        self.register_buffer(
            "_act_threshold",
            torch.tensor(float(cfg.act_threshold)),
            persistent=False,   # ← state_dict に載せない
        )

    def forward(self, ...):
        ...
        newly_halting = still_running & (cumulative_p >= self._act_threshold)
```

更新側は `fill_()` の in-place 書き込み:

```python
@torch.no_grad()
def set_act_curriculum_values(self, threshold: float, ponder_weight: float) -> None:
    self.recurrent._act_threshold.fill_(threshold)
    self._act_aux_loss_weight.fill_(ponder_weight)
    self.cfg.act_threshold = float(threshold)          # ログ互換用に同期
    self.cfg.act_aux_loss_weight = float(ponder_weight)
```

設計上のポイントは 3 つ。

- **`persistent=False`**: buffer が `state_dict` に載らないので、**既存 checkpoint との互換性が完全に保たれる**(load も save も従来通り)。resume 時の値は step から再計算する。
- **compile ラッパー越しの更新**: `torch.compile` は `OptimizedModule` でモデルを包むため、学習スクリプト側は `getattr(model, "_orig_mod", model)` で実体を取ってから setter を呼ぶ。
- **テストで性質を固定**: `torch._dynamo.testing.CompileCounter` をバックエンドにして、「buffer 更新 → 再 forward で **frame_count が増えない** かつ **aux loss は実際に変わる**」の両方を assert した。片方だけだと「更新が効いていないだけ」を検出できない。

```python
def test_buffer_updates_do_not_trigger_recompile(self):
    counter = CompileCounter()
    compiled = torch.compile(model, backend=counter)

    model.set_act_curriculum_values(0.5, 0.02)
    compiled(ids, n_loops=1)
    initial_frames = counter.frame_count

    model.set_act_curriculum_values(0.75, 0.01)
    compiled(ids, n_loops=1)

    assert counter.frame_count == initial_frames   # 再コンパイルなし
    assert model._last_aux_loss.item() != first_aux  # でも値は効いている
```

---

## 罠: 最初のベンチマークは「compile が 2.2 倍遅い」

eager と compile を同一重み・同一バッチ列・同一シードで比較するハーネスを書き、A100 で回した最初の結果がこれだった。

| 条件: batch 1, steps 20, warmup 3 | eager | compile |
|---|---|---|
| tokens/sec | 2,936 | **1,325** |
| Dynamo unique graphs | — | 8 |

**speedup 0.451×**。再コンパイル対策をしたのに、compile が 2.2 倍遅い。

原因は再帰ループの構造にあった。Python の for/while ループを Dynamo は**展開してコンパイル**するため、**ACT の早期終了で実行ループ数が変わると、ループ数ごとに別グラフが生成される**。20 ステップの間に閾値が 0.5 → 0.99 まで一気にランプする設定だったので、実行ループ数が次々に変わり、**8 個のグラフのコンパイル時間が warmup 3 ステップに収まらず計測区間へ漏れた**。eager 換算 1.7 秒分の仕事を測るところに、数秒のコンパイルが混入していた。

つまりこの 0.451× は「定常速度」ではなく「コンパイル代込みの速度」で、**52k ステップの本番 run の速度について何も言えない**数字だった。

---

## 改修: グラフ生成数を warmup / 計測区間で分離する

「計測にコンパイルが混入したか」を**測定結果自体から判定できる**ように、ハーネスを改修した。

- Dynamo counter を warmup 終了時点と計測終了時点で読み、`unique_graphs` / `graph_breaks` を **warmup 区間と計測区間に分けて記録**
- **計測区間の新規グラフが 0 でない結果は、定常速度として扱わない**(合格条件をデータで機械判定)
- warmup をランプ全体が一巡する長さ(`warmup == steps`)にして、全ループ数バリアントを warmup 中にコンパイルさせる
- batch も本番と同じ 16 に(batch 1 はカーネル起動オーバーヘッド支配で、inductor の利得が出にくい条件だった)

```text
eager  : 11.222s  36499.0 tok/s  peak=3292.0 MiB  graphs=0
compile: 10.436s  39250.4 tok/s  peak=3137.0 MiB  graphs=8 (warmup=8, measured=0)
speedup=1.075x  max_loss_delta=0.00154
```

## 結果: 定常 1.075 倍 + VRAM 155 MiB 減

本番相当条件(batch 16, seq 256, 8 loops, gradient checkpointing, bf16)で 3 回計測。**3 回すべて計測区間の新規グラフ/graph break は 0**、中央値:

| 指標 | eager | compile |
|---|---|---|
| tokens/sec | 35,360 | **38,054(1.075×)** |
| peak VRAM | 3,292 MiB | **3,137 MiB** |
| 最大 loss 差 | — | 0.00154(相対 ~0.01%) |

派手な数字ではないが、+7.5% は **2 日の学習 run で 3〜4 時間の短縮**に相当し、コストは冒頭の一回のコンパイルだけ。VRAM も減る。品質ゲート(loss 差)も bf16 の誤差範囲に収まった。これで「ACT カリキュラムを使うなら compile は捨てる」というトレードオフは解消した。

---

## 学び

1. **毎ステップ変わる値は Python 属性ではなく 0 次元 buffer に持つ**。`persistent=False` なら checkpoint スキーマも壊れない。ACT 閾値に限らず、温度・損失重み・カリキュラム係数など「学習中に動くスカラー」全般に効くパターン。
2. **compile のベンチマークは「コンパイル代の混入」をデータで検出できる形にする**。warmup を置くだけでは不十分で、動的制御フローがあるモデルは warmup 後も新グラフが生まれ得る。グラフ生成数を区間別に記録し、「計測区間 0」を合格条件にすると誤判定を機械的に弾ける。
3. **最初の測定が悪くても、原因を分離してから結論を出す**。0.451× を額面通りに受け取っていたら、「ACT と compile は両立しない」という誤った結論をロードマップに書くところだった。

## まとめ

| 項目 | 結果 |
|---|---|
| 再コンパイル対策 | ◎ buffer 更新で frame_count 不変(CompileCounter で検証) |
| checkpoint 互換 | ◎ `persistent=False` でスキーマ不変 |
| 定常速度 | ○ 1.075×(3 回中央値、計測区間の新規グラフ 0) |
| peak VRAM | ○ 3,292 → 3,137 MiB |
| 品質 | ◎ 最大 loss 差 0.00154(bf16 誤差範囲) |

実装・ベンチハーネス・計測 JSON はすべてリポジトリにあります: [BushidoMythos](https://github.com/tedorigawa001/BushidoMythos)(`training/bench_act_compile.py`, `docs/performance_roadmap.md`)
