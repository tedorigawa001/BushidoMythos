# MoE の expert ループを grouped GEMM に置き換えて学習ベンチ約1.99倍 — ただし最初のベンチは「何も変わらない」だった

## TL;DR

- 自作 MoE-LLM の routed experts は「expert ごとの Python ループ」で計算していた。再帰ループ 12 回 × 全層で毎ステップ回るため、**ここが学習の最大ボトルネック**だった。
- PyTorch 2.11 の **native `grouped_mm`** で全 expert を1カーネルにまとめた。primitive は autograd 非対応なので、**forward / 入力勾配 / 重み勾配をすべて grouped GEMM で計算する custom autograd** を書いた。expert 重みは forward 時に stackするため **model state_dictスキーマは不変**。
- ところが最初の A100 ベンチは **ベースラインと完全一致**(peak VRAM が MiB 単位で同値、loss 差 8 桁一致)。grouped 経路は**一度も実行されておらず、旧ループへ silent fallback したものを測っていた**。
- 対策: **fail-fast + 観測可能化**。有効化要求時に不活性なら理由コード付きで即エラー、JSON に `grouped_moe_active` を記録、さらに計測前に実 API を `F.linear` 参照と比較する**数値プローブ**(空 expert 含む、全 delta 0.0 = bit 一致)を仕込んだ。
- 結果: 転置修正後の定常計測2回は **75,903 / 75,205 tok/s(ベースライン中央値38,054比で約1.98〜1.99倍)**、graph break増なし、実データPPLは **85.47 → 85.36(-0.13%)**で品質同等。これはモデルforward/backward部分の計測で、データ供給やcheckpoint I/Oを含む総wall-clockは別途計測が必要。

:::note warn
研究用コードの実験ログです。対象は自作の Recurrent-Depth Transformer(98.6M、MoE FFN、金融特化学習)。数値はこのモデル・この構成での実測であり、一般化はできません。
:::

---

## 背景: MoE の「expert ごとの Python ループ」

自作モデル [BushidoMythos](https://github.com/tedorigawa001/BushidoMythos) の FFN は sparse MoE(routed experts + shared experts)で、従来の routed 側はこう動いていた:

```python
counts_cpu = counts_int.tolist()   # GPU→CPU 同期
out = torch.zeros_like(flat)
offset = 0
for eid, cnt in enumerate(counts_cpu):   # expert ごとの Python ループ
    if cnt > 0:
        tok_rows_e = tok_rows_sorted[offset : offset + cnt]
        expert_out = scores_e * self.routed_experts[eid](flat[tok_rows_e])
        out = out.index_add(0, tok_rows_e, expert_out)
    offset += cnt
```

問題は 3 つ。**(1)** expert 数ぶんの小さいカーネルが毎回発行される、**(2)** `tolist()` の GPU→CPU 同期がある、**(3)** この動的制御フローのため MoE 全体を `@torch._dynamo.disable` で compile 対象から外していた。しかもこのモデルは再帰ブロックを最大 12 回ループするので、**この forward が 1 ステップに何十回も走る**。性能ロードマップで P1 筆頭に置いていた項目だ。

## 実装: native grouped_mm + custom autograd

PyTorch 2.11 の `torch.nn.functional.grouped_mm` は、ソート済み入力とグループ境界(offsets)を渡すと**全グループの GEMM を 1 カーネルで**計算する。SwiGLU expert の gate / up / down を 3 回の grouped GEMM に置き換えた:

```python
routed = flat[tok_rows_sorted].to(dtype=torch.bfloat16)
offsets = counts_int.cumsum(0).to(dtype=torch.int32)   # dispatch 完全 on-device

gate_weight = torch.stack([e.gate.weight for e in self.routed_experts]).to(torch.bfloat16)
# up_weight / down_weight も同様

gate = _GroupedLinear.apply(routed, gate_weight, offsets)
up   = _GroupedLinear.apply(routed, up_weight, offsets)
routed_out = _GroupedLinear.apply(F.silu(gate) * up, down_weight, offsets)
```

設計判断のポイント:

- **model state_dictスキーマ不変**: expert 重みを最初からstacked tensorで持てばstackコストは消えるが、`state_dict`の形が変わり既存checkpointと非互換になる。forward時stackならmodel weightの互換性を保てる。checkpoint payloadには実行モード監査用の`runtime_config.grouped_moe`だけを後方互換で追加した。
- **custom autograd**: `grouped_mm` は勾配を提供しないので、`torch.autograd.Function` で backward も grouped GEMM で書く。`grad_x = grad_out @ W`、`grad_W = grad_outᵀ @ x`(2D×2D モードは offsets が縮約次元を分割し、グループごとの勾配を返す)。
- **`.tolist()` 全廃**: dispatch 境界(cumsum)が device 上に留まるので、grouped 経路は `@torch._dynamo.disable` なしで compile にトレースさせられる。
- CPU 参照実装(monkeypatch)に対して、**出力・入力勾配・全 expert パラメータ勾配・expert counts の一致**をテストで固定した。

---

## 罠: 最初のベンチは「何も変わらない」

A100 で `--grouped_moe` を付けてベンチを回した最初の結果がこれだ。

| 指標 | ベースライン | --grouped_moe(1回目) |
|---|---|---|
| compile tok/s | 38,054 | 38,845(+2%、run 間ノイズ) |
| peak VRAM | 3,137.0 MiB | **3,137.0 MiB(完全一致)** |
| max loss delta | 0.00154114 | **0.00154114(8桁一致)** |

速くも遅くもない。そして peak VRAM と loss 差が**完全一致**している。別カーネルが本当に動いていれば、bf16 の丸めで loss は微妙に変わり、gather や stacked 重みの分だけ peak も動くはずだ。つまりこれは「grouped GEMM の性能」ではなく、**旧ループへ silent fallback したものをもう一度測っただけ**だった。

原因は有効化判定にあった。`x.dtype == torch.bfloat16` を MoE の入力で見ていたが、**autocast 下では残差ストリームが fp32** のため常に False。3 条件 AND の1つが静かに倒れ、fallback が動き、**ベンチは何のエラーも出さずに「正常な数字」を返した**。

前回の記事(ACT×compile)で「コンパイル代の混入を測定結果から判定できるようにする」という教訓を得たばかりだったが、今回はその一段手前 — **「測りたい経路がそもそも動いたか」を判定できるようにしていなかった**。

## 対策: fail-fast + 観測可能化 + 数値プローブ

3 層で作り直した。

**(1) fail-fast**: 有効化判定を「入力 tensor の観察」から「setup 時の環境判定」に変更し、`--grouped_moe` 要求時に不活性なら理由コード付きで即エラーにする。学習スクリプトも同じ。

```text
[grouped_moe] requested=true active=false reason=dtype_not_bfloat16
RuntimeError: --grouped_moe requested but inactive: dtype_not_bfloat16
```

**(2) 観測可能化**: ベンチの各結果とコンソール出力に `grouped_moe_active` を記録し、JSON に `steady_state_valid`(定常判定 AND 経路有効)を保存。「要求したのに inactive な結果」は定常速度として扱えない構造にした。

**(3) 数値プローブ**: 計測前に、実 API の forward / 入力勾配 / 重み勾配を**非正方形状 + 空 expert**の条件で `F.linear` 参照値と比較する。実際、active化直後のrunは79,396 tok/sまで伸びた一方、first lossが14.60から13.97へ崩れており、正方形の本番重みに隠れた転置規約の誤りを発見した。このrunは性能集計から除外した。転置修正後、A100の整列済みプローブは全delta 0.0、有効runのfirst lossはlegacyと0.0003未満で一致した。

weight-gradient用の2D kernelには16-byte row-stride制約もある。総routing行数が境界を満たさない場合は最終expertへゼロ行をpaddingし、offsetを延長してからgradientを計算する。7 routing行の回帰テストで参照gradientとの一致を確認している。

## 結果: 1.99 倍、品質同等

転置修正後、本番相当条件(batch 16、seq 256、8 loops、bf16、gradient checkpointing、steps=warmup=100)で2回:

| 指標 | ベースライン(compile) | grouped MoE(compile) |
|---|---|---|
| compile tok/s | 38,054(ベースライン中央値) | **75,903 / 75,205(約1.98〜1.99×)** |
| eager tok/s | 35,360(ベースライン中央値) | 63,587 / 63,220 |
| Dynamo graphs/breaks | 8 / 1 | 8 / 1(増加なし) |
| 計測区間の新規 graph | 0 | 0(両run) |

最後の品質ゲートとして、学習済み checkpoint の **WikiText-103 PPL を grouped 有効/無効で比較**(同一チャンクの決定的評価):

| | legacy | grouped |
|---|---|---|
| PPL | 85.47 | **85.36(-0.13%)** |

差はノイズ水準(grouped 経路は bf16 で通す精度ポリシーのため厳密同値にはならないが、実データへの影響は無視できる)。**本番採用を確定**し、Colab notebook は bf16 対応 GPU + API 存在時のみ `--grouped_moe` を自動付与するようにした。

ただし、この採用判断は学習サイズのtoken batchに限定される。A100のbatch 1 chat生成（prompt 64、生成32 token、4 loops）ではlegacy 46.82 tok/sに対してgrouped 37.51 tok/s（0.801倍）で、peak VRAMも461.1 MiBから547.9 MiBへ増えた。出力IDは完全一致したため数値不具合ではなく、小さいrouting行数ではweight stackとkernel起動コストを償却できないことが原因と考えられる。interactive chatはlegacyを既定とし、groupedはbatch serving向けのopt-inに留めた。

---

## 学び

1. **最適化の「ON/OFF」は計測結果から判定可能にする**。フォールバック付きの最適化は堅牢だが、**silent fallback はベンチマークの毒**になる。「要求したのに動いていない」は設定ミスではなく異常として即エラーにし、有効フラグを結果 JSON に残す。
2. **「変化がない」も異常のシグナル**。peak VRAM の MiB 単位一致と loss 差の 8 桁一致が fallback を暴いた。カーネルを置き換えたのに数値が動かないときは、まず「本当に置き換わったか」を疑う。
3. **外部 primitive は計測前に数値プローブで検証する**。転置規約・autograd・空グループのようなドキュメントで確信が持てない点は、非正方形状 + 縮退ケースで参照実装と突き合わせれば数分で白黒がつく。今回は bit 一致という最良の結果だった。
4. **autocast 下の dtype 判定は罠**。残差ストリームは fp32 のままなので、「入力が bf16 か」で機能を切り替えると autocast 環境で常に無効化される。判定は入力の観察ではなく setup 時の環境(device/dtype ポリシー)で行う。

## まとめ

| 項目 | 結果 |
|---|---|
| モデル学習ステップ | ◎ 約1.98〜1.99×(38,054 → 75,903 / 75,205 tok/s) |
| compile 相性 | ◎ graph break 増なし(dynamo.disable も撤廃) |
| checkpoint 互換 | ◎ model state_dict不変、runtime設定は後方互換metadata |
| 数値正しさ | ◎ プローブ全 delta 0.0(bit 一致)、CPU 全勾配一致 |
| 実データ品質 | ◎ PPL 85.47 → 85.36(-0.13%、ノイズ水準) |
| 総合 | **モデル計算時間は約半減。総wall-clockはデータ/I/O込みで別途計測** |

ACT カリキュラム × torch.compile の両立(前回記事)と合わせ、次の学習runは`ACT_CURRICULUM + compile + grouped MoE`で実行する。総wall-clockへの効果は、GPU utilization、データ待ち、checkpoint I/Oを含めて記録する。実装・ベンチハーネス・計測JSONはリポジトリにあります: [BushidoMythos](https://github.com/tedorigawa001/BushidoMythos)(`bushido_mythos/main.py`, `training/bench_act_compile.py`, `docs/performance_roadmap.md`)
