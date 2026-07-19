# BushidoMythos Performance Roadmap

この文書は、メモリ効率と実行性能に関する残りの改善候補を、実装優先度と検証条件を含めて整理したものです。モデル品質とcheckpoint互換性を維持し、現在の本番workloadに対する総wall-clock短縮を優先します。

## 現在の基準点

以下は実装済みです。

- GQA/MLAのPyTorch scaled dot-product attention（SDPA）対応
- 通常の学習・prefillにおける明示的な`T x T`因果maskの削減
- 予約ストレージを使うKV cache。保存済みの過去はdetachし、現在チャンクのK/Vは勾配を維持する
- 生成時の最終位置だけの語彙projection
- recurrent loopのgradient checkpointing
- CUDA AMP（float16/bfloat16）、fused/8-bit AdamW、`torch.compile`
- ACT curriculumのcompile-safe tensor buffer（A100定常計測済み、総wall-clock計測待ち）
- dataset build、batch wait、optimizer、checkpoint save、総wall-clockの学習ログとJSON
- ColabローカルNVMeへのatomic checkpoint保存と単一workerによる非同期Driveコピー
- GQAとMLAによるKV cache圧縮

最適化前後の比較では、同じcheckpoint、入力、loop数、dtype、deviceを使用します。速度だけでなく、logits一致、損失、perplexity、生成品質も確認します。

## Runtime互換性の方針

当面は次の二重経路を明示的に維持します。

| 環境 | 位置付け | 方針 |
|---|---|---|
| Python 3.9 + PyTorch 2.2 | 互換性baseline | 現行test suiteとSDPA fallbackを維持する |
| Colab + 新しいPyTorch | CUDA最適化baseline | native GQAなど利用可能な高速経路をcapability判定で使う |
| 現行ローカルPython 3.12 + PyTorch 2.2 | CPU正当性確認 | CUDA/MPS性能の基準には使わない |

新しい最適化をPyTorch 2.2の制約へ合わせて設計する必要はありません。新経路とfallbackの数値一致をテストし、checkpoint形式は共通に保ちます。Colab計測ではPython、PyTorch、CUDA、GPU名を結果へ必ず記録します。Python 3.9対応を終了するかは次のmajor release前に再判断し、その時点でColab、ローカルCPU/MPS、配布先の利用状況を確認します。

## 優先順位

| 優先度 | 改善 | 主な対象 | 期待効果 | 実装リスク |
|---|---|---|---|---|
| P0 | CUDAベンチマーク基盤 | 学習・推論 | 改善効果と退行の可視化 | 低 |
| P1 | ACT curriculumとcompileの両立 | 学習 | ACTを維持したままcompileを本番runへ復帰 | 中 |
| P1 | ローカルcheckpoint保存と非同期Driveコピー | 学習 | checkpoint I/OによるGPU停止時間を削減 | 低〜中 |
| P1 | MoE grouped GEMM | 学習・prefill | kernel数、CPU同期、Python overhead削減 | 中〜高 |
| P1 | Fused Linear Cross Entropy | 学習 | `B x T x vocab` logitsのピークVRAM削減 | 中 |
| P2 | GQA native SDPA | 学習・推論 | `repeat_interleave()`によるKV展開削減 | 中 |
| P2 | ACTの実計算スキップ | 推論 | 平均loop数に応じた計算削減 | 高 |
| P2 | Fused optimizer | 学習 | optimizer step短縮、kernel数削減 | 低 |
| P3 | FSDP/ZeRO・expert parallel | 大規模学習 | 複数GPUへの重み・状態・expert分散 | 高 |
| P3 | GPU向け低ビット推論 | 配布・推論 | 重みメモリと帯域削減 | 高 |
| P3（条件付き） | MLA latent-native decode | 長文脈推論 | 長いdecodeでのKV復元計算を削減 | 高 |

## P0: CUDAベンチマーク基盤（ハーネス実装・A100定常計測済み）

最初に、変更前後を同じ条件で比較できるハーネスを用意します。CPU結果からCUDA性能やVRAMを外挿しません。

記録する指標:

- prefill tokens/secとpeak allocated/reserved VRAM
- decode tokens/sec、1 token当たりのlatency、最終KV cacheサイズ
- 学習tokens/sec、optimizer step時間、peak VRAM
- Attention、MoE、LM head、optimizerのCUDA profiler時間
- 学習中のGPU utilization、batch取得待ち時間、data preparation時間
- phase開始時のdataset構築・tokenize・cache load時間
- checkpoint serialize、ローカル保存、Drive同期の各所要時間
- loop数、sequence長、batch、dtype、Attention種別ごとの結果
- 同一入力に対する最大logit差、loss差、perplexity差

最低限の比較条件:

| 軸 | 候補 |
|---|---|
| GPU | T4でハーネスを検証後、A100で本計測 |
| dtype | float16、bfloat16 |
| Attention | GQA、MLA |
| sequence length | 64、128、256 |
| loop | 1、4、8 |
| workload | prefill、decode、forward/backward |

学習済みモデルのRoPEテーブルは`max_seq_len=256`相当のため、1024や4096を通常の比較行列へ入れません。長文脈計測は、RoPE frequency tableの拡張、位置外挿の仕様、perplexityと生成品質の検証を別実験として完了した後に追加します。

合格条件は、品質指標が許容範囲内で、対象指標が複数回の中央値で改善することです。初回compileとdataset downloadはkernel benchmarkから除外しますが、end-to-end phase時間ではdataset準備、batch待ち、checkpoint保存を含めます。GPU utilizationが低い場合は、kernel最適化より入力・I/O側を先に改善します。

`training/bench_act_compile.py`は同一checkpoint・乱数batch・ACTランプでeager/compileのforward/backwardを比較し、初回compileをwarmupへ分離します。JSONにはruntime、tokens/sec、peak VRAM、loss差に加え、Dynamo graph数とgraph breakをwarmup区間・計測区間の別に保存します。計測区間の`measured_unique_graphs`が0でない結果は、残存compile時間を含むため定常速度とは扱いません。通常学習側にはdataset build、`data_wait`、checkpoint save時間のログを実装済みです。

A100での最初の予備計測（batch 1、steps 20、warmup 3、seq 256、8 loops、gradient checkpointing）はcompile 0.451倍でしたが、旧ハーネスではgraph生成時点を分離できず、定常速度を判定できませんでした。続く本番相当の再計測（batch 16、steps 100、warmup 100）は3回すべて計測区間の追加graph/breakが0でした。中央値はeager 35,360 tokens/sec、compile 38,054 tokens/sec（1.075倍）、peak VRAM 3292/3137 MiB、最大loss差0.00154114です。5%のkernel benchmark合格基準を満たしたため、対応A100 runではcompileを有効にします。ただし総wall-clock効果はoptimizer、data wait、checkpoint I/Oを含むphase時間で別途確認します。

## P1: ACT curriculumとtorch.compileの両立（実装・A100定常計測済み）

### 優先理由

ACT curriculumを維持したまま`torch.compile`を本番runへ復帰できるようにし、個別kernelだけでなく52k step全体のforward/backwardへ効かせます。当初想定した1.3〜1.5倍には届かなかったものの、本番batch形状の定常計測で1.075倍を確認しました。

### 改善案

- `act_threshold`とponder weightをdevice上の0次元tensor bufferとして保持する。
- forward graphはbuffer値を読み、Python属性値をguard対象にしない。
- curriculum更新は`torch.no_grad()`でbufferへcopyする。
- ACT設定を変化させながらcompile回数とgraph breakを記録する。

compile回数、学習tokens/sec、loss、平均loop数、resume後のcurriculum位置を検証します。bufferをcheckpointへ保存するか、global stepから再計算するかも仕様として固定します。

実装ではbufferを非永続としてcheckpoint schemaを維持し、`cfg`値をログ互換用に同期します。Python 3.9/PyTorch 2.2のDynamo counterではbuffer更新後の追加compileが発生せず、A100の定常計測でも計測区間の追加graphは0でした。次は実際のphaseログで、dataset、optimizer、checkpoint I/Oを含む総wall-clock効果を確認します。

## P1: ローカルcheckpoint保存と非同期Driveコピー（実装済み・本番計測待ち）

ColabのDriveへ大きなcheckpointを直接、頻繁に保存すると、serializeとnetwork filesystem書き込みの間に学習が停止します。

実装:

1. checkpointをColabローカルNVMeへatomic saveする。
2. 学習プロセスを止めず、完了済みファイルを別workerでDriveへcopyする。
3. copy workerは1本に制限し、投入順を維持する。Drive側の古い中間checkpointはcopy完了後にrotationする。
4. phase final checkpointはDriveへのcopy完了を確認してからphase完了とする。
5. queue投入、pending数、copy完了時間、失敗理由を必ずログへ出す。copy失敗は次の投入またはflushで学習プロセスへ伝播させる。

`--local_ckpt_dir /content/checkpoints/<run>`を指定した場合だけ非同期経路を有効にし、`--ckpt_dir`はresume元かつ耐久保存先のまま維持します。`wall_clock_phaseN.json`にはphase/総wall-clock、dataset build、data wait、optimizer、serialize、copy累計、最大queue深度を保存します。hard runtime lossではpending中のローカルファイルを失うため、phase finalのflushとDrive上の直近periodic checkpointを耐障害性の境界とします。

次の本番runで、直接Drive保存との総wall-clock差、foreground serialize時間、background copy時間、最大queue深度、Driveからのresumeを確認します。`max_queue_depth`が継続的に増える場合は保存間隔またはcheckpointサイズを見直します。

## P1: MoE grouped GEMM

### 現状

`MoEFFN.forward()`はexpert IDでtokenをsortした後、expertごとのPythonループでFFNを実行します。`counts_int.tolist()`によるGPUからCPUへの同期があり、`@torch._dynamo.disable`によってMoE全体がcompile対象外です。

### 改善案

1. expertのgate/up/down重みをexpert軸付きtensorへ統合する。
2. token dispatchをGPU上で完結させる。
3. grouped GEMMまたはTriton kernelでactive expertをまとめて計算する。
4. load balancing用countをGPU上に保持し、optimizer更新時だけ集約する。
5. MoEから`@torch._dynamo.disable`を除き、compile互換性を確認する。

候補実装は、PyTorch/Tritonのgrouped GEMM、MegaBlocksなどです。新規依存を導入する場合は、未導入時に現在の実装へ戻れるfallbackを残します。

PyTorch 2.11のnative `torch.nn.functional.grouped_mm`を使うruntime経路を実装しました。既存`ModuleList`のexpert重みをforward時にstackするためmodel `state_dict` schemaは変わりません。checkpoint payloadには後方互換な`runtime_config.grouped_moe`を追加しています。primitive自体はautogradを提供しないため、forward、input gradient、weight gradientをgrouped GEMMで計算するcustom autogradを追加しています。A100 BF16かつ`--grouped_moe`指定時だけ有効です。未指定時は従来loopを使いますが、指定時にinactiveならsilent fallbackせず、`api_unavailable`、`device_not_cuda`、`dtype_not_bfloat16`、`cuda_unavailable`、`compute_capability_lt_80`の理由を1行出力して即時エラーにします。JSONの各resultにも`grouped_moe_active`を保存します。CPU参照kernelではforward、全parameter/input gradient、expert countの一致を確認済みです。さらに測定前に非正方行列でnative forward、input gradient、weight gradientを`F.linear`参照値と比較し、転置方向やautograd実装が不一致なら即時エラーにします。最大絶対差はJSONの`config.grouped_moe_probe`へ保存します。

A100定常計測（batch 16、steps=warmup=100、bf16、8 loops、gradient checkpointing、`steady_state_valid: true`）のうち、重み転置修正後の有効な2回はcompile 75,903 / 75,205 tokens/sec、eager 63,587 / 63,220 tokens/secでした。full-logits compileベースライン中央値38,054 tokens/sec比で約1.98〜1.99倍です。79,396 tokens/sec / first loss 13.97だった先行runは転置規約の不具合を含むため集計から除外しました。有効runのfirst lossはlegacyと0.0003未満で一致し、graph/breakはベースラインと同じ8/1、計測区間の追加graph/breakは0です。A100の整列済み数値プローブはforward・input gradient・weight gradientとも最大絶対差0.0でした。さらに16-byte境界を満たさない7 routing行はcustom backwardで最終expertへゼロpaddingし、参照gradientと一致する回帰テストを追加しました。実データでのperplexity同等確認も完了しました。`eval_perplexity.py --seq_len 256 --max_chunks 100 --dtype bfloat16`（WikiText-103 validation、A100、同一チャンクの決定的評価）で、legacy 85.47に対しgrouped 85.36（差-0.13%）でした。**本番採用を確定**します。Colab notebookはbf16対応GPUかつ`grouped_mm` APIが存在する環境でのみ`--grouped_moe`を自動付与し、学習子プロセスの非zero終了をセル失敗として伝播します。checkpointは`runtime_config.grouped_moe`を記録し、resume時の実行モード不一致を警告します。

A100、batch 16、seq 256、8 loops、gradient checkpoint、full-logits CEの同条件比較では、compiled throughputがlegacyの37,564 tok/sから75,903 tok/sへ向上しました（2.021倍、計測時間50.5%削減）。compiled peak VRAMは3,137 MiBから3,147 MiBへの10 MiB増です。数値probeのoutput/input gradient/weight gradient差はすべて0.0で、legacyとのfirst loss差はeager 0.000130、compile 0.000229でした。計測区間の追加graph/breakも0です。この環境ではruntime preflight成功を条件に本番学習へ採用します。

一方、A100のbatch 1 chat decode（prompt 64、生成32 token、4 loops、実`model.generate()`経路）ではlegacy 46.82 tok/s・461.1 MiBに対しgrouped 37.51 tok/s・547.9 MiBで、`0.801x`（19.9%低速、+86.8 MiB）でした。出力token IDは完全一致しています。小さいrouting行数では毎回のexpert weight stackとgrouped kernel起動コストを償却できないため、対話chatはlegacyを既定のまま維持します。grouped chatはbatch servingの損益分岐点を将来測るためのopt-inとします。

### 検証

- 同一重み・routing結果でforwardとgradientが一致する。
- expert countとrouter bias更新結果が一致する。
- 空expert、偏ったrouting、top-k境界を含む。
- 小batchでは退行し得るため、複数token数で損益分岐点を測る。

## P1: Fused Linear Cross Entropy

### 現状

学習ではLM headが`(B, T, vocab_size)`の全logitsを生成し、その後Cross Entropyを計算します。sequence長と語彙数が大きいほど、このtensorがactivation memoryを支配します。

### 改善案

- LM headとCross Entropyを融合し、全logitsを保持せずlossとgradientを計算する。
- 外部kernelを使わないfallbackとして、sequence/token軸をchunk分割してCross Entropyを加算する。
- SFTの`loss_mask`を融合経路でも維持する。
- 通常の`forward()` APIは全logitsを返し、training scriptだけ融合loss APIを使用する。

候補にはLiger Kernel系のfused linear cross entropyがあります。ただしcustom autogradを導入する場合は、weight tyingとmixed precisionのgradient精度を重点的に検証します。

追加依存なしのfallbackとして、tied LM headとCEをtoken軸でchunk化し、各chunkをactivation checkpointする経路を実装しました。forwardでchunk logitsを破棄してbackward時にLM headを再計算するため、保持logitsは`B*T*vocab_size`から最大`ce_chunk_size*vocab_size`になります。通常の`forward()`は全logitsを返し、`training/finance_pretrain.py --ce_chunk_size N`を指定した場合だけhidden-state経路を使用します。既定0は従来動作です。

CPU fp32ではmaskなし・部分mask・全maskについてloss、hidden gradient、tied weight gradientの一致を確認済みです。A100の`ce_chunk_size=1024`計測では、compile経路が33,561 tokens/sec、peak 1353 MiB、計測区間の追加graph/breakは0、loss差0.00154114でした。full-logits compile（37,564 tokens/sec、3137 MiB）比で速度は10.7%低下し、peak VRAMは1784 MiB（56.9%）減りました。このfallbackは通常runの高速化としては不採用とし、OOM回避またはmicrobatch拡大で総throughputを回復できる場合に限定します。通常は`ce_chunk_size=0`を維持し、速度とメモリを同時改善する候補としてLiger等のCUDA fused kernelを別途評価します。

### 検証

- fp32でlossとgradientがbaselineに一致する。
- float16/bfloat16でNaNがなく、許容誤差内に収まる。
- SFT maskあり・なし、全mask、部分maskを含む。
- checkpoint save/load後もembeddingとLM headのweight tyingが維持される。

## P2: GQA native SDPA（実装・A100計測済み）

現在のSDPA fallbackはPyTorch 2.2互換性のため、K/V headを`repeat_interleave()`でquery head数まで展開します。native GQAを利用できるPyTorch 2.5以降では、`enable_gqa=True`相当の経路を使い、この展開を削減できます。

PyTorchの`scaled_dot_product_attention`が`enable_gqa`を公開し、deviceがCUDAの場合にnative経路を自動選択します。Python 3.9 + PyTorch 2.2やCPU/MPSでは現在の`repeat_interleave()`経路を維持します。Flash Attention 2が利用可能な同長prefillでは既存のFlash Attention経路を優先し、chunked decodeなどその対象外ではnative SDPAを使用します。

学習起動時は`[native_gqa] active=... reason=...`を必ず1行出力します。`training/bench_gqa_sdpa.py`はnativeを要求したのにinactiveなら即時エラーとし、legacy/nativeのforward/backward throughput、peak VRAM、output差、Q/K/V gradient差をJSONへ記録します。

A100 BF16、batch 16、sequence 256、12 query heads/4 KV headsの初回計測では、legacy 8,826,091 tokens/secに対してnative 15,864,937 tokens/sec（1.798倍）でした。peak allocationは76.4から68.4 MiBへ8.0 MiB（約10.5%）減少し、outputとQ/K/V gradientの最大差はすべて0でした。nativeを本番既定として維持します。ただしこれはGQA単体のforward/backward結果であり、総学習wall-clock効果はphase JSONで別に確認します。

## P2: ACTの実計算スキップ（全row halt経路実装・A100計測済み）

### 現状

ACTはhalt済み位置のhidden更新を止めますが、密なAttentionとMoEにはその位置も入力されます。cacheなしでは全位置halt時にloopを終了できましたが、KV cache利用時は将来tokenが全loop depthのcacheを必要とするため、従来はhalt後もfull blockを実行していました。

### 改善候補

- decode時にsequence単位でhaltし、終了したbatch行をactive batchから外す。
- 必要loop数が近いrequestをbucket化する。
- trainingではponder lossとloop curriculumで平均loop数を下げる。
- token単位packingはcausal Attentionとcache indexを複雑化するため、最後に検討する。

品質への影響があるため、固定loop baselineとのperplexity、金融行動評価、平均loop数を必ず併記します。

第一段階として、no-grad inferenceかつKV cache利用時に全batch row/tokenがhaltした場合、残りdepthをK/V projectionとcache appendだけへ縮退させました。query、SDPA、MoE、LoRA、LTI injection、ACT predictorを省略します。halt-stepの正規化済み表現を各depthへ書くためlegacy cacheと一致し、GQA/MLAのpromptと次token decodeでlogits・全cache tensorの完全一致をCPUテスト済みです。学習、gradient-enabled forward、部分halt batchは従来経路のままです。

`training/bench_chat_act_skip.py`はbatch-1 chatでlegacy/cache-onlyを比較し、full loop数、cache-only loop数、skip率、tokens/sec、peak VRAM、output ID一致をJSONへ記録します。phase5 checkpoint、prompt 64、生成32、8 loopsのA100 BF16計測では、1280 loop slotのうち955（74.6%）をcache-only化し、full computeはtokenあたり平均8から2.03 loopsへ減りました。decodeは25.60から62.61 generated tokens/sec（2.446倍）、peakは両方459.4 MiB、output IDは完全一致です。batch-1 chatでは既定有効として採用します。部分batch compactionとrequest bucketingはbatched servingを採用する場合の別段階とします。

## P2: Fused optimizer（実装・A100比較済み）

CUDAでは通常のAdamWに`fused=True`を利用できる環境があります。8-bit AdamWを使わない場合の低リスクな高速化候補です。

現行NotebookはA100/Ampere+でfused AdamW、旧GPUで8-bit AdamWを選択します。fused fp32 AdamWはoptimizer stateのVRAMが増えるため、「fused fp32」と「8-bit」の比較は速度だけでなく、peak VRAMと利用可能batch/sequence長を含む総throughputで判断します。

- `--fused_optimizer`指定時だけcapabilityを検出してfused AdamWを選択する。
- 要求したのに非対応ならfallbackせず即時エラーとする。
- checkpointにbackendを記録する。通常/fused AdamW間は互換stateを復元して警告し、8-bitとの切替はstateをリセットする。
- `training/bench_optimizer.py`で通常、fused、8-bit AdamWの速度・optimizer時間・VRAM・lossを別々に比較する。

比較はA100 BF16、batch 16、sequence 256、8 loops、gradient checkpointing、grouped MoE、`torch.compile`の本番相当条件で実施しました。fusedは55,074 tokens/sec、optimizer 2.278 ms/step、peak 3883.3 MiB、8-bitは47,396 tokens/sec、13.002 ms/step、3322.1 MiBでした。fusedは総throughput 1.162倍、optimizer時間82.5%短縮、VRAMは561.2 MiB増です。A100 40GBではbatchを下げないため新規runの既定として採用します。通常AdamWは51,838 tokens/sec、6.410 ms/step、3883.2 MiBで、fusedは同じVRAMでさらに6.2%高速でした。

初回実行ではColabにbitsandbytesが無く、`OPTIM8BIT=True`が通常AdamWへfallbackしていたことをハーネスが検出しました。以後は`--optim8bit`も要求時fail-fastとし、Notebookの依存導入は`subprocess.run(..., check=True)`とimport/version確認を必須にします。依存導入とベンチは同じ`sys.executable`を使います。

## P3: 複数GPU学習

モデル規模が単一GPUを超える場合は、次の順で検討します。

1. FSDPまたはZeROでparameter、gradient、optimizer stateをshardする。
2. MoE expertをGPU間で分割するexpert parallelを導入する。
3. sequence parallelまたはcontext parallelを長文脈向けに評価する。

recurrent blockは同じparameterを複数回再利用するため、FSDP wrapping境界とgradient checkpointingの組み合わせを慎重に設計します。router biasのexpert countは全rankで集約し、rankごとのrouting driftを防ぎます。

## P3: GPU向け低ビット推論

CPU dynamic INT8とQATの検証結果から、MLA `kv_down`、shared experts、dense FFNの一部は精度感度が高いことが分かっています。GPU量子化では一律INT4/INT8を避け、mixed precisionを前提にします。

優先順:

1. weight-only INT8のCUDA kernelを小規模checkpointで検証する。
2. `kv_down`など高感度層をbf16/fp16に維持する。
3. expert単位・module単位の量子化ablationを行う。
4. finance perplexity、WikiText perplexity、固定prompt、loop深度別劣化を測る。

標準GPTQ/AWQ実装はcustom RDT/MoEへそのまま適用できない可能性があるため、対応範囲を先に確認します。

## P3（条件付き）: MLA latent-native decode

### 現状

MLA cacheは圧縮latent `c_kv`を保存しますが、decodeごとに過去全tokenの`K_nope`と`V`を`kv_up`で再構成します。cache容量は小さくても、sequenceが伸びるほど復元計算と一時tensorが増えます。

現在の主なdecode workloadは8 prompt評価とchat demoで、sequence長も概ね256以下です。この範囲では複雑なcustom経路へ投資する根拠が弱いため、通常ロードマップから外します。`max_seq_len`を拡張し、長文脈サービングを正式な利用形態にすると決めた場合のみ着手します。

### 改善案

MLAのprojection行列をquery側とoutput側へ吸収し、圧縮latentのままAttentionを計算するdecode専用経路を追加します。training/prefill経路はまず現状を維持し、`T=1`のdecodeから段階導入します。

### リスク

- RoPE部分と非RoPE部分の分離が必要。
- value projection吸収後の数値誤差がloopごとに増幅される可能性がある。
- SDPAの標準形から外れるため、custom kernelが必要になる可能性がある。

### 検証

- tokenごとのlogitsを既存MLA decodeと比較する。
- 生成長を伸ばして誤差の累積を確認する。
- finance perplexityと固定prompt生成を比較する。
- cacheサイズだけでなく、decode時の一時VRAMとtokens/secを測る。

## 推奨実装順

1. 無料T4でCUDA benchmark、data wait、checkpoint I/Oの計測を固定する。
2. ACT curriculumと`torch.compile`を両立させ、A100投入前にT4でgraph再利用を確認する。
3. A100でACT+compileの総wall-clock効果を測る。
4. Fused Linear Cross Entropyをtraining script限定で導入する。
5. MoE grouped GEMMをfallback付きで導入する。
6. checkpointのローカル保存と非同期Drive copyを本番計測し、Drive resumeを検証する。
7. native GQAとfused optimizerを環境別に評価する。
8. 単一GPUの結果を基準に、分散学習とGPU量子化へ進む。
9. 長文脈サービングを採用した場合のみ、MLA latent-native decodeを再評価する。

各段階を独立した変更として扱い、性能改善と品質非劣化を同じレポートに記録します。複数の最適化を同時に入れると原因を分離できないため、benchmark、実装、回帰テスト、品質評価の順で1項目ずつ進めます。
