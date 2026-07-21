# Finance behavior eval — fixed-prompt comparison

- prompts: 8  | max_tokens=96 loops=8 temp=0.7 top_k=40 seed=0

> mention != correct: キーワード一致は言及の有無のみを測る rough な指標 (n が小さい点にも注意)。

> 判定: Phase 5 は Phase 1 より形式・反復・リスク語彙の proxy が改善した。ただし、以下の
> 実出力には質問への直接回答、数値の整合性、金融概念の正確さに重大な不足がある。
> この結果だけで金融QA品質または実運用適合を主張しない。


## Aggregate metrics

| metric | phase1_final | phase5_final |
|---|---|---|
| ③ format adherence (EOS termination) ↑ | 0% | 25% |
| non-degenerate rate ↑ | 100% | 100% |
| structured-reasoning rate ↑ | 38% | 88% |
| ④ avg risk concepts (0-6) ↑ | 0.00 | 1.38 |
| avg words | 90.50 | 82.00 |
| repetition rate ↓ | 0.08 | 0.02 |

## ④ Risk-concept coverage

| concept | phase1_final | phase5_final |
|---|---|---|
| stop_loss | 0% | 0% |
| position_sizing | 0% | 0% |
| leverage | 0% | 0% |
| liquidity | 0% | 12% |
| event_risk | 0% | 25% |
| uncertainty | 0% | 100% |

## ② Fixed-prompt outputs


### 1. `high leverage risk`

**phase1_final** (concepts: —)

```
.The American Civil War , for example , has been referred to as " the New Deal of the World Wars " by historian James J. Crumb ) in his early years ( 1885 @-@ 1947 : Cesar Chavez ) . The first major economic analysis on political economy was published during a time when local Republicans were trying strategies regarding free markets or work projects ; one focus group wanted Democratic groups to organize their own campaign committee meetings between different factions under an executive order known only
```
**phase5_final** (concepts: event_risk uncertainty)

```
I'm not sure what the stock is doing - you're looking to do your own investing in FX (or FX) -- which I think will look at most of these as bonds or ETFs over different timeframes for things like health-care costs.... You can easily go back past total returns if there are no other investments available...  That's probably a good idea whether it would yield better investment capital than equity vs corporate earnings? If this was my case, but only when
```

### 2. `overnight volatile position`

**phase1_final** (concepts: —)

```
.The fact of a strong institutional role in the global economy , which has led to an increasing emphasis on public management with regards not only for its activities but also as much as it is expected by many people within the European Union ; other important trends tend to support this assumption : ' What we will take ? Why are you doing our part ? ’= = = International strategy = = =International co @-@ operation advocates state that " globalization must compete against all forms of cultural
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what I mean by "market conditions". That said there are certainly ways to do this - it just changes the fact you have with your brokerage account as a broker like yours anyway! You can get better advice in an attempt to answer that question (your best bet would be to invest more money into stocks instead of something other than buying individual stock) or buy ETFs which provide much worse returns for each fund compared against time spent on them. Also consider whether they
```

### 3. `Fed rate and inflation`

**phase1_final** (concepts: —)

```
.The following table shows the results obtained through the IEC := = Process events in arts @-@ related health issues = =After several contests , most of these problems were settled by their initial selection procedure to identify new forms available for business management or organizational work ( such as design methodology ) ; more recently , few studies have been conducted since 2000 under a specific program at the same time on investment rules which govern individual projects with specialized programs , including employment rights , property ownership laws
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what you mean by "market price". That said there are certainly ways to do this - it's just a possibility depends on the particular circumstances of your investment (which is usually why people have an exchange in their stock market fund). There may also be enough exposure when they will take more risks than others but some funds won't pay much interest at all or part of a day for which most participants would charge interest unless such was available either way so I doubt anyone
```

### 4. `position sizing`

**phase1_final** (concepts: —)

```
.The following table shows the committee seating of delegates with respect to which party members were selected , while those attending regularly are based on a " member 's report " as a result := = = = Resignation in finance = = = =It was agreed by all parties involved at least some form of collective bargaining including funding for construction projects ( i ) would not have been possible until after the introduction or transfer period ; indeed there must also come under criticism over unpaid bills rather
```
**phase5_final** (concepts: liquidity uncertainty)

```
I'm not sure what you mean by "market price". That said there are certainly ways to do this - it just depends on how the market reacts to real-time in pricing is expected for a given stock (which may or might not always tend to have more of an impact).  If prices stay low they will go down as well when people start giving up points at higher rates after fees beginting their bid/ask spread can become important factors to consider than those who think
```

### 5. `Should I add to a losing position?`

**phase1_final** (concepts: —)

```
.The following table shows the results of various research studies on the subject published in New York University 's journal Proceedings , which analyzed 4 @-@ 80 % as well as 24 percentage points for evaluating different methods :( i ) tests ( including selection criteria ) can indicate whether or not to accept one exception only when using statistical analysis techniques other than quantitative models are eliminated ; ( ii ) lower average score quantitatively evaluate 1 – 2= = History with zero access theory = =This
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what the stock is doing - but there are certain inputs you can use when calculating shorting for long term gains (and possibly buying at higher than your option).  The big advantage of using Shorting refers in which puts might affect underlying stocks; if it's much less risky then more volatile or as aggressive "shares" will take over unless other investors find out they're making some bad bets because this doesn't necessarily mean taking all into account without knowing something
```

### 6. `Explain stop-loss placement.`

**phase1_final** (concepts: —)

```
.The system 's main innovation is that the player must complete a minimum of three different standards for both parties , with certain exceptions being due to their ability or weaknesses as well :Costa was originally created in the first phase beginning on June 23 , 2004 when he resigned his post ; instead of returning it only after 4 hours ( 2 days ) , until April 3 at 02 : 00 am on July 6 – 15 , 2006 .= = = = = = = =In January
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what you mean by "market price". That said there are certainly ways to do this - it just depends how much the buyer is willing to pay for a stock or whatever other person has mentioned above; in particular I'd say they were offered more options than me when my broker was offering them on your own 1st day of pre-market trading (if possible). The last trade will have been closed overnight but no one knows their method here so if there's
```

### 7. `How do earnings announcements affect trading risk?`

**phase1_final** (concepts: —)

```
.The following table shows the results obtained through a proxy server , which has been published by the Board of Governors for each group with responsibility for its operations at most in advance or performance on an external level ( if possible ) .= = = = = Internationalization is an international organization ; however to some extent this was merely one based upon other principles : it seeks only to participate as a member state , not directly related to financial management unless there are certain kinds of institutions offered elsewhere under
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what the stock is doing but I think it's important to remember why when you look at a company as an example of how they handle most trades on various exchanges including Facebook or Yahoo! (there are many), so your broker may still consider making an offer for some investors who will simply accept their advice in exchange for a free subscription being made with this service..
```

### 8. `What is liquidity risk in trading?`

**phase1_final** (concepts: —)

```
.The fact of a strong economic position , especially with respect to investor relations was often characterized by the lack of clarity about market conditions for traders who are interested only on large scale or acquisitions ; an assumption has been made against many different processes within industry terms : " Banks says it 's my way to make deals more attractive ... But if I am not going directly into this situation [ you ] think they will have no chance at all right now when their work day ends meet our price
```
**phase5_final** (concepts: event_risk uncertainty)

```
I'm not sure what the stock exchanges work with you on a daily basis - it's usually best to wait for confirmation of some news or speculation about future events -- as there are factors outside of time-frames going against me (although I've never heard this worded correctly).  However if they were only published by day traders then those prices could still fall at their own pace without being written out after many days...
```

---

## 追記: v2 との総合比較 — 2026-07-20

対象 run: `finance_a100_v3_full`(フル最適化構成: bf16 / compile / grouped MoE / fused AdamW /
非同期 Drive コピー、batch 32 × grad_accum 4 × **seq_len 1024**、52,000 steps、
意図した学習量 **6.816B tokens**)。最終wall-clock JSONの3.932B tokensは、最後のresume区間
(step 22,000-52,000)だけの値である。
v2(`finance_a100_v2`、batch 16 × accum 8 × seq 256、1.70B tokens)との比較。
v3のtrain.log上の全経過時間は **13時間47分10秒**(08:40:48-22:27:58)。途中resumeで
約2,900 stepsを再実行している。v2実働約48時間に対して見かけ上約3.5倍短いが、
sequence長、実行構成、処理token数が異なるため、単一最適化の厳密なA/B速度比較ではない。

### WikiText-103 PPL(validation、先頭100/50チャンク)

| checkpoint | v2 (native 256) | v3 (256 で評価) | v3 (native 1024) |
|---|---|---|---|
| phase1 | 66.09 | 79.27 | — |
| phase2(ベスト) | 57.70 | 50.46 | **40.99** |
| phase3 | 64.55 | 58.92 | — |
| phase4 | 71.17 | 71.88 | — |
| phase5 / final | 85.19 | 94.53 | **77.20** |

- 同じ1024窓でのv3内部比較では、Phase 2がベスト40.99、finalが77.20。
- v2はnative 256、v3はnative 1024であり、表のnative列同士は条件が異なるため厳密な直接比較ではない。
- v3 を 256 窓で評価すると悪化して見える(94.53)のは評価条件ミスマッチ。
  **seq 拡張後のモデルは native 長で評価すること**(今後の評価プロトコルの注意点)。
- 忘却比はベスト比 ×1.88(v2 ×1.48)とやや拡大したが、終着点の絶対値は v3 が良い。
  replay 5% は本 run でも有効。

### 金融挙動(Phase5、8プロンプト、seed 0)

| 指標 | v2(3シード範囲) | v3 |
|---|---|---|
| 構造化推論率 | 38〜88%(平均 63%) | 88% |
| 平均リスク概念数 (0-6) | 0.75〜1.12 | **1.38(全シード超え)** |
| 反復率 ↓ | 0.07〜0.08 | **0.02** |
| 概念カバレッジ | uncertainty 中心 | uncertainty 100% + event_risk 25% + liquidity 12% |

- リスク概念言及は v2 のどのシードよりも高く、event_risk / liquidity への言及が初めて出現。
- Phase1(0.00)→ Phase5(1.38)の伸び幅は過去最大。
- v3 挙動は現状 1 シードのため、構造化推論率はシード変動幅(±25pt 規模)を考慮して読むこと。
- `uncertainty` 100%は多くの回答が定型句 `I'm not sure` で始まる影響が大きく、金融推論の正確さを示さない。
- `stop_loss`、`position_sizing`、`leverage`はいずれも0%。対応する直接プロンプトでも有効な説明を生成できていない。
- 生成文には質問との不一致、意味の通らない数値、架空URLがあるため、現段階のPhase 5を金融助言用途へ使用しない。

### 結論

v3 run は、52,000 stepsを約13時間47分で完走し、Phase 1からPhase 5にかけて形式・反復・
リスク語彙proxyを改善した。一般言語PPLはPhase 2が最良で、その後は77.20まで悪化しており、
specializationに伴う忘却が残る。`finance_a100_v3_full/final.pt`は最新の金融特化checkpointだが、
金融回答品質は未合格である。次の採否判定には、複数seedに加えて正解付き金融QA、拒否・安全性、
数値整合性を人手またはrubricで評価する必要がある。

### 正解付きFinance QA追試

12問・3 seedのPhase 2-5比較を実施した結果、全checkpointのpass率は0%で、採用候補はありません。
Phase 5はscore 0.272、必須概念recall 3.7%、計算精度5.6%でした。Phase 4は36/36応答が
`neutral`となり、非退化率0%でした。その後の500-step対照実験でも、forecaster-onlyは
Phase 3のscore 0.270から0.256へ悪化し、balanced 1:1は36/36応答が短いsentiment labelに
なって最大同一応答率27.8%でcollapse gateを超えました。Phase 4は広範なFinance QAの本番
経路から除外し、Phase 5はPhase 3から直接開始します。詳細は
[`finance_qa_phase2_5_analysis.md`](finance_qa_phase2_5_analysis.md)を参照してください。

curated Phase 5の初回500-step pilotはscore 0.310まで上がりましたが、旧flat SFT loaderにより
最大同一応答率22.2%となりました。example-aligned loaderで再実行すると最大同一応答率5.6%、
数値精度16.7%へ改善した一方、score 0.289、必須概念recall 5.3%、pass率0%で不採用です。
次は同じPhase 3起点・data・LR設定で10/50/200 optimizer stepを比較します。
