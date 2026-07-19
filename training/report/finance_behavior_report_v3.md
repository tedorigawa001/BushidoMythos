# Finance behavior eval — fixed-prompt comparison

- prompts: 8  | max_tokens=96 loops=8 temp=0.7 top_k=40 seed=0

> mention != correct: キーワード一致は言及の有無のみを測る rough な指標 (n が小さい点にも注意)。


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
非同期 Drive コピー、batch 32 × grad_accum 4 × **seq_len 1024**、52,000 steps、3.93B tokens)。
v2(`finance_a100_v2`、batch 16 × accum 8 × seq 256、1.70B tokens)との比較。
wall-clock は v2 実働 ~48h → v3 推定 13〜14h(約 3.5 倍短縮、データ量は 4 倍)。

### WikiText-103 PPL(validation、先頭100/50チャンク)

| checkpoint | v2 (native 256) | v3 (256 で評価) | v3 (native 1024) |
|---|---|---|---|
| phase1 | 66.09 | 79.27 | — |
| phase2(ベスト) | 57.70 | 50.46 | **40.99** |
| phase3 | 64.55 | 58.92 | — |
| phase4 | 71.17 | 71.88 | — |
| phase5 / final | 85.19 | 94.53 | **77.20** |

- v3 native(1024)は**ベスト 40.99・final 77.20 で v2 を明確に上回る**(長文脈化+データ4倍)。
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

### 結論

v3 run は **wall-clock 約 3.5 倍短縮・データ量 4 倍・品質同等以上(native 評価では明確に改善)**。
現行ベストモデルは `finance_a100_v3_full/final.pt`。
