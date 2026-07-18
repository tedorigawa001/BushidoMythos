# Finance behavior eval — fixed-prompt comparison

- prompts: 8  | max_tokens=96 loops=8 temp=0.7 top_k=40 seed=0

> mention != correct: キーワード一致は言及の有無のみを測る rough な指標 (n が小さい点にも注意)。


## Aggregate metrics

| metric | phase1_final | phase5_final |
|---|---|---|
| ③ format adherence (EOS termination) ↑ | 0% | 38% |
| non-degenerate rate ↑ | 100% | 100% |
| structured-reasoning rate ↑ | 25% | 38% |
| ④ avg risk concepts (0-6) ↑ | 0.50 | 1.12 |
| avg words | 89.75 | 68.75 |
| repetition rate ↓ | 0.18 | 0.07 |

## ④ Risk-concept coverage

| concept | phase1_final | phase5_final |
|---|---|---|
| stop_loss | 0% | 0% |
| position_sizing | 12% | 12% |
| leverage | 0% | 0% |
| liquidity | 0% | 0% |
| event_risk | 0% | 0% |
| uncertainty | 38% | 100% |

## ② Fixed-prompt outputs


### 1. `high leverage risk`

**phase1_final** (concepts: position_sizing uncertainty)

```
.The study of the U.S.-based research on the effectiveness of the U.S @-@ based global scale model has led to a wide variety of studies in developing countries , with a focus on the U.S. technology being developed by U.S. President Barack Obama . The studies showed that there is little difference between the US and Europe which would have been more successful for those countries than China or South Africa may not exceed 3 % of all countries ' GDP
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what the stock could go after you but it is very unlikely to happen if your order goes down until you are 100% below their bid price, as a factor of -10%.
```

### 2. `overnight volatile position`

**phase1_final** (concepts: —)

```
.The first phase of the investigation was conducted by the British Commission in October 2002 , which found that both companies were not getting paid for their work because they had no money to pay for their own equipment .In an attempt to prevent an agreement between the agencies involved ( and also known as the Financial Services Agency ) on 28 December 2008 , a joint venture between HM Revenue & Customs and Local Government Management , with responsibility for planning regulations affecting the company 's operations within this period . In
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what the stock could go after you but it is very unlikely to happen if your order goes below $30/year; then they may find that as a result of an error there are several possible outcomes in which an event will occur (or can't wait for confirmation) until some time interval. There's some magic involved when people take risks with this method and I am going through them too, so this doesn't necessarily mean someone who does this all at once
```

### 3. `Fed rate and inflation`

**phase1_final** (concepts: —)

```
.The results of a recent investigation by the University of British Columbia in 2009 were positive , according to the study methodology used . The report was reviewed with a report recommending that the study is not an appropriate explanation for the current application . Studies on the study found that the analysis also suggests that the studies are more reliable than others ; some research colleagues argue that there are no better examples available between a single study ( the study ) or even entirely exclude ( 3 ) or less specifically , but
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what the stock could go after you but it is possible to predict a future event like this for your father... If there was no such call then I would try setting up an exit strategy with the current yield.  You might also ask if you want to buy more of your mother's money than you will get in a year or two over 3 years at most (maybe around 4). The market value can't change between different time frames as well? Depending on
```

### 4. `position sizing`

**phase1_final** (concepts: —)

```
.The first phase of the program was created in December 2009 , which focused on providing a wide @-@ ranging , nonlinear narrative format for each part . In addition to these factors , an initial estimate ( the total cost is estimated at between $ 1 – 2 billion USD or 4 million USD ) per year , including one ( $ 2 / -S ) , a reduction in the costs associated with full @-@ time management practices such as the use of equipment , safety and
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what the stock could go after you but it is possible for anyone else to get into your question just yet again... The answer to this is that if a stock has gone down then it will probably fail in some direction on the day of expiration (your time value) - 1/1000% + 1000€ = 0 or so*(1/(0-10)/2000€ = 612 € For this question we're talking about: 3 years ago our dividend
```

### 5. `Should I add to a losing position?`

**phase1_final** (concepts: —)

```
.The first phase of the investigation was conducted by the British Commission in October 2009 , which found that both companies were not getting paid for their work because they had no experience with these factors . The initial funding came from the British Government 's Department of State ( DAD ) who provided additional benefits to the project through the company ’ s efforts to obtain an agreement on the part @-@ time management of the NAP program .= = = = = Dementia :
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what the stock could go after you but it is possible for anyone else to get into your question just yet again... The answer to this is that if there were no specific answers to the question then you had something to "withhold" on how much information he would ever need to make up his or her own money (i.e., you can't find a way around).
```

### 6. `Explain stop-loss placement.`

**phase1_final** (concepts: —)

```
.The application of a recent case to improve performance in the United States has been criticized for being inconsistent with the methodology used by some other companies , including Apple 's own accountants and the management of the industry . In 2013 , an agreement was made to establish whether an investment company would fund its clients more than they had on board operations ( see below ) or no financial support , but a lack of capital investments have led to it leading to " the cost of having not adequately met
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what the stock could go after you but it is possible to get a professional advice on this question (it might help with understanding it; I can't find an answer here), or it may well make sense in your situation as you suggest. However if you're thinking of using a more efficient method than doing something for a specific purpose there are no guarantees made by a reputable organization which will tell you how much money they have available at all. You cannot do anything
```

### 7. `How do earnings announcements affect trading risk?`

**phase1_final** (concepts: uncertainty)

```
.The results of a recent investigation by the University of British Columbia have focused on providing a theoretical basis for the study , as well as the study with the work done in China , France @-@ based methods which are being developed worldwide . The study found that the analysis also suggests that the studies may not examine human capital outcomes ( see below ) will take place at the conclusion := = = = = = Other approaches to management involves the identification of work objectives ; improving treatment is
```
**phase5_final** (concepts: uncertainty)

```
I'm not sure what the stock could go after you but it is possible to look at different values (e.g. looking for stocks) so I'd suggest reading a bit more on this topic http://www8-dividend.com/what_marketingindia.htm
```

### 8. `What is liquidity risk in trading?`

**phase1_final** (concepts: uncertainty)

```
.The fact that a total of the UGGS personnel , which had been trained for the first time since 2006 has not yet gained approval at all ) also may have been an important goal of saving people ’ s lives throughout the world as they were treated by an advisor who would assist others to improve the quality of life @-@ threatening outcomes without giving them access into the economy through high production costs – a high rate for projects such as the United States National Bank 's
```
**phase5_final** (concepts: position_sizing uncertainty)

```
I'm not sure what the stock could go after you but it's possible to trade with different exchanges (and/or just looking at their options) for a short time period ending next year or two; if the underlying price goes up, say, 1% of the value on date of your contract, plus whatever else happens - they'll see nothing until expiration day later). For example a buyout will have a price of $100 per share / 0.05 = $0
```

---

## 追記: 再学習 run(ACT curriculum)の追加検証 — 2026-07-18

このレポートは 2026-07 の再学習 run(`finance_a100_v2`、ACT curriculum 有効・replay 5%・52,000 steps)の評価。
旧 run の記録は `finance_behavior_report.md` を参照。

### 多シード比較(Phase5、8プロンプト)

構造化推論率は 8 プロンプトでは 1 件 = 12.5% 刻みのため、シード間で大きく揺れる。

| 指標 | seed 0 | seed 1 | seed 2 | 平均 | 旧 run (seed 0) |
|---|---|---|---|---|---|
| 構造化推論率 | 38% | 88% | 62% | 63% | 75% |
| フォーマット追従率 (EOS) | 38% | 12% | 38% | 29% | 38% |
| 平均リスク概念数 (0-6) | 1.12 | 1.12 | 0.75 | 1.00 | 0.62 |
| 反復率 ↓ | 0.07 | 0.08 | 0.08 | 0.08 | 0.02 |

- 構造化推論率の seed 0 での見かけの低下(75%→38%)は**サンプリングノイズ**。旧 run の値は 3 シードの分布(38〜88%)内に収まる。
- **平均リスク概念数は 3 シードすべてで旧 run 以上**(0.75〜1.12 vs 0.62)— 再現性のある改善。

### WikiText-103 PPL(validation 先頭100チャンク、seq_len=256)— 忘却耐性

| checkpoint | PPL |
|---|---|
| phase1 | 66.09 |
| phase2 | 57.70 |
| phase3 | 64.55 |
| phase4 | 71.17 |
| phase5 / final | 85.19 |

- Phase1 → Phase5 の劣化率は **×1.29**。replay なしの旧 run は ×6.6(54.9 → 361.5、full 評価)だった。
- **replay 5% が本番スケール(52k steps)でも破滅的忘却をほぼ防いだ**ことを確認。
  パイロット実験(`qiita_memory_replay.md`)の結論と整合。
- 絶対値は部分評価のため旧 run の full 評価値と直接比較不可。要点は同一条件内の劣化率。

### 結論

再学習 run は旧 v2 の挙動プロファイルを再現しつつ、**リスク概念言及(全シードで上回る)と
忘却耐性(PPL 劣化 ×6.6 → ×1.29)で旧モデルを上回る**。
