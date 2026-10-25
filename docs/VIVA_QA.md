# Viva preparation: questions an examiner will ask, with answers from our code and results

Compiled October 25, 2026, for the Phase 1 (machine learning) and Phase 2 (deep learning) evaluations. Every number is sourced from `outputs/metrics.csv`, `outputs/predictions.csv` (recomputed), the notebooks' committed outputs, or `AML-DL-Report.pdf`; code is cited as `file:line`. Read section 1 first, then your own area in section 5.

---

## 1. Summary: how to use this document

The viva is individual. Every member will be asked about their own area in depth, and anyone can be asked about the core of the project. So everyone must be able to answer section 2 without notes, and each member should go deep on their row in section 5.

Examiners in this course grade against the rubric in the project brief. Phase 1 covers literature review, dataset quality and EDA, feature engineering, theoretical rigour, model application, GitHub repository and code quality, presentation, viva and the IEEE report. Phase 2 covers architecture logic, hybrid innovation, ablation studies, deep learning literature, DL dataset and regularisation, technical validation, theoretical rigour, repository, presentation, viva and report. Sections 3 and 4 follow those headings in that order.

Three habits score well. Answer with a number and say where it came from ("WAPE 0.499, from `outputs/metrics.csv`"). Say what a design choice costs as well as what it buys. And when something was not done, say so plainly and give the concrete plan; examiners punish bluffing far more than a stated gap.

| Member | Area to own | Go deepest on |
|---|---|---|
| Sayan Bhattacharya | Cleaning pipeline | Q 3.2.1 to 3.2.6, section 5.1 |
| Aryu Rao | Demand classification, data audit | Q 3.2.7 to 3.2.9, Q 3.4.6, section 5.2 |
| Debasish Karan | Feature engineering | Q 3.3.1 to 3.3.8, section 5.3 |
| Saswataduity Bhuin | EDA | Q 3.2.10 to 3.2.14, section 5.4 |
| Aryan Vibhuti | Model training | Q 3.5, all of section 4, section 5.5 |

Source: roles from `AML-DL-Report.pdf`, Appendix B.

---

## 2. The numbers everyone must know cold

| Fact | Value | Source |
|---|---|---|
| Data | M5, store CA_1, category FOODS, 40 SKUs, 1,941 days (2011-01-29 to 2016-05-22) | `data/processed/m5_clean_panel.csv` |
| Panel | 77,640 rows by 24 columns | same |
| Zero SKU-days | 42.9% in the panel, 68.0% in the full raw M5 | panel; report Table II |
| Protocol | 4 origins, 28 days apart, 28-day horizon, 4,480 scored SKU-days per model | `outputs/predictions.csv` |
| First and last origin | train until 2016-01-31 and 2016-04-24 | same |
| Best model | Global LSTM: MAE 1.555, RMSE 3.533, WAPE 0.499 | `outputs/metrics.csv` |
| Runner-up | Gradient boosting: WAPE 0.566 | same |
| Naive bar | Lag-7 seasonal naive: WAPE 0.623 | same |
| Kalman | +weekly 0.676, level+trend 0.785, both worse than naive | same |
| Calibration | LSTM P10-P90 coverage 0.812, width 5.98; Kalman 0.945 to 0.950, width 10.60 to 11.11 | metrics and predictions |
| Robustness | LSTM wins all 4 origins, all 4 classes, all 4 horizon weeks, 31 of 40 SKUs | recomputed from predictions |
| Seeds | LSTM WAPE 0.501 ± 0.003 over seeds 0, 1, 2 | notebook 02 section 7.5 |
| Concentration | FOODS_3_090 supplies 42.9% of test units; without it LSTM 0.677, GBM 0.789 | recomputed from predictions |
| LSTM size | 56-day look-back, LSTM(32), 4-dim embedding, MLP 176 to 64 to 84, 22,068 parameters | `src/global_model.py:27` |

If you remember one sentence: under one leakage-tested rolling backtest, a single global quantile LSTM beats per-SKU Kalman filters, pooled gradient boosting and seasonal naive on every view we checked, and it is the only model whose 80% interval covers about 80%.

---

## 3. Phase 1: machine learning

### 3.1 Literature review

**3.1.1 Which papers shaped your design, and how?**
Six strands, each with a job. Kalman (1960) and Harvey (1989) give the state-space filter and the local linear trend model. Croston (1972) and Syntetos, Boylan and Croston (2005) give intermittent-demand theory and the ADI/CV² classes we use to split the errors. Friedman (2001) and LightGBM give gradient boosting. Salinas et al. (DeepAR) and Wen et al. (MQ-RNN) give the global network design. Koenker and Bassett (1978) and Chernozhukov et al. (2010) give pinball loss and the sorting fix for crossing quantiles. Makridakis et al. (2022) and Montero-Manso and Hyndman (2021) frame the local-versus-global question. Report Table I maps each to its use.

**3.1.2 What did the M5 competition find, and does your result agree?**
The M5 organisers report that machine-learning methods, mostly LightGBM models trained across many series, beat the statistical benchmarks, and that cross-series learning was key ([Makridakis et al., 2022](https://doi.org/10.1016/j.ijforecast.2021.11.013)). Our result agrees in direction: both global learners (LSTM and gradient boosting) beat both local Kalman filters. Our study is far smaller, so it confirms the pattern on 40 SKUs rather than proving anything new at scale.

**3.1.3 What is the "research gap" you claim?**
Published comparisons mostly report one pooled score per method on a big benchmark. Few place a local state-space model and a global learned model under identical rolling backtests and then ask where each wins, by demand class, horizon, origin and interval calibration. That disaggregated, reproducible view is our contribution. Be modest: it is a student-scale replication with a careful protocol, not a new method.

**3.1.4 What does Montero-Manso and Hyndman's paper say, in one sentence?**
A global model fitted to a group of series can be as expressive as fitting each series separately, and it can generalise better because its complexity does not grow with the number of series ([2021](https://doi.org/10.1016/j.ijforecast.2021.03.004)). That is the theoretical reason to expect our global LSTM to beat 40 separate filters.

**3.1.5 Which literature did you not follow, and why?**
Croston and SBA should have been a baseline, since 22 of 40 SKUs are intermittent; that is a gap, and we say so in report section IX-B. We did not use DeepAR's negative-binomial likelihood because the GluonTS tooling was a setup risk, and the specification allowed a quantile LSTM as fallback. We did not use the Temporal Fusion Transformer because attention is unlikely to pay for its complexity on 40 series.

### 3.2 Dataset quality and EDA

**3.2.1 Why this subset and not the full 30,490 series?**
The specification asked for 30 to 50 SKUs from one store and one category, so the work fits a laptop and every model can be inspected. We chose CA_1 FOODS because food items are SNAP-eligible, which makes the SNAP flag meaningful.

**3.2.2 How were the 40 SKUs chosen, and is that selection biased?**
`select_skus` (`src/data.py:25`) keeps CA_1 FOODS series that first sell within 200 days and average at least 0.3 units a day, sorts them by mean sales, and takes 40 evenly spaced ones. Yes, it is biased on purpose: it excludes late launches and near-dead items. Evenly spacing by volume gives a mix of fast and slow sellers instead of a random draw dominated by slow items. The cost is that our results say nothing about new products, which is why the cold-start rule never fires on real data.

**3.2.3 What was dirty in the raw data?**
The wide format had 1,941 day columns. Event fields were empty on 1,807 of 1,969 days. 68.0% of all sales cells were zero. 48.5% of series first sold after day 200. Only 79.6% of item-store-weeks had a price. There were no duplicates and no negative values. Source: report Table II.

**3.2.4 What exactly does `build_panel` do?**
Four steps (`src/data.py:50`). It melts wide to long, one row per SKU-day. It joins the calendar, fills empty event fields with `"None"`, derives `is_event`, and picks the SNAP column for the store's state. It left-joins weekly prices, keeping missing prices as missing. Then it adds `price_missing`, `potential_unavailable` and a forward-filled price. `check_panel` (`src/data.py:89`) then asserts no duplicate SKU-dates, no negative sales, all 1,941 days per SKU and no date gaps.

**3.2.5 Why did you not impute zeros or remove outliers?**
A zero can mean "no demand" or "could not be bought", and M5 has no stockout label, so imputing would invent data we can never check. Spikes are flagged (`high_sales_flag`, 165 days, 0.21%) but kept. The largest, 599 units of FOODS_3_090 on 2013-09-01, is part of a multi-week surge, not an error. Removing real spikes would make every model look better than it would perform in a store.

**3.2.6 What is `potential_unavailable`, and what did it find?**
It is 1 when sales are zero and the item had no price that week (`src/data.py:80`). It flags 679 days, all before each SKU's first sale, none mid-history, none in a forecast window. So in this subset it marks launches, not stockouts. It is a diagnostic only, and we call it a proxy everywhere.

**3.2.7 Explain the Syntetos-Boylan classification.**
ADI is the average gap in days between non-zero sales. CV² is the squared coefficient of variation of the non-zero sale sizes. Cut-offs 1.32 and 0.49 give four classes: smooth (low ADI, low CV²), erratic (low ADI, high CV²), intermittent (high ADI, low CV²) and lumpy (high ADI, high CV²). We drop pre-launch zeros first, because zeros before launch are not "no demand" (`src/data.py:111`).

**3.2.8 Why compute the classes on training data only?**
Classes computed on the full history would use the test period to decide how the test errors are grouped, a mild form of leakage. Training-only classes (before 2016-02-01) are 22 intermittent, 9 lumpy, 8 smooth and 1 erratic; on the full history they are 21, 9, 9 and 1. All error analysis uses the training-only version.

**3.2.9 Why does one high-volume SKU count as "intermittent"?**
FOODS_3_090 has zero stretches in its history, so its training-period ADI is 1.345, just over the 1.32 cut-off, with CV² 0.315. It is a borderline case that sells a lot when it sells. It supplies 64.9% of the intermittent class's test volume, which is why that class's low WAPE (0.412 for the LSTM) needs care. Source: recomputed with `demand_class` on data before 2016-02-01 and `outputs/predictions.csv`.

**3.2.10 What are the key EDA findings?**
Daily SKU sales have mean 3.90, SD 14.25, median 1, max 599: sparse, right-skewed counts. Weekend sales are 53% above Monday to Thursday. Autocorrelation peaks at lags 7, 14, 21 and 28. SNAP days lift mean sales about 10% (4.155 vs 3.779). Event days show no average uplift (3.841 vs 3.909). The median price-sales correlation across SKUs is −0.15. Sales drop to near zero each Christmas, when the store closes. Sources: report section IV, notebook 01 sections 8 to 14.

**3.2.11 How did each EDA finding change the design?**
Weekly autocorrelation gave lags at multiples of 7, the lag-7 naive baseline and the weekly Kalman states. A moving level justified a filter that tracks level and slope. The SNAP uplift made SNAP a feature. Sparse counts meant clipping forecasts at zero and using MAE, WAPE and pinball loss. Mixed demand classes justified a global model with an item embedding and per-class reporting. Report Table III lists each pairing.

**3.2.12 Events show no uplift, so why keep `is_event`?**
The average hides opposite effects: some events raise sales and others lower them, and they cancel. The flag is known in advance and cheap, so a model can learn the event-specific effect even though the pooled mean is flat. A sharper version would encode event type rather than one 0/1 flag.

**3.2.13 Why is price not a feature?**
Future prices are not part of the specified feature set, and the price-sales link is weak (median correlation −0.15) because food prices barely move. Adding price, price change and promotions is listed as Phase 2 work with an ablation.

**3.2.14 What does the trend plot show, and why does it matter?**
Total sales rise in late 2011 as more SKUs launch, spike in August to September 2013 (FOODS_3_090), drift afterwards, and drop each Christmas. A non-stationary level is the reason to test a model, the Kalman filter, that tracks a moving level instead of assuming a fixed mean.

### 3.3 Feature engineering

**3.3.1 List the features and justify each.**
Twelve in all (`src/features.py:10`). `lag_1`, `lag_7`, `lag_14`, `lag_28` carry the recent value and weekly shape. `roll_mean_7`, `roll_mean_28` carry the recent level. `dayofweek`, `is_weekend`, `month`, `is_event`, `snap` are calendar features known in advance. `item_code` is a categorical id that lets one pooled model learn SKU-specific levels.

**3.3.2 Define data leakage, and show me where your code prevents it.**
Leakage is using information at training or prediction time that would not be available when the real forecast is made. Lags use `shift(k)`. Rolling means call `shift(1)` before `rolling(w)` (`src/features.py:38`), so the window for day t covers t−w to t−1. If you rolled first and shifted later, the window would include day t itself.

**3.3.3 How do you know there is no leakage? "We were careful" is not an answer.**
Two automated tests in notebook 02. The feature test adds 1,000 to every sale from day 1,500 onward and checks that all features up to day 1,500 are unchanged while day 1,501's features do change, which proves the test can detect leakage. The model test replaces every sale after the last origin with 999, reruns all six models and checks their forecasts are identical to 1e-4. Both pass.

**3.3.4 Gradient boosting needs `lag_1` for day 2 of the forecast. Where does it come from?**
From its own forecast. `gradient_boosting` (`src/baselines.py:47`) forecasts recursively: predict day 1, clip at zero, write it into the history, rebuild features for day 2, repeat 28 times. It never sees a real future value. The cost is that errors compound across the horizon.

**3.3.5 Why `history.tail(40)` inside the recursive loop?**
To build the features for one new day, the longest look-back is 28 days for `lag_28` and `roll_mean_28`, so 29 rows are enough. Forty rows leave a safety margin and keep each of the 28 rebuilds cheap.

**3.3.6 Why is `item_code` declared categorical instead of numeric?**
SKU ids have no order. As a number, a tree would split on "item_code < 17", which is meaningless. As a native categorical, `HistGradientBoostingRegressor` can group SKUs into arbitrary sets at a split.

**3.3.7 How does the LSTM see the same information?**
As sequences instead of columns: 56 days of scaled sales plus five calendar covariates per day, and the same covariates for the 28 forecast days. Day of week is encoded as sine and cosine of 2π·dow/7 (`src/global_model.py:19`), so Sunday and Monday sit next to each other instead of at opposite ends of a 0 to 6 scale.

**3.3.8 What features would you add next?**
Price and price change, a rolling standard deviation, days since launch, and event type instead of one flag. Each should come with an ablation that removes one feature group at a time and reports the change in WAPE.

### 3.4 Theoretical rigour (ML)

**3.4.1 Write the Kalman filter equations and explain the gain.**
Predict: x̂ = F x, P = F P Fᵀ + Q. Update: S = H P Hᵀ + R, K = P Hᵀ S⁻¹, x = x̂ + K (y − H x̂), P = (I − K H) P (`src/kalman.py:23`). The gain weighs the new observation against the model's prediction. If Q is large relative to R, the model is uncertain and K is large, so the filter follows the data. If R is large, the data are noisy and K is small, so the filter smooths.

**3.4.2 When is the Kalman filter optimal, and is it optimal here?**
It is the minimum mean-squared-error estimator when the system is linear and both noises are Gaussian; without Gaussianity it is still the best linear estimator. Here the data are non-negative, zero-heavy counts, so the Gaussian assumption is wrong. That is one reason its intervals are too wide.

**3.4.3 How did you set Q and R, and what is the weakness?**
A fixed rule, not maximum likelihood (`src/kalman.py:50`). R is the variance of the last 365 training days around their weekday means, floored at 0.1. Q is 0.05R for the level and 0.0005R for the slope, plus 0.001R on the seasonal state in the weekly model. Nothing is fitted to the data, so the noise ratio is a guess. Fitting Q and R by likelihood (Durbin and Koopman) is the obvious fix.

**3.4.4 How does the weekly Kalman model represent seasonality?**
Six extra states hold the last six seasonal effects. The new effect is minus the sum of the previous six plus noise (`src/kalman.py:69`), so the seven weekday effects sum to about zero. The observation adds the level and the current seasonal effect: H = (1, 0, 1, 0, 0, 0, 0, 0).

**3.4.5 How is the local linear trend model related to exponential smoothing?**
In steady state, when the gain stops changing, the local linear trend Kalman filter produces the same forecasts as Holt's linear trend method, with smoothing constants set by Q/R. So the filter is a principled, probabilistic version of a familiar smoother, and it gives a forecast variance for free.

**3.4.6 Why is MAPE not used?**
MAPE divides by each actual, so it is undefined on the 40.3% of test days with zero sales. WAPE divides total absolute error by total sales, so it stays defined unless every actual in a group is zero, in which case `wape` returns `NaN` (`src/metrics.py:13`).

**3.4.7 What does a WAPE above 1.0 mean?**
Forecasting zero every day gives WAPE exactly 1.0, because the absolute error then equals the actual. So above 1.0 means worse than predicting nothing. On lumpy SKUs every model except the LSTM (0.839) is above 1.0.

**3.4.8 Which point forecast does each metric reward?**
MAE is minimised by the median, MSE and RMSE by the mean ([Kolassa, 2020](https://doi.org/10.1016/j.ijforecast.2019.02.017)). Gradient boosting with squared-error loss estimates a mean; the LSTM's point forecast is its P50, a median. On right-skewed counts the median sits below the mean, so MAE and WAPE favour the LSTM. It also wins RMSE, but by less (3.533 against 3.672), so part of its MAE lead reflects the statistic, not skill.

**3.4.9 Explain gradient boosting in one paragraph.**
Start from a constant prediction. At each of M = 200 rounds, fit a shallow tree to the negative gradient of the loss at the current prediction, which under squared error is the residual, and add it scaled by the learning rate 0.1. The result is an additive ensemble that does gradient descent in function space ([Friedman, 2001](https://doi.org/10.1214/aos/1013203451)). The "histogram" version bins each feature into at most 255 values first (scikit-learn's default `max_bins`), which makes split-finding fast.

**3.4.10 Why rolling-origin evaluation instead of one train/test split?**
One split gives one number that might reflect a lucky or unlucky month. Four origins show whether the ranking is stable: here the LSTM wins at all four, while gradient boosting loses to naive at the first. It also mirrors deployment, where the model is refitted as data arrive ([Tashman, 2000](https://doi.org/10.1016/S0169-2070(00)00065-0)).

**3.4.11 Is the LSTM's win statistically significant?**
We did not run a formal test. The evidence is consistency: it wins at every origin, class and horizon week, on 31 of 40 SKUs, across three seeds, and with the dominant SKU removed. The right formal test is Diebold-Mariano on the daily loss differences between the LSTM and gradient boosting, and it is listed as next work.

### 3.5 Model application

**3.5.1 Walk me through `run_backtest`.**
For each origin (`src/backtest.py:43`) it takes the actuals for the next 28 days and finds cold-start SKUs (fewer than 56 days since first sale). For each model it calls `model(wide, cal, end, 28)`, overwrites cold-start SKUs with the fallback forecast and blanks their intervals, and appends a tidy table of origin, date, step, SKU, model, actual, forecast, P10, P90 and the cold-start flag. Every model sees the same SKUs, origins and horizon.

**3.5.2 Why did the level+trend Kalman fail so badly at origin 3?**
A short run of high sales of FOODS_3_090 just before the origin pushed the slope state up. The forecast multiplies by F every step, so a positive slope adds the same increment every day for 28 days. It forecast 2,878 units against 1,423 actual, a 102% over-forecast. A damped trend would cap that.

**3.5.3 Where is the Kalman filter useful, if anywhere?**
Only in the first forecast week and only against naive: weekly Kalman WAPE 0.567 against 0.635 for lag-7 naive in days 1 to 7. By days 22 to 28 it degrades to 0.786. It never beats gradient boosting or the LSTM in any week, class or origin.

**3.5.4 Why does gradient boosting beat naive overall but lose on mean per-SKU WAPE?**
Its per-SKU WAPE has median 0.936 (better than naive's 1.102) but mean 1.204 (worse than naive's 1.131). A few slow SKUs get very large relative errors from the pooled model, a heavy tail that the pooled metric hides because those SKUs carry little volume.

**3.5.5 How does the cold-start fallback work, and how did you test it?**
A SKU with fewer than 56 days of selling history gets, for each forecast day, the median sales of established SKUs on that weekday over the last 28 days, with no interval (`src/backtest.py:31`). No real SKU triggers it, so notebook 02 section 6 erases all but the last ten days of one SKU and checks that every model returns the fallback for it and leaves the other SKUs unchanged.

### 3.6 GitHub repository and code quality

**3.6.1 How is the code organised?**
Two notebooks drive eight modules in `src/`: data, features, baselines, kalman, global_model, backtest, metrics, plots. Notebooks hold the narrative and the tests; modules hold reusable logic. The clean panel is committed so notebook 02 runs without the raw files.

**3.6.2 What single design decision makes the code easy to extend?**
The shared model interface `model(wide, cal, end, horizon) -> {"point", "p10", "p90"}`. A new model is one function. The backtest, the leakage test and the fairness check cover it automatically.

**3.6.3 How do you know the metric and Kalman code are correct?**
Notebook 02 section 4.2 checks each metric against hand-computed values on a tiny example. Section 4.3 runs the Kalman filter on three synthetic series with known answers: a constant level, a linear trend (it recovers slope 0.484 against a true 0.5) and a weekly pattern. Section 12 runs an 11-point acceptance check, all PASS.

**3.6.4 Is the run reproducible?**
Mostly. Seeds are fixed (`torch.manual_seed`, `np.random.seed`, `random_state=0`), the clean data is committed, and the backtest runs in about 4 minutes (239 s) on a laptop CPU. Two weaknesses: `requirements.txt` sets only minimum versions with no lockfile, and the code uses paths relative to `notebooks/`, so it must be run from there.

**3.6.5 What would you improve in the code?**
Move the notebook tests into a `pytest` suite so they run in seconds without the notebook, pin exact versions in a lockfile, resolve paths from the module location instead of the working directory, and add a quantile-loss gradient boosting model so the comparison with the LSTM uses the same point statistic.

### 3.7 Presentation and report

**3.7.1 State your three research questions and the answers.**
RQ1: does a global model beat per-SKU state-space models and baselines? Yes, on every view. RQ2: when is the local filter competitive? Only in week one, and only against naive. RQ3: how well calibrated are the intervals? The LSTM is close to nominal (0.812); the Kalman filters over-cover (0.945 to 0.950) at nearly twice the width.

**3.7.2 What are the threats to validity in your report?**
Construct: the LSTM forecasts a median, the others a mean. Internal: untuned models, heuristic Kalman noise, LSTM validation windows that overlap training targets, no significance test. External: one store, one category, 40 SKUs, four windows in February to May 2016, data from a mirror. Report section IX-C.

**3.7.3 If a retailer asked which model to deploy tomorrow, what would you say?**
The global LSTM if they can run PyTorch, because it is the most accurate and its intervals can set safety stock directly. Otherwise gradient boosting: second on every pooled metric, trains in seconds, and costs about 13% more WAPE (0.566 against 0.499). Not the Kalman filter as configured here.

---

## 4. Phase 2: deep learning

Be clear about status. The deep model built and evaluated so far is the global quantile LSTM. Ablations, extra regularisation and a hybrid model are planned, not done (report section X). Answer those questions with what exists, what you would do, and how you would know it worked.

### 4.1 Architecture logic

**4.1.1 Draw the architecture and justify each block.**
Input: 56 days by 6 features (scaled sales plus 5 calendar covariates). An LSTM with 32 hidden units encodes the history into its final hidden state. A 4-dimensional item embedding adds SKU identity. The 28 future days' calendar covariates are flattened (140 values) and concatenated, giving 176 values. A two-layer MLP, 176 to 64 to 84 with ReLU, outputs 28 days by 3 quantiles (`src/global_model.py:27`). The LSTM summarises the past, the embedding says which SKU, the future covariates say what is coming, and the head maps all three to quantiles.

**4.1.2 Why an LSTM and not a plain RNN?**
A plain RNN's gradients shrink or blow up as they pass back through 56 steps. The LSTM's cell state is updated additively through the forget gate, m = f ⊙ m_prev + i ⊙ c̃, so gradients can flow across the window and the weekly pattern 7, 14 and 21 days back can still be learned ([Hochreiter and Schmidhuber, 1997](https://doi.org/10.1162/neco.1997.9.8.1735)).

**4.1.3 Why not a GRU, a TCN or a Transformer?**
A GRU would likely perform about the same with fewer parameters, and it is a fair ablation. A temporal convolutional network is a reasonable alternative for a fixed 56-day window. A Transformer such as TFT is built for many series and long contexts; with 40 series and about 69,000 heavily overlapping windows it would probably overfit and cost far more to tune. The LSTM is the smallest model that matches the DeepAR family the specification pointed to.

**4.1.4 Why output all 28 days at once instead of one day at a time?**
This is the direct multi-horizon design of MQ-RNN ([Wen et al., 2017](https://arxiv.org/abs/1711.11053)). A recursive model feeds its own predictions back as input, so errors compound and training sees true inputs while forecasting sees predicted ones. Direct output avoids both. The evidence is the horizon table: the LSTM's WAPE stays between 0.468 and 0.517 across the four weeks, while the recursive gradient boosting rises to 0.594 in week two.

**4.1.5 Why scale by 1 + mean of the look-back window?**
Sales range from about 0.1 to 48 units a day across SKUs. Without scaling, the loss would be dominated by the biggest SKU and the network would struggle to share patterns. Dividing by the window mean puts every SKU on a similar scale, as DeepAR does. The +1 avoids dividing by zero for an all-zero window. The scale uses only past data, so it does not leak.

**4.1.6 Why is it valid to multiply the quantiles back by κ?**
Quantiles are equivariant under positive scaling: if q is the α-quantile of y/κ, then κ·q is the α-quantile of y. So predicting quantiles in scaled space and rescaling gives quantiles in units.

**4.1.7 What does the item embedding learn, and why dimension 4?**
A dense 4-number vector per SKU, learned jointly with the network, that lets the shared model adjust for each SKU's behaviour. One-hot would need 40 inputs. Four dimensions for 40 items is a small, untuned choice; varying it, including 0 (no embedding), is the first ablation to run.

**4.1.8 How many parameters, and is that reasonable?**
22,068: 5,120 in the LSTM (4 gates × 32 × (6 + 32) weights plus two bias vectors of 128), 160 in the embedding, 11,328 and 5,460 in the two linear layers. There are about 68,700 training windows at the first origin, though they overlap heavily, so the effective sample size is smaller. Early stopping keeps the fit in check.

**4.1.9 Can your global model forecast a brand-new SKU?**
Not well. Its embedding would be untrained and it needs 56 days of history. The backtest's cold-start rule handles SKUs with fewer than 56 days of history by falling back to a weekday median of established SKUs. A better answer for Phase 2 is to replace the item id with SKU attributes (department, price band) so new items inherit a sensible representation.

### 4.2 Deep learning literature

**4.2.1 Compare DeepAR, MQ-RNN and TFT, and place your model.**
DeepAR ([Salinas et al., 2020](https://doi.org/10.1016/j.ijforecast.2019.07.001)) is an autoregressive LSTM across many series that outputs the parameters of a likelihood, for example negative binomial, and forecasts by sampling. MQ-RNN outputs several quantiles for all horizons directly. TFT ([Lim et al., 2021](https://doi.org/10.1016/j.ijforecast.2021.03.012)) adds attention and variable selection to direct multi-horizon forecasting. Ours takes DeepAR's global training, embedding and scaling, and MQ-RNN's direct quantile output.

**4.2.2 What would DeepAR's negative-binomial likelihood buy you?**
A full predictive distribution for counts: non-negative, discrete and able to put real mass on zero. Any quantile is available, not just three. The cost is an assumed distribution that may be wrong for lumpy SKUs, and sampling at prediction time. It is the planned Phase 2 upgrade, with pinball loss and coverage as the comparison.

**4.2.3 What is a proper scoring rule, and why does it matter?**
A score is proper if a forecaster gets the best expected score by reporting what they truly believe. Pinball loss is proper for quantiles ([Gneiting and Raftery, 2007](https://doi.org/10.1198/016214506000001437)), so training on it rewards honest quantiles rather than hedged ones.

### 4.3 DL dataset and regularisation

**4.3.1 How is the training set built, and what is the train/validation split?**
`make_windows` (`src/global_model.py:44`) cuts sliding windows with `sliding_window_view`. Each window has 56 days of input and 28 days of target, one per SKU per start day. At an origin τ, training starts run from day 56 to τ − 56, so every training target ends at least 29 days before the origin; that is 1,718 starts by 40 SKUs = 68,720 windows at the first origin. The 28 most recent starts form the validation set. The split is by time, never random, because a random split would let the model train on days that come after its validation days.

**4.3.2 Your validation targets overlap the training targets. Is that a problem?**
Yes, a small one, and we say so in the report. Validation windows start in the 28 days after the last training start, so their targets run from 55 days to 1 day before the origin. The latest training targets cover 56 to 29 days before the origin, so the two overlap by up to 27 days, and their inputs overlap too. The validation loss is therefore optimistic. It only affects which epoch is kept and never touches test data, which the 999 test confirms. The Phase 2 fix is a gap so that validation targets are disjoint from training targets.

**4.3.3 What regularisation do you use?**
Only early stopping: at most 20 epochs, patience 3 on validation pinball loss, best weights restored (`src/global_model.py:117`). There is no dropout and no weight decay. The model is small (22,068 parameters) and seed variation is low (WAPE ± 0.003), so overfitting is not severe, but adding explicit regularisation and measuring its effect is planned.

**4.3.4 You want dropout. Where exactly would you add it?**
Not through `nn.LSTM(dropout=...)`: with a single layer that argument does nothing, because PyTorch applies it only between stacked layers. Put `nn.Dropout` on the concatenated vector before the head, or between the two linear layers. Use AdamW for weight decay. Judge each by validation pinball loss and the test coverage.

**4.3.5 Are your training windows independent samples?**
No. Consecutive windows share 55 of 56 input days, and the 40 SKUs share calendar effects. So 68,720 overstates the information content, and that is one more reason to keep the network small and to check stability across seeds.

### 4.4 Technical validation

**4.4.1 How do you show the network does not cheat?**
The end-to-end test replaces every post-origin sale with 999 and reruns the LSTM: forecasts are unchanged to 1e-4. In code, prediction windows take sales only up to the origin (`y[:end]`) and calendar values up to origin + 28 (`cov[:end + horizon]`, `src/global_model.py:129`). The calendar is published in advance, so that look-ahead is legitimate.

**4.4.2 How do you know the result is not a lucky seed?**
The full four-origin backtest was rerun with seeds 0, 1 and 2: WAPE 0.501 ± 0.003, coverage 0.816 ± 0.003. The worst seed (0.504) still beats gradient boosting by 0.062.

**4.4.3 Does 81.2% coverage prove the intervals are good?**
It is necessary, not sufficient. Coverage near 80% with narrow width is what you want, and the LSTM has both: 0.812 at width 5.98. But coverage on zero-heavy counts is lenient: when P10 is clipped to 0 and the actual is 0, the point always counts as covered, and 40.3% of test days are zeros. A stricter check would look at P10 and P90 hit rates separately, a PIT histogram, or CRPS.

**4.4.4 Why sort the three outputs?**
Three quantiles trained independently can cross, for example P10 above P50. Sorting is the monotone rearrangement of [Chernozhukov et al. (2010)](https://doi.org/10.3982/ECTA7880), which removes crossing without making the estimates worse.

**4.4.5 Quick sanity check: why is the LSTM's pinball loss at 0.5 exactly half its MAE?**
At α = 0.5 the pinball loss is 0.5·|y − q|, so its mean is MAE/2: 1.555 / 2 ≈ 0.778, which is what `outputs/metrics.csv` shows. The same holds for the Kalman filters (2.108 / 2 = 1.054). It is a quick way to show the metric code is consistent.

### 4.5 Theoretical rigour (DL)

**4.5.1 Write the LSTM equations.**
Input, forget and output gates: i, f, o = σ(W u + U h_prev + b), each with its own weights. Candidate: c̃ = tanh(W_c u + U_c h_prev + b_c). Cell: m = f ⊙ m_prev + i ⊙ c̃. Hidden: h = o ⊙ tanh(m). The additive cell update is what keeps gradients alive across long windows.

**4.5.2 Prove that pinball loss gives the quantile.**
Take the expected loss E[ρ_α(Y − q)] and differentiate in q. The derivative is −α·P(Y > q) + (1 − α)·P(Y < q). Setting it to zero gives P(Y < q) = α, so q is the α-quantile ([Koenker and Bassett, 1978](https://doi.org/10.2307/1913643)). At α = 0.9 the under-forecast weight 0.9 is 9 times the over-forecast weight 0.1.

**4.5.3 Why Adam, and what do its hyperparameters mean?**
Adam keeps running averages of the gradient and of its square and scales each parameter's step by them, so it needs little tuning and copes with differently scaled inputs ([Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980)). We use learning rate 1e-3 with the default β1 = 0.9 and β2 = 0.999, batch size 256.

**4.5.4 Why does early stopping act as regularisation?**
Training error keeps falling while validation error starts rising once the network fits noise. Stopping at the validation minimum limits how far the weights move from their small initial values, which has an effect similar to an L2 penalty.

### 4.6 Ablation studies

**4.6.1 Which ablations have you run?**
None yet. The LSTM in the repository is one configuration. Say this directly. The comparison with five other models and the seed study are validation, not ablation.

**4.6.2 Which ablations would you run, and how would you judge them?**
In priority order: remove the item embedding; remove the future calendar covariates; change the look-back to 28, 56 and 112 days; remove scaling; swap pinball loss for MSE and check that intervals disappear and P50 shifts toward the mean; add dropout and weight decay. Run each through the same four-origin backtest with three seeds and report pooled WAPE, pinball loss and coverage, mean ± SD. A component matters only if removing it changes WAPE by more than the seed spread of about ± 0.003.

**4.6.3 What do you expect the future-covariate ablation to show?**
A clear loss. The horizon analysis already suggests the calendar is what keeps the LSTM flat across 28 days while the Kalman filter, which has no calendar, degrades. If removing the covariates made the LSTM degrade with the horizon too, that would confirm it.

### 4.7 Hybrid innovation

**4.7.1 Propose a hybrid with a real rationale.**
Feed the Kalman filter's filtered level, slope and predictive variance at each day as three extra input channels to the LSTM. The filter tracks the current level fast, and the LSTM knows the calendar and shares patterns across SKUs. A second option is a residual hybrid where the LSTM models what the filter gets wrong. A simple ensemble that averages forecasts is the weakest option, because it adds no new information.

**4.7.2 What is the leakage trap in that hybrid?**
Use the filter, never the smoother. The Kalman smoother revises each day's state using later observations, so its states leak the future. Filtered states at day t use data up to t only. The 999 test would catch a mistake here.

**4.7.3 The LSTM already beats the Kalman filter in week one. Why would the hybrid help?**
It might not, which is exactly why it needs an ablation. The case for it is that the filter's level adapts within a day to shifts that a 56-day window averages over. The test: hybrid against plain LSTM, same backtest, three seeds, and the hybrid counts only if its WAPE or pinball loss improves by more than the seed spread. Without that evidence it does not qualify as a hybrid contribution.

---

## 5. Deep-dive questions by member

### 5.1 Sayan Bhattacharya, cleaning pipeline

**Why melt instead of keeping the wide format?** Every join (calendar on `d`, prices on store, item and week) and every per-SKU check is simpler on one row per SKU-day, and the models later pivot back to a dates-by-SKUs table with `to_wide`.

**Why a left join for prices?** An inner join would delete SKU-days with no price, which are exactly the days we need to see to build `price_missing` and the availability proxy.

**What does `check_panel` assert, and what would fail if you skipped it?** No duplicate SKU-dates, no negative sales, all 1,941 days per SKU, no gaps in the date index. Without it, a bad join could silently duplicate rows, and the pivot in `to_wide` would then either fail or double-count sales.

**Why forward-fill price only within each SKU?** Filling across SKUs would copy one item's price onto another. `groupby("item_id").ffill()` (`src/data.py:83`) keeps each fill inside its own series.

### 5.2 Aryu Rao, demand classification and data audit

**Where do the 1.32 and 0.49 cut-offs come from?** Syntetos, Boylan and Croston (2005) derived them by comparing when Croston's method, SBA and simple exponential smoothing each give the lowest error ([doi:10.1057/palgrave.jors.2601841](https://doi.org/10.1057/palgrave.jors.2601841)). They are rules of thumb, not laws.

**Compute ADI for a series 0, 3, 0, 0, 2, 0 after launch.** Six days and two non-zero days give ADI = 6 / 2 = 3. CV² of the sizes 3 and 2: mean 2.5, population SD 0.5, CV² = (0.5 / 2.5)² = 0.04. ADI ≥ 1.32 and CV² < 0.49, so intermittent.

**What did the audit find that changed the subset?** 48.5% of all series first sell after day 200, so leading zeros mean "not launched yet". That is why the subset requires an early first sale and why `demand_class` drops pre-launch zeros.

**Why does the classification matter for the results?** A pooled score hides class-specific failure. By class, lumpy SKUs are hard for everyone (WAPE 0.839 to 1.143), and the LSTM wins all four classes.

### 5.3 Debasish Karan, feature engineering

**Show me the line that prevents rolling-mean leakage.** `wide.shift(1).rolling(w).mean()` at `src/features.py:38`. Shift first, then roll.

**Why lags 1, 7, 14 and 28 and not 1 to 28?** The autocorrelation peaks at multiples of 7, so those lags carry most of the signal; lag 1 adds the most recent level. Twenty-eight correlated lags would add noise and training time for little gain.

**Why does the feature table drop rows with missing features?** The first 28 days of each SKU have no `lag_28` or full `roll_mean_28`. `dropna(subset=FEATURES)` (`src/baselines.py:38`) removes them from training only.

**How would you measure feature importance here?** Permutation importance on a held-out origin, measured in WAPE, or an ablation that drops one feature group at a time and reruns the backtest. Impurity-based importance from trees is biased towards high-cardinality features such as `item_code`.

### 5.4 Saswataduity Bhuin, EDA

**How did you establish the weekly pattern?** Two ways: average sales by weekday (weekends 53% above Monday to Thursday) and the autocorrelation of total sales, which peaks at 7, 14, 21 and 28.

**What does the "weekly strength" number mean?** The share of a SKU's variance explained by its weekday means (`src/data.py:133`), after dropping pre-launch zeros. Near 0 means no weekly pattern, near 1 a very strong one.

**Why did SNAP show an uplift but events did not?** SNAP days put food-assistance money in shoppers' hands, which raises food sales across the board (about 10%). Events push in different directions, raising sales on some event days and lowering them on others, and the effects cancel in the average. Splitting by event type in notebook 01 section 11 is the way to show which way each one goes.

**What would you show an examiner to justify a global model from the EDA alone?** The demand-type figure: 22 of 40 SKUs are intermittent and 9 lumpy. A single intermittent series has too few non-zero days to estimate a weekly profile on its own, while a model pooled over 40 series can.

### 5.5 Aryan Vibhuti, model training

**Walk through one training epoch of the LSTM.** Shuffle the training windows into batches of 256. For each batch, run the LSTM over 56 steps, concatenate the last hidden state with the embedding and the future covariates, predict 28 by 3 scaled quantiles, compute pinball loss against the scaled targets, back-propagate and take an Adam step. After the epoch, compute validation pinball loss with gradients off, keep the weights if it improved, and stop after three epochs without improvement.

**Why retrain from scratch at every origin instead of fine-tuning?** It guarantees each origin's model has seen only data before that origin, which keeps the four results independent and leak-free. Fine-tuning from the previous origin would be faster, but it would mix in decisions made with earlier validation sets.

**How long does training take, and what dominates?** The full six-model backtest takes 239 s on a laptop CPU, and notebook 02 takes about 15 minutes including two extra LSTM seeds. LSTM training at four origins dominates.

**Why does the gradient boosting model use squared-error loss, and what would you change?** It was the default, single configuration. A quantile or Poisson objective would match the count data and the LSTM's median, and gives gradient boosting its own intervals. That is the fairest next experiment, and report section IX-B already says the current 0.067 WAPE gap is an upper bound.

**What happens if two quantile outputs cross after rescaling?** Rescaling by a positive κ cannot create a crossing, so sorting after rescaling (`src/global_model.py:132`) fixes any crossing the network produced, then the values are clipped at zero.

---

## 6. Curveballs, and the answer to avoid

**"Isn't 95% coverage better than 81%?"** No. For an 80% interval the target is 80%. Ninety-five per cent means the interval is too wide, and at 10.6 units against 6.0 it would make a store hold far more safety stock than needed. Avoid saying "more coverage is safer".

**"Your best model forecasts a median and the others a mean. Isn't the comparison unfair?"** Partly, and we say so. The LSTM also wins RMSE, which favours means, by a smaller margin. The fair fix is a quantile-loss gradient boosting model. Avoid denying the point.

**"Why not just use LightGBM like the M5 winners?"** It is a strong, reasonable choice, and we would add it with a Tweedie objective and tuning. Our `HistGradientBoostingRegressor` uses the same histogram idea. Avoid claiming the LSTM would beat a tuned LightGBM; we have not tested it.

**"Did you use ChatGPT or similar tools?"** Answer honestly according to your team's actual practice and the course policy. Whatever the answer, be ready to explain any line of code you are shown, because that is what the question is really testing.

**"What would you do with two more weeks?"** Quantile and Poisson gradient boosting with tuning, a Croston/SBA baseline, likelihood-fitted Kalman noise with a damped trend, Diebold-Mariano tests, then the Phase 2 work: a negative-binomial head, the ablations in 4.6.2 and the Kalman-feature hybrid judged by ablation.

---

## 7. Further reading, in order

1. Report sections VI to IX in `AML-DL-Report.pdf`, then [docs/METHODOLOGY.md](METHODOLOGY.md) and [docs/RESULTS.md](RESULTS.md) in this repository.
2. Hyndman and Athanasopoulos, Forecasting: Principles and Practice, chapters on evaluation and ETS: [otexts.com/fpp3](https://otexts.com/fpp3/)
3. Makridakis et al., M5 results paper, 2022: [doi:10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013)
4. Salinas et al., DeepAR, 2020: [doi:10.1016/j.ijforecast.2019.07.001](https://doi.org/10.1016/j.ijforecast.2019.07.001)
5. Wen et al., MQ-RNN, 2017: [arXiv:1711.11053](https://arxiv.org/abs/1711.11053)
6. Kolassa, the best point forecast depends on the error measure, 2020: [doi:10.1016/j.ijforecast.2019.02.017](https://doi.org/10.1016/j.ijforecast.2019.02.017)

---

## 8. Caveats

The answers describe the code at commit `cb61b84` and the committed run (LSTM seed 0). The seed-stability, runtime and synthetic Kalman test figures come from notebook 02's saved output and the report, not from a fresh run on the day this was written. Phase 2 answers about ablations, extra regularisation and the hybrid describe plans, not results; if any of that is implemented before the viva, update sections 4.3, 4.6 and 4.7 with the measured numbers.
