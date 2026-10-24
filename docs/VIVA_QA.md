# Viva Q&A cheat sheet (answers use our real results)

## Project and data
**What problem do you solve?** Forecast daily demand for 40 SKUs for the next 28 days, with uncertainty (P10/P50/P90), and compare a local model (Kalman) with a global model (LSTM) and simple baselines under identical backtests.

**Why this subset?** Full M5 = 30,490 series. PRD asks for 30-50 SKUs from one store/category: CA_1 FOODS, 40 SKUs, chosen evenly by average sales among SKUs that start selling in the first 200 days and sell >= 0.3 units/day.

**What was "dirty" in the dataset?** Wide format with d_1..d_1941 columns, event columns mostly NaN (meaning "no event"), 20% of item-week prices missing (item not on sale), 68% zero cells overall, items that start selling late. We melted to long format, joined dates/calendar/prices, filled event NaN with "None", kept zeros, and flagged missing price. No duplicates / negative values were found.

**Did you remove outliers?** No. We only flag them (`high_sales_flag`). Spikes are usually real; removing them would make results look better than reality.

**What is `potential_unavailable`?** Zero sales AND no price that week. It is a proxy, not stockout truth. In our subset all 679 flagged days are before the SKU's first sale (pre-launch); none fall in test windows. We say this openly.

**Why not replace zeros with estimated demand?** We cannot verify they are stockouts; replacing would invent data.

## Features and leakage
**What is data leakage and how did you avoid it?** Using information from after the forecast origin. Lags use `shift(k)`; rolling means use `shift(1)` first; the LSTM only sees sales before the origin (only the calendar is allowed to look ahead, because holidays are known in advance). Two tests: (1) change all sales from day t onward, features up to t are unchanged; (2) replace all post-origin sales by 999, every model's forecast is unchanged.

**Why lags 1, 7, 14, 28?** The autocorrelation plot shows peaks at multiples of 7.

**How does gradient boosting forecast 28 days?** Recursively: predict day 1, feed it back as lag_1 for day 2, etc. Errors can accumulate; that is a known weakness.

## Models
**Kalman filter in one sentence?** A recursive estimator of a hidden state (level, trend): predict with the model, then correct with each new observation using the Kalman gain.

**What do Q and R mean?** Q = process noise (how fast the true level moves), R = measurement noise (day-to-day randomness). Big Q/R ratio -> filter reacts fast. We set R from the variance of sales around the weekday average over the last year; Q = fixed fractions of R (level 0.05R, trend 0.0005R). No optimiser (PRD says keep it simple).

**Why is your Kalman interval so wide?** Forecast variance grows each step and our noise rule is conservative: coverage 95% vs the ideal 80%, mean width 10.6 vs 6.0 for the LSTM.

**Why did level+trend Kalman do badly?** It ignores the weekly cycle and extrapolates the trend. At origin 3 one fast seller made it forecast 2,878 units vs 1,423 actual. The weekly extension (our addition) improved WAPE from 0.785 to 0.676.

**Why LSTM and not DeepAR?** The PRD allows a global quantile LSTM as fallback. DeepAR (GluonTS) samples from a fitted distribution (e.g. negative binomial); ours predicts P10/P50/P90 directly with pinball loss. Setup risk was the reason for the fallback.

**What is in the LSTM?** Input: last 56 days (sales divided by 1 + mean of the window, plus weekday sin/cos, weekend, event, SNAP), LSTM(32) -> concatenated with a 4-dim item embedding and the next 28 days' calendar features -> MLP -> 28 x 3 quantiles. Adam, lr 1e-3, early stopping on the last 28 days before the origin (patience 3). Trained once per origin on past data only.

**What is pinball loss?** max(q(y-yhat), (q-1)(y-yhat)). With q=0.9, under-forecasts cost 9x more than over-forecasts, so the minimiser is the 90th percentile.

**Why do you sort the three LSTM outputs?** To make sure P10 <= P50 <= P90 (quantile crossing can happen).

## Evaluation
**What is rolling-origin evaluation?** 4 forecast origins 28 days apart (training ends 2016-01-31, 02-28, 03-27, 04-24); at each, models use only the past and forecast the next 28 days. Better than one split because it shows stability.

**Metrics?** MAE, RMSE (punishes big misses), WAPE = sum|error| / sum(actual), pinball loss at 0.1/0.5/0.9, coverage of P10-P90. WAPE is NaN if actuals sum to 0 (guarded).

**Results?** LSTM: MAE 1.555, RMSE 3.533, WAPE 0.499 | Gradient boosting: 1.763 / 3.672 / 0.566 | Seasonal naive lag-7: 1.940 / 4.310 / 0.623 | Kalman weekly: 2.108 / 5.025 / 0.676 | Kalman level+trend: 2.445 / 6.475 / 0.785. LSTM seeds: WAPE 0.501 +/- 0.003.

**Is the LSTM win just because it predicts the median and MAE likes medians?** Partly possible for MAE, but it also wins RMSE (3.53 vs 3.67), where the gap is smaller. We say so in the notebook.

**When does Kalman help?** In days 1-7 the weekly Kalman (WAPE 0.567) beats seasonal naive (0.635). Its error grows with horizon (0.786 by day 22-28) while the LSTM stays about flat. It was not best in any demand class.

**Cold start?** SKU with < 56 days of selling history gets the median demand per weekday of established SKUs over the last 28 days. None of our SKUs triggers it, so we tested it by pretending a SKU was new (notebook section 6).

## Honest limitations
40 SKUs / one store; 4 windows (Feb-May 2016); no hyper-parameter search; Gaussian Kalman intervals on count data; single data subset; data downloaded from a public mirror of the Kaggle files (same shapes as the competition files).
