# Inventory Demand Forecasting: Kalman Filter vs Global Probabilistic LSTM

AML (CSAI) project, Phase 1. Dataset: **M5 Forecasting - Accuracy** (Walmart), limited subset.

We forecast daily demand for **40 SKUs** (store CA_1, category FOODS) for **28 days ahead** and compare:

| Model | Idea | P10/P90 |
|---|---|---|
| Seasonal naive (lag-7, lag-28) | repeat last week / last 4 weeks | no |
| Gradient boosting | pooled `HistGradientBoostingRegressor` on lag / rolling / calendar features | no |
| Kalman filter (one per SKU) | local linear trend state-space model (+ a weekly-seasonality extension) | yes, from state variance |
| Global LSTM | one network for all SKUs, item embedding, quantile (pinball) loss | yes |

All models use the **same SKUs, same 4 rolling forecast origins and same 28-day horizon**. Metrics: MAE, RMSE, WAPE, pinball loss, P10-P90 coverage.
Product requirements: see the PRD (`docs/` has our notes). Nothing from the PRD's "non-goals" is built.

## Repository structure

```
notebooks/
  01_data_cleaning_eda.ipynb     raw-data audit, cleaning, availability proxy, EDA -> saves the clean file
  02_model_comparison.ipynb      literature review, theory, features, leakage tests, models, backtest, results
src/
  data.py          loading, SKU selection, cleaning, long panel, demand classes
  features.py      leakage-safe lag / rolling / calendar features
  metrics.py       MAE, RMSE, WAPE, pinball, coverage
  baselines.py     seasonal naive + gradient boosting
  kalman.py        Kalman filter class + per-SKU forecasts
  global_model.py  global quantile LSTM (PyTorch)
  backtest.py      rolling origins, cold-start fallback, metric tables
  plots.py         forecast / interval charts
data/
  raw/             M5 csv files (NOT in git, too big)
  processed/       m5_clean_panel.csv  <- the cleaned dataset (in git)
outputs/           predictions.csv, metrics.csv, metrics_by_sku.csv, sku_classes.csv, figures/
docs/              viva Q&A and notes
```

## How to reproduce (raw data -> final report)

1. Install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. Get the M5 data (Kaggle competition `m5-forecasting-accuracy`, accept the rules, download) and put
   `calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv` in `data/raw/`.
3. Run the notebooks **in order** (Jupyter, or `jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_cleaning_eda.ipynb`, then the same for `02_...`).
   - Notebook 01 takes about 1 minute, notebook 02 about 15 minutes on a laptop CPU (the LSTM is trained at 4 origins, plus 2 extra seeds for the stability check).
4. Results appear in `outputs/`. Notebook 02 ends with an acceptance-criteria self-check; every line must say PASS.

The cleaned dataset is already included (`data/processed/m5_clean_panel.csv`), so notebook 02 can be run without the raw files.
Google Colab: upload the repo, run `!pip install -r requirements.txt`, and run the notebooks from the `notebooks/` folder.

## Honest notes / limitations

* The M5 data in `data/raw` was downloaded from a public Hugging Face mirror of the Kaggle files (`denephew/M5_Forecasting`), because the Kaggle API needs a personal login. File shapes match the competition data (30,490 series, 1,941 sales days, 6,841,121 price rows).
* The global model is an **LSTM with quantile loss**, the PRD's allowed fallback to DeepAR/GluonTS.
* `potential_unavailable` is a **proxy** (zero sales and no price that week), not stockout truth. In this subset it only marks pre-launch days.
* The Kalman "weekly" variant is our extension to the PRD's local linear trend model; both are reported.
* Subset of 40 SKUs only; one training seed per reported LSTM run (a seed-stability check is in notebook 02).
