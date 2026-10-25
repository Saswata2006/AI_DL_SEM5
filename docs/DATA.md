# Data card: the M5 subset used in this project

Compiled October 25, 2026. Figures are cited by repository path; external sources are linked inline. Every number here was recomputed from the committed files on that date.

---

## 1. The short answer

The project uses 40 daily sales series from the M5 Forecasting Accuracy dataset: store CA_1, category FOODS, 1,941 days from 2011-01-29 to 2016-05-22. The cleaned panel has 77,640 rows (40 SKUs times 1,941 days) and 24 columns and is committed at `data/processed/m5_clean_panel.csv`, so the modelling notebook runs without the raw files. Source: shape and dates recomputed from that file.

Three facts about this data drive every modelling decision. Zeros dominate: 42.9% of SKU-days in the panel have zero sales. Demand is concentrated: one SKU, FOODS_3_090, supplies 42.9% of all units in the test windows. And the only availability signal M5 offers, a missing weekly price, marks product launches in this subset rather than stockouts. Sources: `data/processed/m5_clean_panel.csv`, `outputs/predictions.csv`.

---

## 2. Where the data comes from

M5 is the Walmart dataset released for the 2020 M5 competition, described by its organisers in [Makridakis, Spiliotis and Assimakopoulos, International Journal of Forecasting, 2022](https://doi.org/10.1016/j.ijforecast.2021.11.013). It has three files: `sales_train_evaluation.csv` (30,490 item-store series by 1,941 days), `calendar.csv` (1,969 rows of dates, events and SNAP days) and `sell_prices.csv` (6,841,121 weekly prices). Source: raw-data audit in `notebooks/01_data_cleaning_eda.ipynb`, section 2, and Table II of `AML-DL-Report.pdf`.

The official copy sits behind a login on [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy). The team downloaded the files from a public Hugging Face mirror, [denephew/M5_Forecasting](https://huggingface.co/datasets/denephew/M5_Forecasting), and checked that their shapes match the competition files exactly. Byte-level identity was not checked. Anyone reusing this repository outside coursework should read the competition rules on Kaggle before redistributing the derived panel.

The raw files are not in git (`data/raw/` is listed in `.gitignore`). Only notebook 01 needs them.

---

## 3. How the 40 SKUs were chosen

`select_skus` in `src/data.py:25` applies four rules. It keeps only CA_1 FOODS series (1,437 of them). It keeps series whose first sale falls within the first 200 days, so every SKU has a long history. It keeps series averaging at least 0.3 units a day, so no series is almost all zeros. Finally it sorts the survivors by mean sales and takes 40 evenly spaced positions, which gives a deliberate mix of fast and slow sellers instead of a random draw dominated by slow items. Sources: `src/data.py:10` to `:47`, report section III-C.

FOODS was picked because its items are SNAP-eligible, which makes the SNAP calendar flag a meaningful feature. The result is 23 FOODS_3, 11 FOODS_2 and 6 FOODS_1 items. Source: department counts recomputed from the panel.

The selection is a design choice with a cost. Requiring an early first sale removes late launches, which nearly half of all M5 series are (48.5% first sell after day 200, report Table II). That is why the cold-start rule in the backtest never fires on real data.

---

## 4. What was dirty, and what we did about it

| Issue in the raw files | Extent | Treatment |
|---|---|---|
| Wide format, one column per day | 1,941 day columns | Melted to one row per SKU-day, calendar joined on `d` |
| Event fields empty | 1,807 of 1,969 days | `NaN` replaced by `"None"`, meaning no event |
| Zero-sales cells | 68.0% of all cells | Kept, never imputed |
| Late first sale | 48.5% of series | Subset restricted to early starters |
| Item-store-weeks with a price | 79.6% | No imputation, `price_missing` flag added |
| Duplicates, negative sales | 0 and 0 | Asserted in `check_panel` |

Source: `AML-DL-Report.pdf` Table II, from notebook 01 section 2.

`build_panel` (`src/data.py:50`) does the melt and joins, and `check_panel` (`src/data.py:89`) raises an `AssertionError` if any SKU-date is duplicated, any sale is negative, any SKU is missing a day, or the date index has a gap. Notebook 01 reloads the saved CSV and confirms it matches the in-memory panel.

Zeros were kept on purpose. A zero can mean nobody wanted the item or the item could not be bought, and M5 does not say which. Replacing zeros with estimated demand would invent data that no model could later be checked against.

---

## 5. Flags, and what they do not mean

`potential_unavailable` is 1 when a SKU-day has zero sales and no price for that Walmart week (`src/data.py:80`). It is a proxy, not a stockout label. In this panel 679 days carry the flag, and all of them fall before the SKU's first sale. None falls mid-history and none falls in a forecast window, so here the flag identifies launches. Source: count recomputed from `data/processed/m5_clean_panel.csv`, location analysis in notebook 01 section 5.

`high_sales_flag` is 1 when a day's sales exceed the SKU's mean by more than five standard deviations. It marks 165 days (0.21%). Flagged days were neither removed nor capped. The largest, 599 units of FOODS_3_090 on 2013-09-01, belongs to a multi-week surge rather than being a recording error, and removing real spikes would make every model look better than it would perform. The flag is computed in notebook 01 section 6, not in `src/`.

`price_missing` is 1 when the weekly price is absent. `sell_price_filled` forward-fills the price within each SKU so it can be plotted, but no model uses price, because future prices are not part of the specified feature set.

---

## 6. Schema of the clean panel

`data/processed/m5_clean_panel.csv` has one row per SKU per day, sorted by `item_id` then `date`.

| Column | Type | Meaning |
|---|---|---|
| `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` | string | M5 identifiers; constant `FOODS`, `CA_1`, `CA` here |
| `date` | date (YYYY-MM-DD) | Calendar day |
| `d` | string | M5 day key, `d_1` to `d_1941` |
| `wm_yr_wk` | int | Walmart week id, the key for prices |
| `weekday`, `wday`, `month`, `year` | mixed | Calendar fields from `calendar.csv` |
| `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2` | string | Event names and types, `"None"` if no event |
| `is_event` | 0/1 | 1 if `event_name_1` is not `"None"` |
| `snap` | 0/1 | SNAP day for the store's state (California) |
| `sales` | int | Units sold, the target |
| `sell_price` | float or empty | Weekly price, empty when the item was not on sale |
| `sell_price_filled` | float | Price forward-filled within the SKU |
| `price_missing` | 0/1 | 1 when `sell_price` is empty |
| `potential_unavailable` | 0/1 | Availability proxy (section 5) |
| `high_sales_flag` | 0/1 | Spike flag (section 5) |

Source: header of `data/processed/m5_clean_panel.csv` and `build_panel` in `src/data.py`.

---

## 7. Schema of the outputs

`outputs/predictions.csv` holds every forecast in tidy form: 26,880 rows, which is 6 models by 4 origins by 28 days by 40 SKUs. Its columns are `origin` (last training day), `date`, `step` (1 to 28), `item_id`, `model`, `actual`, `y_pred` (the point forecast, which is P50 for the LSTM), `p10` and `p90` (empty for models without intervals) and `cold_start` (always `False` in this run). Every other result table can be recomputed from this file, and `docs/RESULTS.md` was built that way.

`outputs/metrics.csv` has one row per model with MAE, RMSE, WAPE, pinball loss at 0.1, 0.5 and 0.9, their average, and P10-P90 coverage. `outputs/metrics_by_sku.csv` has the same columns per model and SKU. `outputs/sku_classes.csv` has, per SKU, the department, mean sales, zero share, ADI, CV², Syntetos-Boylan class and weekly strength, computed on the full history. The error analysis instead uses classes computed on training data only, which differ for one SKU (section 8).

`outputs/figures/` holds 18 PNGs. Figures 01 to 10 come from notebook 01's EDA and figures 11 to 18 from notebook 02's results.

---

## 8. Properties of the target that matter for modelling

| Property | Value | Consequence |
|---|---|---|
| Mean, SD, median, max of daily SKU sales | 3.90, 14.25, 1, 599 | Right-skewed counts: clip forecasts at 0, prefer MAE and WAPE |
| Share of zero SKU-days | 42.9% | No zero imputation; WAPE instead of MAPE |
| Mean sales on SNAP days vs other days | 4.155 vs 3.779 | About 10% uplift, so `snap` is a feature |
| Mean sales on event days vs other days | 3.841 vs 3.909 | No average uplift, but events kept as a known-in-advance flag |
| Demand classes, training period only | 22 intermittent, 9 lumpy, 8 smooth, 1 erratic | Report errors by class |
| Test-window units, share of zeros | 13,961 units, 40.3% zero days | The scoring set |
| Test-volume share of FOODS_3_090 | 42.9% | Pooled metrics need a check without this SKU |

Source: recomputed from `data/processed/m5_clean_panel.csv` and `outputs/predictions.csv`; classes from `demand_class` (`src/data.py:111`) on data before 2016-02-01.

On the full history the class counts are 21 intermittent, 9 lumpy, 9 smooth and 1 erratic (`outputs/sku_classes.csv`). The training-only split is the one to quote, because grouping by a label computed from the test period would leak.

---

## 9. How to regenerate everything

Put `calendar.csv`, `sales_train_evaluation.csv` and `sell_prices.csv` in `data/raw/`, then run `notebooks/01_data_cleaning_eda.ipynb` (about one minute) and `notebooks/02_model_comparison.ipynb` (about 15 minutes on a laptop CPU, most of it LSTM training at four origins plus two extra seeds). Run both from inside `notebooks/`, because the code uses paths relative to that folder. Notebook 02 alone is enough if you only want to reproduce the models, since the clean panel is committed.

---

## 10. Caveats

The data came from a third-party mirror whose shapes match the official release; nobody compared checksums. The subset is one store, one category and 40 SKUs, chosen by rules that exclude late launches, so results say nothing direct about new products or other stores. The availability proxy cannot detect real stockouts in this subset, and M5 has no ground truth for them. The 68.0%, 48.5% and 79.6% audit figures describe the full raw files and come from notebook 01's committed output and the report, not from a fresh run, because the raw files are not in the repository.
