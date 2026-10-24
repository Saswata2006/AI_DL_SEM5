"""Baseline models (PRD FR-3): seasonal naive and gradient boosting.

All model functions in this project have the same interface:
    model(wide, cal, end, horizon) -> dict with "point" (horizon x n_skus array)
    and optionally "p10" / "p90".
`wide` = full sales table (dates x SKUs), `cal` = calendar features,
`end`  = forecast origin: the model may only use rows wide.iloc[:end].
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURES, make_feature_table


def seasonal_naive(wide, cal, end, horizon=28, season=7):
    """Repeat the last `season` days again and again.
    season=7  -> "same weekday last week"
    season=28 -> "same weekday 4 weeks ago" """
    last_season = wide.iloc[end - season:end].values          # (season, n_skus)
    point = np.array([last_season[h % season] for h in range(horizon)])
    return {"point": point}


def seasonal_naive_28(wide, cal, end, horizon=28):
    return seasonal_naive(wide, cal, end, horizon, season=28)


def gradient_boosting(wide, cal, end, horizon=28):
    """One pooled HistGradientBoostingRegressor for all SKUs.

    Training uses only days before `end`. Forecasting is recursive: we predict
    day 1, write the prediction into the history, then build the lag features
    for day 2 from it, and so on. So the 28 forecasts never see real future sales.
    """
    train_wide = wide.iloc[:end]

    # 1. training table (lags are built with shift, so no leakage)
    table = make_feature_table(train_wide, cal).dropna(subset=FEATURES)
    model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.1, max_depth=6,
        categorical_features=[FEATURES.index("item_code")], random_state=0)
    model.fit(table[FEATURES], table["sales"])

    # 2. recursive forecast
    history = train_wide.copy()
    point = []
    for h in range(horizon):
        day = wide.index[end + h]
        history.loc[day] = np.nan                       # tomorrow is unknown
        rows = make_feature_table(history.tail(40), cal)
        rows = rows[rows["date"] == day]                # only the day we predict
        pred = np.clip(model.predict(rows[FEATURES]), 0, None)
        history.loc[day] = pred                         # feed the prediction back
        point.append(pred)
    return {"point": np.array(point)}
