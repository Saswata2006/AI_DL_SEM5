"""Plot helpers for forecast / interval charts (PRD FR-9)."""
import matplotlib.pyplot as plt

COLORS = {"Seasonal naive (lag-7)": "tab:gray", "Seasonal naive (lag-28)": "tab:brown",
          "Gradient boosting": "tab:green", "Kalman (level+trend)": "tab:orange",
          "Kalman (+weekly)": "tab:red", "Global LSTM (quantile)": "tab:purple"}


def plot_sku_forecast(wide, preds, item_id, origin, models, band_model=None, history_days=42, ax=None):
    """Actual demand vs the forecast of each model for one SKU and one origin.
    If band_model is given, its P10-P90 range is shaded."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    g = preds[(preds["item_id"] == item_id) & (preds["origin"] == origin)]
    hist = wide.loc[:origin, item_id].tail(history_days)

    ax.plot(hist.index, hist.values, color="black", lw=1, label="history")
    actual = g[g["model"] == models[0]].sort_values("date")
    ax.plot(actual["date"], actual["actual"], color="black", lw=2, label="actual")
    for m in models:
        d = g[g["model"] == m].sort_values("date")
        ax.plot(d["date"], d["y_pred"], lw=1.5, color=COLORS.get(m), label=m)
    if band_model is not None:
        d = g[g["model"] == band_model].sort_values("date")
        ax.fill_between(d["date"], d["p10"], d["p90"], alpha=0.2, color=COLORS.get(band_model),
                        label=f"{band_model} P10-P90")
    ax.axvline(origin, color="gray", ls="--")
    ax.set_title(f"{item_id} - forecast origin {origin.date()}")
    ax.set_ylabel("units sold")
    ax.legend(fontsize=7, ncol=2)
    return ax
