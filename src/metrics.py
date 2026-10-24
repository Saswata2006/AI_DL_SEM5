"""Forecast accuracy metrics: MAE, RMSE, WAPE, pinball loss, coverage."""
import numpy as np


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def wape(y, yhat):
    """Sum of absolute errors / sum of actuals.
    If the actuals add up to 0 the ratio is undefined, so we return NaN
    (these windows are flagged instead of giving a misleading number)."""
    total = np.sum(np.abs(y))
    if total == 0:
        return np.nan
    return float(np.sum(np.abs(y - yhat)) / total)


def pinball(y, q_pred, q):
    """Pinball (quantile) loss for quantile level q.
    Under-forecasting is punished with weight q, over-forecasting with 1-q."""
    diff = y - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y, low, high):
    """Fraction of actual values that fall inside [low, high]."""
    return float(np.mean((y >= low) & (y <= high)))
