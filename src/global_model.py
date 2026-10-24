"""Global probabilistic model (PRD FR-5, FR-7): one LSTM for all SKUs, trained
with quantile (pinball) loss to output P10 / P50 / P90.

This is the PRD's accepted fallback to DeepAR. Like DeepAR it
  * is ONE model trained on all SKUs together (shared patterns),
  * uses the item id as a static categorical feature (embedding),
  * uses calendar fields known in advance as dynamic features.
"""
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

QUANTILES = [0.1, 0.5, 0.9]
LOOKBACK = 56       # days of history the LSTM looks at
N_COV = 5           # number of calendar features


def make_covariates(cal):
    """Calendar features that are known in advance, as a (n_days, 5) array."""
    dow = cal["dayofweek"].values
    cov = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                    cal["is_weekend"].values, cal["is_event"].values, cal["snap"].values], axis=1)
    return cov.astype(np.float32)


class GlobalLSTM(nn.Module):
    def __init__(self, n_items, horizon=28, hidden=32, emb_dim=4):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(1 + N_COV, hidden, batch_first=True)
        self.embedding = nn.Embedding(n_items, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden + emb_dim + horizon * N_COV, 64), nn.ReLU(),
            nn.Linear(64, horizon * len(QUANTILES)))

    def forward(self, past, future_cov, item):
        # past: (batch, LOOKBACK, 1 + N_COV), future_cov: (batch, horizon, N_COV)
        _, (h, _) = self.lstm(past)
        z = torch.cat([h[-1], self.embedding(item), future_cov.flatten(1)], dim=1)
        return self.head(z).view(-1, self.horizon, len(QUANTILES))


def make_windows(y, cov, starts, horizon):
    """Cut training/prediction windows. For each forecast start t and each SKU:
    past = days t-56 ... t-1, target = days t ... t+27.
    Sales are divided by (1 + mean of the past window) so that big and small
    SKUs look alike to the network (DeepAR does a similar scaling)."""
    n_items = y.shape[1]
    y_win = sliding_window_view(y, LOOKBACK, axis=0)            # (n_days-L+1, n_items, L)
    cov_win = sliding_window_view(cov, LOOKBACK, axis=0)        # (n_days-L+1, N_COV, L)
    starts = np.asarray(starts)

    past_y = y_win[starts - LOOKBACK]                           # (n_t, n_items, L)
    scale = 1.0 + past_y.mean(axis=2)                           # (n_t, n_items)
    past_y = past_y / scale[:, :, None]
    past_cov = np.repeat(cov_win[starts - LOOKBACK].transpose(0, 2, 1)[:, None], n_items, axis=1)
    past = np.concatenate([past_y[..., None], past_cov], axis=3)  # (n_t, n_items, L, 1+N_COV)

    fut_idx = starts[:, None] + np.arange(horizon)[None, :]
    future_cov = np.repeat(cov[fut_idx][:, None], n_items, axis=1)  # (n_t, n_items, H, N_COV)
    item = np.tile(np.arange(n_items), (len(starts), 1))

    if starts.max() + horizon <= len(y):
        target = np.stack([y[fut_idx[i]] for i in range(len(starts))])   # (n_t, H, n_items)
        target = target.transpose(0, 2, 1) / scale[:, :, None]
    else:
        target = None   # prediction time: the future is unknown

    flat = lambda a: a.reshape(-1, *a.shape[2:])
    tensors = [torch.tensor(flat(past), dtype=torch.float32),
               torch.tensor(flat(future_cov), dtype=torch.float32),
               torch.tensor(item.ravel(), dtype=torch.long)]
    if target is not None:
        tensors.append(torch.tensor(target.reshape(-1, horizon), dtype=torch.float32))
    return tensors, scale.ravel()


def pinball_loss(pred, target):
    """Average pinball loss over the 3 quantiles. pred: (B, H, 3), target: (B, H)."""
    q = torch.tensor(QUANTILES).view(1, 1, -1)
    diff = target.unsqueeze(-1) - pred
    return torch.max(q * diff, (q - 1) * diff).mean()


def global_lstm(wide, cal, end, horizon=28, max_epochs=20, patience=3, seed=0):
    """Train on data before `end`, forecast the next `horizon` days."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    y = wide.values.astype(np.float32)
    cov = make_covariates(cal.loc[wide.index])

    # train windows: targets end at least 28 days before the origin
    # validation windows: targets are inside the last 28 days before the origin
    train_starts = np.arange(LOOKBACK, end - 2 * horizon + 1)
    val_starts = np.arange(end - 2 * horizon + 1, end - horizon + 1)
    train_t, _ = make_windows(y[:end], cov[:end], train_starts, horizon)
    val_t, _ = make_windows(y[:end], cov[:end], val_starts, horizon)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_t), batch_size=256, shuffle=True)

    model = GlobalLSTM(n_items=y.shape[1], horizon=horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        for past, fut, item, target in train_loader:
            optimizer.zero_grad()
            loss = pinball_loss(model(past, fut, item), target)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = pinball_loss(model(*val_t[:3]), val_t[3]).item()
        if val_loss < best_loss:                       # early stopping
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)

    # forecast: one window per SKU starting at the origin
    model.eval()
    # (sales are cut at `end`; only the calendar is allowed to look ahead)
    pred_t, scale = make_windows(y[:end], cov[:end + horizon], [end], horizon)
    with torch.no_grad():
        out = model(*pred_t[:3]).numpy()               # (n_items, H, 3)
    out = np.sort(out * scale[:, None, None], axis=2)  # rescale, keep P10 <= P50 <= P90
    out = np.clip(out, 0, None)
    return {"p10": out[:, :, 0].T, "point": out[:, :, 1].T, "p90": out[:, :, 2].T}
