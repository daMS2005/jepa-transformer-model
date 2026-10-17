"""Temporal forecasting optimization over detached JEPA representations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.distributions import StudentT
from torch.nn import functional as F

from jepa_transformer.training.runtime import autocast_context, make_grad_scaler


def forecast_loss(pred, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    location = pred.loc if isinstance(pred, StudentT) else pred
    if target.shape != location.shape or valid.shape != location.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("targets must be [B,T,H] and valid must be boolean [B,T]")
    if not valid.any():
        raise ValueError("a forecasting batch must contain at least one labelled row")
    selected = target[valid]
    if not torch.isfinite(selected).all():
        raise ValueError("labelled targets must be finite")
    if isinstance(pred, StudentT):
        distribution = StudentT(pred.df[valid], pred.loc[valid], pred.scale[valid])
        return -distribution.log_prob(selected.to(distribution.loc.dtype)).mean()
    return F.mse_loss(location[valid].float(), selected.float())


def train_transformer_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    precision: str = "float32",
    accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
) -> float:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    scaler = make_grad_scaler(device, precision)
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    target_count = 0
    pending = 0
    for inputs, targets, valid in batches:
        inputs = inputs.detach().to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        with autocast_context(device, precision):
            result = model(inputs)
            primary = forecast_loss(result.pred, targets, valid)
            loss = primary + model.config.moe_loss_weight * result.aux_loss
        if not torch.isfinite(loss):
            raise ValueError("non-finite Transformer loss")
        scaler.scale(loss / accumulation_steps).backward()
        count = int(valid.sum().item())
        weighted_loss += float(primary.detach()) * count
        target_count += count
        pending += 1
        if pending == accumulation_steps:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
    if pending:
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    if target_count == 0:
        raise ValueError("Transformer epoch contained no labelled rows")
    return weighted_loss / target_count


@torch.no_grad()
def evaluate_transformer(model, batches, *, device: torch.device, precision: str = "float32") -> dict[str, object]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_count = 0
    absolute = None
    horizon_count = None
    try:
        for inputs, targets, valid in batches:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with autocast_context(device, precision):
                result = model(inputs)
                loss = forecast_loss(result.pred, targets, valid)
            location = result.pred.loc if isinstance(result.pred, StudentT) else result.pred
            count = int(valid.sum().item())
            total_nll += float(loss) * count
            total_count += count
            error = (location[valid].float() - targets[valid].float()).abs()
            per_horizon = error.sum(0).cpu()
            absolute = per_horizon if absolute is None else absolute + per_horizon
            counts = torch.full_like(per_horizon, len(error), dtype=torch.float32)
            horizon_count = counts if horizon_count is None else horizon_count + counts
    finally:
        model.train(was_training)
    if total_count == 0:
        raise ValueError("evaluation contained no labelled rows")
    return {
        "loss": total_nll / total_count,
        "mae_by_horizon": (absolute / horizon_count).tolist(),
        "labelled_rows": total_count,
    }


def _rank(x: np.ndarray) -> np.ndarray:
    return x.argsort(kind="stable").argsort(kind="stable").astype(np.float64)


def _newey_west_t(x: np.ndarray, lags: int) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    e = x - x.mean()
    var = e @ e / len(x)
    for k in range(1, min(lags, len(x) - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / len(x)
    return float(x.mean() / np.sqrt(var / len(x))) if var > 0 else float("nan")


@torch.no_grad()
def forecast_ic(model, dataset, *, jepa=None, device: torch.device, precision: str = "float32", batch_size: int = 8,
                limit_batches: int | None = None) -> dict[str, list[float]]:
    """Rank IC of the forecast location against realized labels, per horizon.

    ``ic`` is the mean daily cross-sectional rank IC (pooled over stocks) with a
    Newey-West t (lags = horizon in sessions); ``per_stock_ic`` is the mean of
    each stock's time-series rank IC (stocks with at least 20 rows). With ``jepa``, raw
    feature windows are encoded by it first.
    """
    was_training = model.training
    model.eval()
    days, stocks, preds, labels = [], [], [], []
    count = len(dataset) if limit_batches is None else min(len(dataset), limit_batches * batch_size)
    try:
        for low in range(0, count, batch_size):
            index = range(low, min(low + batch_size, count))
            x, y, valid = (torch.stack(t) for t in zip(*(dataset[i] for i in index)))
            x = x.to(device)
            with autocast_context(device, precision):
                if jepa is not None:
                    x = jepa.encode(x.flatten(0, 1)).view(*x.shape[:2], -1)
                pred = model(x).pred
            location = (pred.loc if isinstance(pred, StudentT) else pred).float().cpu().numpy()
            for j, i in enumerate(index):
                rows = dataset.order[dataset.starts[i] + np.arange(dataset.length)]
                keep = valid[j].numpy()
                days.append(dataset.rows.day[rows][keep])
                stocks.append(dataset.rows.con_id[rows][keep])
                preds.append(location[j][keep])
                labels.append(y[j].numpy()[keep])
    finally:
        model.train(was_training)
    day, stock, pred, label = (np.concatenate(v) for v in (days, stocks, preds, labels))
    horizons = [dataset.rows.meta["horizons"][c] for c in dataset.columns]
    out = {"horizons": horizons, "ic": [], "ic_t": [], "per_stock_ic": [], "rows": int(len(day))}
    for h, horizon in enumerate(horizons):
        daily = []
        for d in np.unique(day):
            sel = day == d
            if sel.sum() >= 10:
                daily.append(np.corrcoef(_rank(pred[sel, h]), _rank(label[sel, h]))[0, 1])
        per_stock = []
        for s in np.unique(stock):
            sel = stock == s
            if sel.sum() >= 20:
                per_stock.append(np.corrcoef(_rank(pred[sel, h]), _rank(label[sel, h]))[0, 1])
        daily = np.asarray(daily)
        out["ic"].append(float(np.nanmean(daily)) if len(daily) else float("nan"))
        out["ic_t"].append(_newey_west_t(daily, horizon))
        out["per_stock_ic"].append(float(np.nanmean(per_stock)) if per_stock else float("nan"))
    return out
