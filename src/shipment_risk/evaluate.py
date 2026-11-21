"""How far the predictions can be trusted, measured on shipments the model never saw.

Everything here reads ``model.holdout_``: the most recent period, shipped
after the model's training and calibration data ended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .data import split_delivered, validate_shipments
from .features import CATEGORICAL
from .model import DelayModel

CALENDAR_INPUTS = ["ship_month", "ship_weekday", "peak_season", "holidays_in_transit"]
MODEL_COLUMNS = {
    "p_constant": "Overall late rate (no model)",
    "p_history": "Carrier and lane track record",
    "p_logistic": "Logistic regression",
    "p_boosted_raw": "Gradient boosting",
    "p": "Gradient boosting, calibrated",
}


def _holdout(model: DelayModel) -> pd.DataFrame:
    if model.holdout_ is None:
        raise RuntimeError("The model has no held-out period; fit it with train_fraction + calib_fraction below 1")
    return model.holdout_


def _scores(y, p) -> dict:
    y, p = np.asarray(y, dtype=int), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    both = len(np.unique(y)) == 2
    return {"roc_auc": roc_auc_score(y, p) if both and len(np.unique(p)) > 1 else np.nan,
            "average_precision": average_precision_score(y, p) if both else np.nan,
            "brier_score": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1])}


def summary(model: DelayModel) -> dict:
    """Headline accuracy of the calibrated model on the held-out period."""
    h = _holdout(model)
    out = {"shipments": len(h), "late_rate": float(h["late"].mean()), **_scores(h["late"], h["p"])}
    out["brier_baseline"] = float(brier_score_loss(h["late"], np.full(len(h), model.prior_)))
    out["skill"] = 1 - out["brier_score"] / out["brier_baseline"]  # share of the baseline's error removed
    return out


def compare_models(model: DelayModel) -> pd.DataFrame:
    """The model against simpler alternatives on the same held-out shipments.

    Higher is better for ``roc_auc`` and ``average_precision``; lower is
    better for ``brier_score`` and ``log_loss``. The track-record row is what
    a planner gets from a carrier scorecard alone.
    """
    h = _holdout(model)
    return pd.DataFrame([{"model": label, **_scores(h["late"], h[col])} for col, label in MODEL_COLUMNS.items()])


def calibration_table(model: DelayModel, bins: int = 10, column: str = "p") -> pd.DataFrame:
    """Predicted against observed late rate, in equal-sized bands of predicted risk."""
    h = _holdout(model)
    band = pd.qcut(h[column].rank(method="first"), q=bins, labels=False)
    g = h.groupby(band)
    out = pd.DataFrame({"shipments": g.size(), "predicted": g[column].mean(), "observed": g["late"].mean()}).reset_index(drop=True)
    out["gap"] = out["observed"] - out["predicted"]
    return out


def lift_table(model: DelayModel, bins: int = 10) -> pd.DataFrame:
    """How concentrated late shipments are among the highest-risk predictions.

    Shipments are ranked by predicted risk and cut into ``bins`` equal groups,
    riskiest first. ``cumulative_capture`` is the share of all late shipments
    found by working down to that group; ``lift`` is the group's late rate
    over the average.
    """
    h = _holdout(model).sort_values("p", ascending=False).reset_index(drop=True)
    group = (np.arange(len(h)) * bins // len(h)) + 1
    g = h.groupby(group)
    out = pd.DataFrame({"group": g.size().index, "shipments": g.size().to_numpy(), "late": g["late"].sum().to_numpy(),
                        "late_rate": g["late"].mean().to_numpy(), "lowest_probability": g["p"].min().to_numpy()})
    out["lift"] = out["late_rate"] / h["late"].mean()
    out["cumulative_capture"] = out["late"].cumsum() / h["late"].sum()
    out["cumulative_share_reviewed"] = out["shipments"].cumsum() / len(h)
    return out


def threshold_table(model: DelayModel, late_cost: float = 400.0, action_cost: float = 60.0, effectiveness: float = 0.5,
                    thresholds=None) -> pd.DataFrame:
    """What acting on each alert threshold would have been worth on the held-out period.

    Assumes acting on a flagged shipment (expediting, re-routing, warning the
    customer) costs ``action_cost`` and prevents the cost of lateness,
    ``late_cost``, with probability ``effectiveness``.

        net_benefit = late shipments flagged x effectiveness x late_cost - shipments flagged x action_cost

    The three inputs are business assumptions; the table shows how the best
    threshold depends on them. In theory it is worth acting whenever the late
    probability exceeds ``action_cost / (effectiveness x late_cost)``.
    """
    h = _holdout(model)
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    total_late = h["late"].sum()
    rows = []
    for t in thresholds:
        flagged = h[h["p"] >= t]
        caught = flagged["late"].sum()
        rows.append({"threshold": float(t), "flagged": len(flagged), "share_flagged": len(flagged) / len(h),
                     "late_caught": int(caught), "precision": caught / len(flagged) if len(flagged) else np.nan,
                     "recall": caught / total_late if total_late else np.nan,
                     "net_benefit": caught * effectiveness * late_cost - len(flagged) * action_cost})
    out = pd.DataFrame(rows)
    out["best"] = out["net_benefit"] == out["net_benefit"].max()
    return out


def break_even_probability(late_cost: float, action_cost: float, effectiveness: float) -> float:
    """The late probability above which acting pays for itself on average."""
    if late_cost <= 0 or effectiveness <= 0:
        return float("inf")
    return action_cost / (effectiveness * late_cost)


def eta_table(model: DelayModel) -> pd.DataFrame:
    """Whether the delivery-date confidence levels hold on the held-out period.

    For each level: ``coverage`` is the share of shipments that arrived on or
    before the predicted date, which should be at least the level. Dates are
    whole days, so coverage is often well above it: ``coverage_day_before``
    shows what quoting one day earlier would have achieved, and a date is as
    tight as it can be when that falls below the level.
    ``average_days_after_promise`` is how far past the promised date the
    prediction sits on average, and ``meets_target`` allows two points of slack.
    """
    h = _holdout(model)
    rows = []
    for q in model.quantiles:
        col = f"q{round(q * 100)}"
        date = np.floor(h[col])  # whole days: at most 1.6 days late means at most 1
        rows.append({"confidence": q, "coverage": float((h["delay_days"] <= date).mean()),
                     "coverage_day_before": float((h["delay_days"] <= date - 1).mean()),
                     "average_days_after_promise": float(date.mean())})
    out = pd.DataFrame(rows)
    out["meets_target"] = out["coverage"] >= out["confidence"] - 0.02
    return out


def eta_accuracy(model: DelayModel) -> dict:
    """Error of the predicted delivery date against simply trusting the carrier's promise.

    ``mae_model`` uses the median prediction; ``mae_promise`` assumes every
    shipment arrives on its promised date. Both are in days.
    """
    h = _holdout(model)
    q = min(model.quantiles, key=lambda v: abs(v - 0.5))
    pred = np.rint(h[f"q{round(q * 100)}"])  # deliveries happen on whole days
    return {"mae_model": float((h["delay_days"] - pred).abs().mean()), "mae_promise": float(h["delay_days"].abs().mean()),
            "within_one_day_model": float(((h["delay_days"] - pred).abs() <= 1).mean()),
            "within_one_day_promise": float((h["delay_days"].abs() <= 1).mean())}


def by_segment(model: DelayModel, column: str = "carrier") -> pd.DataFrame:
    """Predicted against observed late rate for each carrier, mode or state in the held-out period."""
    h = _holdout(model)
    g = h.groupby(column)
    out = pd.DataFrame({"shipments": g.size(), "observed": g["late"].mean(), "predicted": g["p"].mean()})
    out["gap"] = out["observed"] - out["predicted"]
    return out.sort_values("shipments", ascending=False).reset_index()


def backtest(shipments: pd.DataFrame, folds: int = 4, test_days: int = 60) -> pd.DataFrame:
    """Refit at several past dates and score each on the ``test_days`` that followed.

    Shows whether accuracy is stable over time or was a lucky period. For
    each cut-off the model is trained and calibrated only on shipments
    delivered before it, then scored on shipments tendered in the following
    ``test_days``.
    """
    if "delivered" not in shipments.columns:
        shipments = validate_shipments(shipments)
    delivered, _ = split_delivered(shipments)
    last = delivered["ship_date"].max()
    rows = []
    for k in range(folds, 0, -1):
        cutoff = last - pd.Timedelta(days=test_days * k)
        past = delivered[delivered["delivery_date"] < cutoff]
        future = delivered[(delivered["ship_date"] >= cutoff) & (delivered["ship_date"] < cutoff + pd.Timedelta(days=test_days))]
        if len(past) < 2000 or len(future) < 100 or future["late"].nunique() < 2:
            continue
        m = DelayModel(quantiles=(), train_fraction=0.85, calib_fraction=0.15).fit(past)
        p = m.predict_proba(future)
        rows.append({"cutoff": cutoff, "train_shipments": len(past), "test_shipments": len(future),
                     "late_rate": float(future["late"].mean()), **_scores(future["late"], p),
                     "brier_baseline": float(brier_score_loss(future["late"], np.full(len(future), m.prior_)))})
    if not rows:
        raise ValueError("Not enough history for a backtest; reduce folds or test_days")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- drift
def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index between two samples of one variable.

    Zero means identical distributions. A common reading: below 0.1 stable,
    0.1 to 0.25 a moderate shift, above 0.25 a large shift.
    """
    expected, actual = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if expected.empty or actual.empty:
        return float("nan")
    if expected.dtype == object or str(expected.dtype) in ("category", "str", "string"):
        cats = sorted(set(expected.astype(str)) | set(actual.astype(str)))
        e = expected.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = actual.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(expected.astype(float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected.astype(float), edges)[0] / len(expected)
        a = np.histogram(actual.astype(float), edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(model: DelayModel, recent: pd.DataFrame) -> pd.DataFrame:
    """How far recent shipments have moved from the data the model was trained on.

    One row per model input, plus ``predicted late probability``. ``status``
    is ``stable``, ``moderate shift`` or ``large shift``; calendar inputs are
    marked ``seasonal (expected)``. A large shift in an
    input is a reason to check accuracy and consider refitting.
    """
    X_recent = model.features(recent)
    X_train = model.train_features_
    rows = [{"input": f, "psi": psi(X_train[f].astype(str) if f in CATEGORICAL else X_train[f],
                                    X_recent[f].astype(str) if f in CATEGORICAL else X_recent[f])} for f in model.features_]
    rows.append({"input": "predicted late probability",
                 "psi": psi(pd.Series(model.proba_from_features(X_train)), pd.Series(model.proba_from_features(X_recent)))})
    out = pd.DataFrame(rows)
    out["status"] = np.select([out["psi"] >= 0.25, out["psi"] >= 0.10], ["large shift", "moderate shift"], "stable")
    # Calendar inputs differ between any two periods by construction; that is the season, not drift.
    out.loc[out["input"].isin(CALENDAR_INPUTS), "status"] = "seasonal (expected)"
    order = out["status"].map({"large shift": 0, "moderate shift": 1, "stable": 2, "seasonal (expected)": 3})
    return out.assign(_o=order).sort_values(["_o", "psi"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
