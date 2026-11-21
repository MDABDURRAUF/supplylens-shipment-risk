"""Turning predictions into work: today's risk list, dates to promise, and fair carrier comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from .data import split_delivered, validate_shipments
from .explain import explain_features, top_reasons
from .features import CARRIER_FEATURES, FEATURES, build_features
from .model import DelayModel, _boosted
from .us import METROS, great_circle_miles

TIERS = ["Already late", "High", "Medium", "Low"]


def risk_list(model: DelayModel, in_transit: pd.DataFrame, as_of=None, high: float = 0.50, medium: float = 0.25,
              explain_top: int = 200) -> pd.DataFrame:
    """Every shipment still in transit, riskiest first.

    ``tier`` is ``Already late`` when the promised date has passed without a
    delivery, otherwise ``High`` / ``Medium`` / ``Low`` by late probability.
    ``expected_date`` is the median predicted delivery date and ``safe_date``
    the date nine in ten such shipments arrive by. ``value_at_risk`` is the
    shipment value times its late probability. ``main_reasons`` is filled for
    the ``explain_top`` riskiest shipments.
    """
    if in_transit.empty:
        return pd.DataFrame()
    as_of = pd.Timestamp(as_of) if as_of is not None else in_transit["ship_date"].max()
    X = model.features(in_transit)
    pred = model.predict(in_transit)
    out = in_transit[["shipment_id", "ship_date", "promised_date", "carrier", "mode", "service", "origin_state", "dest_state",
                      "distance_miles", "value_usd"]].copy()
    for c in ("origin", "destination"):
        if c in in_transit.columns:
            out[c] = in_transit[c]
    out["late_probability"] = pred["late_probability"].to_numpy()
    overdue = out["promised_date"] < as_of
    out.loc[overdue, "late_probability"] = 1.0
    out["days_to_promise"] = (out["promised_date"] - as_of).dt.days
    q50 = min(model.quantiles, key=lambda v: abs(v - 0.5))
    q90 = min(model.quantiles, key=lambda v: abs(v - 0.9))
    out["expected_date"] = pred[f"date_q{round(q50 * 100)}"].to_numpy()
    out["safe_date"] = pred[f"date_q{round(q90 * 100)}"].to_numpy()
    # A shipment cannot arrive in the past.
    for c in ("expected_date", "safe_date"):
        out[c] = out[c].where(out[c] >= as_of, as_of)
    out["tier"] = np.select([overdue, out["late_probability"] >= high, out["late_probability"] >= medium],
                            ["Already late", "High", "Medium"], "Low")
    out["value_at_risk"] = out["value_usd"].fillna(0) * out["late_probability"]
    out = out.sort_values(["late_probability", "value_at_risk"], ascending=False)
    head = out.index[:explain_top]
    out["main_reasons"] = ""
    if len(head):
        out.loc[head, "main_reasons"] = top_reasons(explain_features(model, X.loc[head]))
    return out.reset_index(drop=True)


def _typical_promise(history: pd.DataFrame, carrier: str, mode: str, service: str, distance: float) -> int:
    """Transit days this carrier usually promises for a shipment like this, from its history."""
    h = history[(history["carrier"] == carrier) & (history["mode"] == mode)]
    same = h[h["service"] == service]
    h = same if len(same) >= 30 else h
    if h.empty:
        raise ValueError(f"{carrier} has no {mode} history to estimate a promise from")
    near = h.iloc[(h["distance_miles"] - distance).abs().argsort()[:60]]
    return int(max(1, round(float(near["promised_days"].median()))))


def plan_shipment(model: DelayModel, ship_date, origin_state: str, dest_state: str, distance_miles: float, mode: str,
                  service: str = "standard", weight_lbs: float = 1000.0, weather_risk: float | None = None,
                  carriers: list[str] | None = None) -> pd.DataFrame:
    """Compare the carriers that could move a planned shipment.

    For each carrier with history in ``mode``: the transit time it usually
    promises for this distance, the chance it misses that promise, and the
    predicted delivery date at each confidence level. Carriers promise
    different dates, so compare the dates, not only the late probability.
    Sorted by the 90%-confidence date, then by late probability.
    """
    history = model.history_
    options = sorted(history.loc[history["mode"] == mode, "carrier"].unique()) if carriers is None else list(carriers)
    if not options:
        raise ValueError(f"No carrier in the history moves '{mode}' shipments")
    ship_date = pd.Timestamp(ship_date).normalize()
    rows = []
    for i, carrier in enumerate(options):
        days = _typical_promise(history, carrier, mode, service, distance_miles)
        rows.append({"shipment_id": f"PLAN-{i}", "ship_date": ship_date, "promised_date": ship_date + pd.Timedelta(days=days),
                     "carrier": carrier, "mode": mode, "service": service, "origin_state": origin_state, "dest_state": dest_state,
                     "distance_miles": distance_miles, "weight_lbs": weight_lbs,
                     "weather_risk": np.nan if weather_risk is None else weather_risk})
    planned = validate_shipments(pd.DataFrame(rows))
    pred = model.predict(planned)
    out = planned[["carrier", "promised_days", "promised_date"]].copy()
    out["late_probability"] = pred["late_probability"].to_numpy()
    for q in model.quantiles:
        out[f"date_{round(q * 100)}pct"] = pred[f"date_q{round(q * 100)}"].to_numpy()
    q90 = f"date_{round(min(model.quantiles, key=lambda v: abs(v - 0.9)) * 100)}pct"
    return out.sort_values([q90, "late_probability"]).reset_index(drop=True)


def lane_distance(origin: str, destination: str) -> tuple[str, str, float]:
    """State codes and approximate road miles between two sample metros."""
    try:
        (so, la1, lo1), (sd, la2, lo2) = METROS[origin], METROS[destination]
    except KeyError as exc:
        raise ValueError(f"{exc.args[0]} is not one of the sample metros") from None
    return so, sd, float(round(great_circle_miles(la1, lo1, la2, lo2) * 1.18))


def carrier_scorecard(shipments: pd.DataFrame, folds: int = 5, since=None, seed: int = 0) -> pd.DataFrame:
    """Carrier performance adjusted for how hard their shipments were.

    A raw late rate punishes a carrier that takes the difficult lanes. Here
    a second model, which is not told who the carrier is, estimates how
    likely each shipment was to be late given its lane, mode, promise,
    season and weather. Each carrier's ``observed`` late rate is then set
    against that ``expected`` rate for its own mix of shipments.

    ``difference``  observed minus expected, in share of shipments; negative
                    means better than the difficulty of its freight implies
    ``z``           the difference in standard errors
    ``verdict``     ``better than expected`` / ``worse than expected`` when
                    the difference is beyond two standard errors

    Expected rates come from cross-fitting: each shipment is predicted by a
    model trained on the other folds. ``since`` limits the scorecard to
    shipments tendered on or after that date (the model still learns from all).
    """
    if "delivered" not in shipments.columns:
        shipments = validate_shipments(shipments)
    delivered, _ = split_delivered(shipments)
    features = [f for f in FEATURES if f not in CARRIER_FEATURES]
    prior = float(delivered["late"].mean())
    X = build_features(delivered, prior=prior)[features]
    y = delivered["late"].astype(int).to_numpy()
    expected = np.zeros(len(delivered))
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(X):
        clf = _boosted(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0,
                                                      random_state=seed), features).fit(X.iloc[train], y[train])
        expected[test] = clf.predict_proba(X.iloc[test])[:, 1]
    frame = delivered.assign(expected=expected)
    if since is not None:
        frame = frame[frame["ship_date"] >= pd.Timestamp(since)]
    g = frame.groupby("carrier")
    out = pd.DataFrame({"shipments": g.size(), "observed": g["late"].mean(), "expected": g["expected"].mean(),
                        "variance": g["expected"].apply(lambda p: float((p * (1 - p)).sum()))})
    out["difference"] = out["observed"] - out["expected"]
    out["z"] = out["difference"] * out["shipments"] / np.sqrt(out["variance"].clip(lower=1e-9))
    out["verdict"] = np.select([out["z"] <= -2, out["z"] >= 2], ["better than expected", "worse than expected"], "as expected")
    out["raw_rank"] = out["observed"].rank(method="min").astype(int)
    out["adjusted_rank"] = out["difference"].rank(method="min").astype(int)
    return out.drop(columns="variance").sort_values("difference").reset_index()


def lane_summary(shipments: pd.DataFrame, min_shipments: int = 30) -> pd.DataFrame:
    """Late rate and average delay by origin and destination region, for lanes with enough shipments."""
    if "delivered" not in shipments.columns:
        shipments = validate_shipments(shipments)
    delivered, _ = split_delivered(shipments)
    g = delivered.groupby(["origin_region", "dest_region"])
    out = pd.DataFrame({"shipments": g.size(), "late_rate": g["late"].mean(), "average_delay_days": g["delay_days"].mean(),
                        "p90_delay_days": g["delay_days"].quantile(0.9)}).reset_index()
    return out[out["shipments"] >= min_shipments].sort_values("late_rate", ascending=False).reset_index(drop=True)


__all__ = ["TIERS", "risk_list", "plan_shipment", "lane_distance", "carrier_scorecard", "lane_summary"]
