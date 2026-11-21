"""Model inputs, built so that nothing from the future leaks in.

Every feature is something known when the shipment is tendered. The history
features (a carrier's or a lane's recent late rate) count only shipments that
had already been **delivered** before the ship date, because a shipment still
on the road has no outcome yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import carrier_load_ratio
from .us import holidays_between, is_peak_season

CATEGORICAL = ["carrier", "mode", "service", "origin_region", "dest_region", "origin_state", "dest_state"]
NUMERIC = ["distance_miles", "log_weight", "promised_days", "miles_per_promised_day", "ship_weekday", "ship_month",
           "peak_season", "holidays_in_transit", "weather_risk", "carrier_late_rate", "carrier_history_n",
           "lane_late_rate", "lane_history_n", "carrier_load_ratio"]
FEATURES = CATEGORICAL + NUMERIC
# Features that identify or describe the carrier; left out when estimating how hard a shipment is in itself.
CARRIER_FEATURES = ["carrier", "carrier_late_rate", "carrier_history_n", "carrier_load_ratio"]

# Groups used to explain a prediction in plain terms.
FACTOR_GROUPS = {
    "carrier": ["carrier", "carrier_late_rate", "carrier_history_n"],
    "carrier workload": ["carrier_load_ratio"],
    "lane": ["origin_region", "dest_region", "origin_state", "dest_state", "distance_miles", "lane_late_rate", "lane_history_n"],
    "mode and service": ["mode", "service"],
    "promise tightness": ["promised_days", "miles_per_promised_day"],
    "timing": ["ship_weekday", "ship_month", "peak_season", "holidays_in_transit"],
    "weather": ["weather_risk"],
    "shipment size": ["log_weight"],
}


def _history_rate(frame: pd.DataFrame, keys: list[str], window_days: int, prior: float, strength: float):
    """Late rate over each row's group among shipments delivered in the ``window_days`` before its ship date.

    Returns ``(count, rate)`` where the rate is pulled toward ``prior`` when
    the count is small: ``(late + strength x prior) / (count + strength)``.
    """
    n_rows = len(frame)
    count, late = np.zeros(n_rows), np.zeros(n_rows)
    ship = frame["ship_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    delivered = frame["delivery_date"].notna().to_numpy()
    deliv = frame["delivery_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    is_late = np.nan_to_num(frame["late"].to_numpy(dtype=float))
    group_key = frame[keys[0]].astype(str)
    for extra in keys[1:]:
        group_key = group_key + "|" + frame[extra].astype(str)
    for _, idx in pd.Series(np.arange(n_rows)).groupby(group_key.to_numpy()).indices.items():
        known = idx[delivered[idx]]
        order = np.argsort(deliv[known], kind="stable")
        when = deliv[known][order]
        cum = np.concatenate([[0.0], np.cumsum(is_late[known][order])])
        hi = np.searchsorted(when, ship[idx], side="left")  # delivered strictly before the ship date
        lo = np.searchsorted(when, ship[idx] - window_days, side="left")
        count[idx] = hi - lo
        late[idx] = cum[hi] - cum[lo]
    return count, (late + strength * prior) / (count + strength)


def build_features(shipments: pd.DataFrame, history: pd.DataFrame | None = None, prior: float = 0.2,
                   carrier_window: int = 60, lane_window: int = 180, strength: float = 20.0) -> pd.DataFrame:
    """The feature table for ``shipments``, indexed like ``shipments``.

    ``history`` is earlier shipments (validated) that the history features may
    draw on; rows of ``shipments`` are also used as history for one another,
    subject to the delivered-before-ship-date rule. ``prior`` is the overall
    late rate that thin histories are pulled toward; pass the rate from the
    training period so that it, too, comes from the past.
    """
    target_ids = shipments["shipment_id"].to_numpy()
    if history is not None and len(history):
        extra = history[~history["shipment_id"].isin(target_ids)]
        frame = pd.concat([extra, shipments], ignore_index=True)
    else:
        frame = shipments.reset_index(drop=True)

    x = pd.DataFrame(index=frame.index)
    for c in CATEGORICAL:
        x[c] = frame[c].astype(str)
    x["distance_miles"] = frame["distance_miles"].astype(float)
    x["log_weight"] = np.log(frame["weight_lbs"].astype(float).clip(lower=1))
    x["promised_days"] = frame["promised_days"].astype(float)
    x["miles_per_promised_day"] = frame["distance_miles"] / frame["promised_days"].clip(lower=1)
    x["ship_weekday"] = frame["ship_date"].dt.dayofweek.astype(float)
    x["ship_month"] = frame["ship_date"].dt.month.astype(float)
    x["peak_season"] = is_peak_season(frame["ship_date"]).astype(float)
    x["holidays_in_transit"] = holidays_between(frame["ship_date"], frame["promised_date"]).astype(float)
    x["weather_risk"] = frame["weather_risk"].astype(float)
    x["carrier_history_n"], x["carrier_late_rate"] = _history_rate(frame, ["carrier"], carrier_window, prior, strength)
    x["lane_history_n"], x["lane_late_rate"] = _history_rate(frame, ["origin_state", "dest_state"], lane_window, prior, strength)
    x["carrier_load_ratio"] = carrier_load_ratio(frame["carrier"], frame["ship_date"])

    x["shipment_id"] = frame["shipment_id"].to_numpy()
    out = x.set_index("shipment_id").loc[target_ids, FEATURES]
    out.index = shipments.index
    return out
