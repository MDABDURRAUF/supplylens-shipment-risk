"""Shipment records: the expected columns, validation, and simulated US sample data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .us import METROS, STATE_REGION, great_circle_miles, holidays_between, is_peak_season, region_of

REQUIRED_COLUMNS = ["shipment_id", "ship_date", "promised_date", "carrier", "mode", "origin_state", "dest_state", "distance_miles"]
# Optional columns. ``delivery_date`` is blank while a shipment is in transit.
OPTIONAL_COLUMNS = ["delivery_date", "origin", "destination", "service", "weight_lbs", "value_usd", "weather_risk"]


class SchemaError(ValueError):
    """Raised when a shipments table does not match the expected layout."""


def validate_shipments(df: pd.DataFrame) -> pd.DataFrame:
    """Check a shipments table and return a cleaned copy.

    Dates are parsed, state codes are checked against the 50 states and DC,
    and optional columns are added (empty) when absent. Problems raise a
    ``SchemaError`` saying how many rows are affected; nothing is dropped
    silently. Adds:

    ``origin_region``, ``dest_region``  Census regions
    ``promised_days``                   promised transit time in days
    ``delivered``                       whether a delivery date is recorded
    ``delay_days``                      delivery minus promised date, in days
    ``late``                            delivered after the promised date
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {', '.join(missing)}")
    out = df.copy()
    for c in OPTIONAL_COLUMNS:
        if c not in out.columns:
            out[c] = "standard" if c == "service" else pd.NaT if c == "delivery_date" else np.nan
    for c in ("ship_date", "promised_date", "delivery_date"):
        out[c] = pd.to_datetime(out[c], errors="coerce").dt.normalize()
    for c in ("distance_miles", "weight_lbs", "value_usd", "weather_risk"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("carrier", "mode", "origin_state", "dest_state"):
        out[c] = out[c].astype(str).str.strip()
    out["origin_state"], out["dest_state"] = out["origin_state"].str.upper(), out["dest_state"].str.upper()
    out["service"] = out["service"].fillna("standard").astype(str).str.strip().str.lower()

    problems = []
    if out["shipment_id"].duplicated().any():
        problems.append(f"{int(out['shipment_id'].duplicated().sum())} duplicate shipment_id value(s)")
    for c in ("ship_date", "promised_date", "distance_miles"):
        if out[c].isna().any():
            problems.append(f"{int(out[c].isna().sum())} row(s) with missing or unreadable '{c}'")
    bad_states = ~out["origin_state"].isin(STATE_REGION) | ~out["dest_state"].isin(STATE_REGION)
    if bad_states.any():
        problems.append(f"{int(bad_states.sum())} row(s) with a state code that is not a US state")
    if (out["promised_date"] < out["ship_date"]).any():
        problems.append("promised_date is earlier than ship_date in some rows")
    if (out["delivery_date"] < out["ship_date"]).any():
        problems.append("delivery_date is earlier than ship_date in some rows")
    if (out["distance_miles"] <= 0).any():
        problems.append("distance_miles must be greater than zero")
    if problems:
        raise SchemaError("Invalid shipments data: " + "; ".join(problems))

    out["origin_region"] = out["origin_state"].map(STATE_REGION)
    out["dest_region"] = out["dest_state"].map(STATE_REGION)
    out["promised_days"] = (out["promised_date"] - out["ship_date"]).dt.days
    out["delivered"] = out["delivery_date"].notna()
    out["delay_days"] = (out["delivery_date"] - out["promised_date"]).dt.days.astype(float)
    out["late"] = (out["delay_days"] > 0).where(out["delivered"]).astype(float)
    return out.sort_values(["ship_date", "shipment_id"]).reset_index(drop=True)


def load_shipments(path: str | Path) -> pd.DataFrame:
    """Read a shipments CSV file and validate it."""
    return validate_shipments(pd.read_csv(path))


def split_delivered(shipments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into delivered shipments (outcome known) and shipments still in transit."""
    return shipments[shipments["delivered"]].reset_index(drop=True), shipments[~shipments["delivered"]].reset_index(drop=True)


# --------------------------------------------------------------- sample data
_MODES = {
    # mode: (miles per day, handling days, transit variability, weather sensitivity, pounds: log-mean, log-sd, $ per lb)
    "truckload": (520, 0.5, 0.12, 1.0, 9.9, 0.35, 2.2),
    "LTL": (380, 1.2, 0.18, 1.0, 7.2, 0.7, 3.0),
    "intermodal rail": (330, 2.0, 0.28, 0.6, 10.1, 0.3, 1.4),
    "parcel": (480, 0.8, 0.12, 0.8, 3.0, 0.9, 14.0),
    "air": (1800, 1.0, 0.10, 1.4, 5.5, 0.9, 30.0),
}
_CARRIERS = {
    # carrier: (mode, share within mode, promise factor, average extra days)
    "Summit Freight": ("truckload", 0.55, 1.10, -0.15),
    "Prairie Lines": ("truckload", 0.45, 1.00, 0.25),
    "BlueRoute LTL": ("LTL", 0.50, 1.12, -0.05),
    "Coastal LTL": ("LTL", 0.50, 1.02, 0.30),
    "IronRail Intermodal": ("intermodal rail", 1.00, 1.10, 0.35),
    "SwiftParcel": ("parcel", 0.60, 1.05, -0.10),
    "NorthStar Parcel": ("parcel", 0.40, 1.15, 0.05),
    "AeroCargo": ("air", 1.00, 1.10, -0.20),
}
_WEATHER_BASE = {  # typical weather severity (0-1) by Census region and month
    "Northeast": [.34, .32, .22, .10, .06, .06, .08, .10, .10, .08, .14, .28],
    "Midwest": [.38, .34, .22, .12, .10, .10, .08, .08, .06, .08, .16, .32],
    "South": [.10, .10, .10, .10, .10, .14, .16, .24, .28, .20, .10, .08],
    "West": [.16, .14, .10, .06, .06, .10, .18, .22, .18, .10, .10, .16],
}


def _weather_series(dates: pd.DatetimeIndex, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A daily weather-severity index (0-1) per region: seasonal level, persistence, and storms."""
    out = {}
    for region, base in _WEATHER_BASE.items():
        level = np.array([base[m - 1] for m in dates.month])
        noise = np.zeros(len(dates))
        storm = np.zeros(len(dates))
        for i in range(1, len(dates)):
            noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 0.05)
        i = 0
        while i < len(dates):
            if rng.random() < 0.012 + 0.10 * level[i]:
                length = int(rng.integers(2, 6))
                storm[i:i + length] = rng.uniform(0.3, 0.6)
                i += length
            i += 1
        out[region] = np.clip(level + noise + storm, 0, 1)
    return out


def carrier_load_ratio(carrier: pd.Series, ship_date: pd.Series, recent_days: int = 7, normal_days: int = 91) -> np.ndarray:
    """How busy each carrier is when a shipment is tendered.

    Shipments the carrier took in the ``recent_days`` before the ship date,
    divided by its usual number for a period that long over the previous
    ``normal_days``. 1.0 is a normal week; 1.4 is 40% busier. Uses only
    shipments tendered before the ship date. Returns 1.0 where there is too
    little history to judge.
    """
    days = pd.to_datetime(ship_date).to_numpy().astype("datetime64[D]").astype(np.int64)
    ratio = np.ones(len(days))
    for _, idx in pd.Series(np.arange(len(days))).groupby(carrier.to_numpy()).indices.items():
        d = np.sort(days[idx])
        recent = np.searchsorted(d, days[idx], side="left") - np.searchsorted(d, days[idx] - recent_days, side="left")
        normal = np.searchsorted(d, days[idx], side="left") - np.searchsorted(d, days[idx] - normal_days, side="left")
        # Early in a carrier's history the comparison window is shorter than normal_days.
        span = np.clip(days[idx] - d[0], 1, normal_days)
        usual = normal * recent_days / span
        ratio[idx] = np.where((normal >= 20) & (span >= 4 * recent_days), recent / np.maximum(usual, 1e-9), 1.0)
    return ratio


def generate_sample_shipments(days: int = 912, per_day: float = 34, start: str = "2024-01-01", seed: int = 11) -> pd.DataFrame:
    """Simulated shipments between 20 US freight hubs, for demos and tests.

    The data is made up. Hidden rules decide lateness: each carrier's
    reliability and how tight its promises are, weather by region and season,
    federal holidays in transit, the year-end peak, how busy the carrier is,
    weak spots (a truckload carrier that struggles in the Northeast, an LTL
    carrier that struggles in the West), and one carrier whose service
    deteriorates in 2026. The model never sees these rules, only the records.

    Shipments not yet delivered on the last day have a blank ``delivery_date``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    weekday = np.array([1.0, 1.05, 1.05, 1.0, 0.95, 0.35, 0.15])[dates.dayofweek]
    volume = per_day * weekday * np.where(is_peak_season(pd.Series(dates)), 1.35, 1.0) * (1 + 0.15 * np.arange(days) / days)
    ship_date = dates.repeat(rng.poisson(volume))
    n = len(ship_date)
    day_index = (ship_date - dates[0]).days.to_numpy()

    names = list(METROS)
    weights = np.array([3, 1.5, 1.5, 1, 1.5, 3, 2.5, 1.5, 1.5, 3.5, 1.5, 1.5, 2, 1.2, 3, 1.5, 1.5, 1.8, 2.5, 1.5])
    weights = weights / weights.sum()
    o = rng.choice(len(names), n, p=weights)
    d = rng.choice(len(names), n, p=weights)
    same = o == d
    d[same] = (o[same] + rng.integers(1, len(names), same.sum())) % len(names)
    state = np.array([METROS[m][0] for m in names])
    lat, lon = np.array([METROS[m][1] for m in names]), np.array([METROS[m][2] for m in names])
    distance = np.round(great_circle_miles(lat[o], lon[o], lat[d], lon[d]) * 1.18)  # road miles exceed straight-line miles
    o_region = np.array([region_of(s) for s in state])[o]
    d_region = np.array([region_of(s) for s in state])[d]

    mode_names = list(_MODES)
    band = np.digitize(distance, [400, 1200])
    mix = np.array([[.35, .35, .00, .28, .02], [.30, .30, .12, .20, .08], [.20, .20, .28, .17, .15]])[band]
    mode = np.array(mode_names)[(rng.random(n)[:, None] > mix.cumsum(axis=1)).sum(axis=1).clip(max=4)]
    carrier = np.empty(n, dtype=object)
    for m in mode_names:
        options = [(c, v[1]) for c, v in _CARRIERS.items() if v[0] == m]
        hit = mode == m
        carrier[hit] = rng.choice([c for c, _ in options], hit.sum(), p=[p for _, p in options])
    expedited = (rng.random(n) < 0.18) & (mode != "intermodal rail")

    spec = pd.DataFrame(_MODES, index=["speed", "handling", "variability", "weather_k", "w_mu", "w_sd", "usd_lb"]).T
    speed, handling = spec.loc[mode, "speed"].to_numpy(), spec.loc[mode, "handling"].to_numpy()
    expected = (distance / speed + handling) * np.where(expedited, 0.8, 1.0)
    promise_factor = np.array([_CARRIERS[c][2] for c in carrier])
    promised_days = np.maximum(1, np.ceil(expected * promise_factor + 0.3)).astype(int)
    weight = np.round(np.exp(rng.normal(spec.loc[mode, "w_mu"].to_numpy(), spec.loc[mode, "w_sd"].to_numpy())), 0).clip(1)
    value = np.round(weight * spec.loc[mode, "usd_lb"].to_numpy() * rng.lognormal(0, 0.4, n), 2)

    weather = _weather_series(dates, rng)
    mid = np.minimum(day_index + np.round(expected / 2).astype(int), days - 1)
    w_true = 0.5 * (np.array([weather[r][i] for r, i in zip(o_region, day_index)])
                    + np.array([weather[r][i] for r, i in zip(d_region, mid)]))
    weather_risk = np.clip(w_true + rng.normal(0, 0.08, n), 0, 1).round(3)  # the forecast available at ship time

    ship = pd.Series(ship_date)
    load = carrier_load_ratio(pd.Series(carrier), ship)
    month = ship.dt.month.to_numpy()
    truck = np.isin(mode, ["truckload", "LTL"])
    delay = (
        expected * (rng.lognormal(0, spec.loc[mode, "variability"].to_numpy()) - 1)
        + np.array([_CARRIERS[c][3] for c in carrier])
        + spec.loc[mode, "weather_k"].to_numpy() * 6.0 * w_true**2
        + 0.6 * holidays_between(ship, ship + pd.to_timedelta(promised_days, unit="D"))
        + is_peak_season(ship) * np.select([mode == "parcel", mode == "LTL"], [0.5, 0.3], 0.0)
        + 2.0 * np.maximum(0, load - 1.15)
        + 0.5 * ((carrier == "Prairie Lines") & (d_region == "Northeast"))
        + 0.6 * ((carrier == "Coastal LTL") & ((o_region == "West") | (d_region == "West")))
        + 0.35 * (truck & np.isin(month, [12, 1, 2]) & ((o_region == "Midwest") | (d_region == "Midwest")))
        + 0.7 * ((carrier == "Coastal LTL") & (ship >= "2026-01-01").to_numpy())
        + rng.normal(0, 0.35, n)
    )
    transit = np.maximum(1, np.round(expected + delay)).astype(int)
    delivery = ship + pd.to_timedelta(transit, unit="D")
    as_of = dates[-1]

    df = pd.DataFrame({
        "shipment_id": [f"SH-{100000 + i}" for i in range(n)],
        "ship_date": ship,
        "promised_date": ship + pd.to_timedelta(promised_days, unit="D"),
        "delivery_date": delivery.where(delivery <= as_of),
        "carrier": carrier,
        "mode": mode,
        "service": np.where(expedited, "expedited", "standard"),
        "origin": np.array(names)[o],
        "origin_state": state[o],
        "destination": np.array(names)[d],
        "dest_state": state[d],
        "distance_miles": distance,
        "weight_lbs": weight,
        "value_usd": value,
        "weather_risk": weather_risk,
    })
    return validate_shipments(df)
