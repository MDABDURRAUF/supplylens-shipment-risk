import numpy as np
import pandas as pd
import pytest

import shipment_risk as sr
from shipment_risk import evaluate as ev
from shipment_risk.cli import main
from shipment_risk.data import carrier_load_ratio
from shipment_risk.features import CARRIER_FEATURES
from shipment_risk.us import great_circle_miles, holidays_between, is_peak_season


@pytest.fixture(scope="module")
def shipments():
    return sr.generate_sample_shipments()


@pytest.fixture(scope="module")
def model(shipments):
    return sr.DelayModel().fit(shipments)


@pytest.fixture(scope="module")
def in_transit(shipments):
    return sr.split_delivered(shipments)[1]


def rows(*specs):
    """Small hand-made shipments table: (id, ship, promised, delivered, carrier, origin, destination)."""
    return sr.validate_shipments(pd.DataFrame([
        {"shipment_id": i, "ship_date": s, "promised_date": p, "delivery_date": d, "carrier": c, "mode": "truckload",
         "origin_state": o, "dest_state": t, "distance_miles": 500, "weight_lbs": 1000} for i, s, p, d, c, o, t in specs]))


# ------------------------------------------------------------------ US data
def test_us_reference_data():
    assert great_circle_miles(41.88, -87.63, 32.78, -96.80) == pytest.approx(803, abs=15)  # Chicago to Dallas
    assert great_circle_miles(40.0, -75.0, 40.0, -75.0) == 0
    start = pd.Series(pd.to_datetime(["2026-07-01", "2026-07-05", "2026-11-25", "2026-12-20"]))
    end = pd.Series(pd.to_datetime(["2026-07-06", "2026-07-08", "2026-11-27", "2027-01-02"]))
    assert holidays_between(start, end).tolist() == [1, 0, 1, 2]  # 4 July; none; Thanksgiving; Christmas and New Year
    peak = is_peak_season(pd.Series(pd.to_datetime(["2026-11-14", "2026-11-15", "2026-12-31", "2027-01-01"])))
    assert peak.tolist() == [False, True, True, False]
    assert sr.region_of("il") == "Midwest" and len({s for g in sr.CENSUS_REGIONS.values() for s in g}) == 51
    assert sr.lane_distance("Chicago", "Dallas") == ("IL", "TX", round(great_circle_miles(41.88, -87.63, 32.78, -96.80) * 1.18))
    with pytest.raises(ValueError, match="not one of the sample metros"):
        sr.lane_distance("Chicago", "Atlantis")


# --------------------------------------------------------------------- data
def test_sample_data_is_reproducible_and_sensible(shipments):
    again = sr.generate_sample_shipments()
    pd.testing.assert_frame_equal(shipments, again)
    assert 20000 < len(shipments) < 35000 and shipments["shipment_id"].is_unique
    delivered, open_ = sr.split_delivered(shipments)
    assert 0.12 < delivered["late"].mean() < 0.30
    assert 0 < len(open_) < 400 and open_["delivery_date"].isna().all() and open_["late"].isna().all()
    assert (open_["ship_date"] > shipments["ship_date"].max() - pd.Timedelta(days=45)).all()
    assert (delivered["delivery_date"] > delivered["ship_date"]).all()
    assert ((delivered["delay_days"] > 0) == (delivered["late"] == 1)).all()
    assert set(shipments["origin_region"]) == {"Northeast", "Midwest", "South", "West"}
    assert shipments["weather_risk"].between(0, 1).all() and (shipments["origin_state"] != shipments["dest_state"]).mean() > 0.9
    # One carrier's service gets worse in 2026, by construction.
    coastal = delivered[delivered["carrier"] == "Coastal LTL"]
    assert coastal[coastal["ship_date"] >= "2026-01-01"]["late"].mean() > coastal[coastal["ship_date"] < "2026-01-01"]["late"].mean() + 0.05


def test_validation_reports_problems():
    good = rows(("a", "2026-01-05", "2026-01-08", "2026-01-09", "X", "IL", "TX"))
    assert good.loc[0, "late"] == 1 and good.loc[0, "delay_days"] == 1 and good.loc[0, "promised_days"] == 3
    assert good.loc[0, "origin_region"] == "Midwest" and good.loc[0, "service"] == "standard"
    raw = good[["shipment_id", "ship_date", "promised_date", "delivery_date", "carrier", "mode", "origin_state", "dest_state", "distance_miles"]]
    with pytest.raises(sr.SchemaError, match="Missing required column"):
        sr.validate_shipments(raw.drop(columns="carrier"))
    with pytest.raises(sr.SchemaError, match="not a US state"):
        sr.validate_shipments(raw.assign(dest_state="ZZ"))
    with pytest.raises(sr.SchemaError, match="promised_date is earlier"):
        sr.validate_shipments(raw.assign(promised_date="2026-01-01"))
    with pytest.raises(sr.SchemaError, match="duplicate shipment_id"):
        sr.validate_shipments(pd.concat([raw, raw]))
    with pytest.raises(sr.SchemaError, match="unreadable 'ship_date'"):
        sr.validate_shipments(raw.assign(ship_date="soon"))
    with pytest.raises(sr.SchemaError, match="greater than zero"):
        sr.validate_shipments(raw.assign(distance_miles=0))
    blank = sr.validate_shipments(raw.assign(delivery_date=None))
    assert not blank.loc[0, "delivered"] and np.isnan(blank.loc[0, "late"])


def test_carrier_load_ratio_counts_only_earlier_shipments():
    days = pd.date_range("2026-01-01", periods=60, freq="D")
    steady = pd.Series(days.repeat(2))  # two shipments a day for 60 days
    carrier = pd.Series(["X"] * len(steady))
    ratio = carrier_load_ratio(carrier, steady)
    assert ratio[:20].tolist() == [1.0] * 20  # too little history to judge
    assert ratio[-1] == pytest.approx(1.0, abs=0.05)  # a normal week
    busy = pd.concat([steady, pd.Series([days[-1]] * 1)], ignore_index=True)
    surge = pd.concat([steady, pd.Series(pd.date_range("2026-02-24", periods=5).repeat(6))], ignore_index=True)
    after = carrier_load_ratio(pd.Series(["X"] * (len(surge) + 1)), pd.concat([surge, pd.Series([pd.Timestamp("2026-03-02")])], ignore_index=True))
    assert after[-1] > 1.5 and len(busy) == len(steady) + 1  # the last week was far busier than usual


# ----------------------------------------------------------------- features
def test_history_features_use_only_delivered_shipments():
    table = rows(
        ("a", "2026-01-01", "2026-01-03", "2026-01-05", "X", "IL", "TX"),  # late, known from 5 Jan
        ("b", "2026-01-02", "2026-01-04", "2026-01-04", "X", "IL", "TX"),  # on time, known from 4 Jan
        ("c", "2026-01-04", "2026-01-06", "2026-01-09", "X", "IL", "TX"),  # shipped 4 Jan: nothing delivered before it
        ("d", "2026-01-05", "2026-01-07", "2026-01-07", "X", "IL", "TX"),  # shipped 5 Jan: only b is known
        ("e", "2026-01-06", "2026-01-08", None, "X", "IL", "TX"),          # shipped 6 Jan: a and b known; c is not
        ("f", "2026-01-06", "2026-01-08", None, "Y", "OH", "GA"),          # another carrier and lane: no history
    )
    x = sr.build_features(table, prior=0.5, strength=2.0)
    by = dict(zip(table["shipment_id"], x["carrier_history_n"]))
    assert by == {"a": 0, "b": 0, "c": 0, "d": 1, "e": 2, "f": 0}
    rate = dict(zip(table["shipment_id"], x["carrier_late_rate"]))
    assert rate["c"] == 0.5  # nothing known: the prior
    assert rate["d"] == pytest.approx((0 + 2 * 0.5) / (1 + 2))  # one on-time shipment, pulled toward the prior
    assert rate["e"] == pytest.approx((1 + 2 * 0.5) / (2 + 2))
    assert dict(zip(table["shipment_id"], x["lane_history_n"]))["f"] == 0
    assert x.loc[table["shipment_id"] == "a", "miles_per_promised_day"].iloc[0] == 250


def test_features_do_not_change_when_later_outcomes_change(shipments):
    cut = pd.Timestamp("2025-06-01")
    sample = shipments[shipments["ship_date"].between(cut - pd.Timedelta(days=200), cut + pd.Timedelta(days=60))]
    before = sr.build_features(sample)
    altered = sample.copy()
    later = altered["delivery_date"] >= cut
    altered.loc[later, "late"] = 1 - altered.loc[later, "late"]  # flip every outcome learned on or after the cut
    after = sr.build_features(altered)
    early = sample["ship_date"] <= cut
    pd.testing.assert_frame_equal(before[early], after[early])
    assert not before[~early]["carrier_late_rate"].equals(after[~early]["carrier_late_rate"])


def test_history_argument_supplies_the_past(shipments):
    delivered, open_ = sr.split_delivered(shipments)
    alone = sr.build_features(open_)
    with_past = sr.build_features(open_, history=delivered)
    assert alone["carrier_history_n"].max() == 0 and with_past["carrier_history_n"].min() > 20
    assert list(with_past.index) == list(open_.index) and list(with_past.columns) == sr.FEATURES


# -------------------------------------------------------------------- model
def test_split_is_by_time_with_no_outcome_used_early(model):
    s, hist = model.split_, model.history_
    train = hist[(hist["ship_date"] < s["train_until"]) & (hist["delivery_date"] < s["train_until"])]
    assert s["train_rows"] == len(train) and train["delivery_date"].max() < s["train_until"]
    assert model.holdout_["ship_date"].min() >= s["test_from"] and s["test_rows"] == len(model.holdout_) > 1000
    assert s["train_rows"] + s["calibration_rows"] + s["test_rows"] <= len(hist)
    assert model.prior_ == pytest.approx(train["late"].mean())


def test_model_beats_simpler_alternatives(model):
    table = ev.compare_models(model).set_index("model")
    boosted, logistic = table.loc["Gradient boosting, calibrated"], table.loc["Logistic regression"]
    history, constant = table.loc["Carrier and lane track record"], table.loc["Overall late rate (no model)"]
    assert boosted["roc_auc"] > logistic["roc_auc"] > history["roc_auc"] > 0.7
    assert boosted["brier_score"] < logistic["brier_score"] < history["brier_score"] < constant["brier_score"]
    s = ev.summary(model)
    assert s["roc_auc"] > 0.82 and 0.15 < s["skill"] < 1 and s["shipments"] == len(model.holdout_)


def test_predictions_are_well_formed_and_survive_saving(tmp_path, model, in_transit):
    pred = model.predict(in_transit)
    assert pred["late_probability"].between(0, 1).all() and len(pred) == len(in_transit)
    delays = pred[[f"delay_q{round(q * 100)}" for q in model.quantiles]].to_numpy()
    assert (np.diff(delays, axis=1) >= -1e-9).all()  # a higher confidence never gives an earlier date
    dates = pred[[f"date_q{round(q * 100)}" for q in model.quantiles]]
    assert (dates["date_q50"] <= dates["date_q95"]).all()
    np.testing.assert_allclose(model.predict_proba(in_transit), pred["late_probability"])
    loaded = sr.DelayModel.load(model.save(tmp_path / "m.joblib"))
    pd.testing.assert_frame_equal(loaded.predict(in_transit), pred)
    with pytest.raises(RuntimeError, match="fit"):
        sr.DelayModel().predict(in_transit)
    with pytest.raises(ValueError, match="At least 500"):
        sr.DelayModel().fit(in_transit)


def test_fit_options(shipments):
    recent = shipments[shipments["ship_date"] >= "2025-09-01"]
    no_test = sr.DelayModel(quantiles=(), train_fraction=0.8, calib_fraction=0.2).fit(recent)
    assert no_test.holdout_ is None and no_test.split_["test_rows"] == 0
    with pytest.raises(RuntimeError, match="no held-out period"):
        ev.summary(no_test)
    blind = sr.DelayModel(quantiles=(0.9,), exclude=tuple(CARRIER_FEATURES)).fit(recent)
    assert "carrier" not in blind.features_ and len(blind.features_) == len(sr.FEATURES) - len(CARRIER_FEATURES)
    with pytest.raises(ValueError, match="sum to at most 1"):
        sr.DelayModel(train_fraction=0.9, calib_fraction=0.3).fit(recent)


# --------------------------------------------------------------- evaluation
def test_calibration_lift_and_thresholds(model):
    cal = ev.calibration_table(model, bins=10)
    assert cal["shipments"].sum() == len(model.holdout_) and cal["predicted"].is_monotonic_increasing
    assert cal["gap"].abs().max() < 0.08  # a predicted rate is close to what happened, in every band
    lift = ev.lift_table(model)
    assert lift["cumulative_capture"].iloc[-1] == pytest.approx(1.0) and lift["cumulative_capture"].is_monotonic_increasing
    assert lift["lift"].iloc[0] > 2.5 and lift["cumulative_capture"].iloc[1] > 0.5  # the top fifth holds over half the late shipments
    assert lift["late_rate"].iloc[0] > lift["late_rate"].iloc[-1] * 10

    tt = ev.threshold_table(model, late_cost=400, action_cost=60, effectiveness=0.5)
    assert tt["flagged"].is_monotonic_decreasing and tt["recall"].is_monotonic_decreasing and tt["best"].sum() >= 1
    best = tt[tt["best"]].iloc[0]
    assert abs(best["threshold"] - ev.break_even_probability(400, 60, 0.5)) <= 0.10
    row = tt.iloc[3]
    assert row["net_benefit"] == pytest.approx(row["late_caught"] * 0.5 * 400 - row["flagged"] * 60)
    costly = ev.threshold_table(model, late_cost=400, action_cost=150, effectiveness=0.5)
    assert costly[costly["best"]]["threshold"].iloc[0] > best["threshold"]  # dearer actions call for a higher bar
    assert ev.break_even_probability(0, 60, 0.5) == float("inf")


def test_delivery_date_confidence_holds(model):
    eta = ev.eta_table(model)
    assert eta["meets_target"].all() and eta["coverage"].is_monotonic_increasing
    assert (eta["coverage_day_before"] < eta["coverage"]).all()
    assert (eta["coverage"] - eta["confidence"]).max() < 0.10  # not needlessly cautious either
    acc = ev.eta_accuracy(model)
    assert acc["mae_model"] < acc["mae_promise"] and acc["within_one_day_model"] > acc["within_one_day_promise"]


def test_segments_backtest_and_drift(shipments, model):
    seg = ev.by_segment(model, "carrier").set_index("carrier")
    assert seg["shipments"].sum() == len(model.holdout_)
    # The model has not fully caught up with the carrier that got worse after training ended.
    assert seg["gap"].idxmax() == "Coastal LTL" and seg.loc["Coastal LTL", "gap"] > 0.05
    bt = ev.backtest(shipments, folds=2, test_days=60)
    assert len(bt) == 2 and (bt["roc_auc"] > 0.75).all() and (bt["brier_score"] < bt["brier_baseline"]).all()
    assert bt["cutoff"].is_monotonic_increasing and (bt["train_shipments"].diff().dropna() > 0).all()

    same = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert ev.psi(same, same) == pytest.approx(0, abs=1e-9)
    assert ev.psi(same, same + 1.0) > 0.25 and ev.psi(same, same + 0.05) < 0.1
    assert ev.psi(pd.Series(["a"] * 90 + ["b"] * 10), pd.Series(["a"] * 50 + ["b"] * 50)) > 0.25
    recent = shipments[shipments["ship_date"] >= shipments["ship_date"].max() - pd.Timedelta(days=60)]
    drift = ev.drift_report(model, recent).set_index("input")
    assert drift.loc["ship_month", "status"] == "seasonal (expected)"
    assert drift.loc["carrier_late_rate", "status"] == "large shift"  # the deteriorating carrier shows up here
    assert drift.loc["distance_miles", "status"] == "stable" and "predicted late probability" in drift.index


# -------------------------------------------------------------- explanation
def test_factor_importance_and_local_reasons(model, in_transit):
    imp = sr.factor_importance(model)
    assert imp.iloc[0]["factor"] == "carrier" and imp["share"].sum() == pytest.approx(1.0)
    assert set(imp["factor"]) == set(sr.FACTOR_GROUPS) and imp["auc_drop"].is_monotonic_decreasing
    effects = sr.explain(model, in_transit)
    assert effects.shape == (len(in_transit), len(sr.FACTOR_GROUPS)) and effects.abs().max().max() <= 1
    carrier = effects["carrier"].groupby(in_transit["carrier"].to_numpy()).mean()
    assert carrier["Coastal LTL"] > 0.05 > carrier["Summit Freight"]  # the weak carrier raises risk, the strong one lowers it
    reasons = sr.top_reasons(effects, n=2)
    coastal = reasons[(in_transit["carrier"] == "Coastal LTL").to_numpy()]
    assert coastal.str.contains("carrier").mean() > 0.8
    flat = pd.DataFrame({"carrier": [0.0], "weather": [0.01]})
    assert sr.top_reasons(flat).iloc[0] == "no single factor stands out"
    assert sr.top_reasons(pd.DataFrame({"carrier": [0.30], "weather": [0.12], "lane": [0.05]}), n=2).iloc[0] == "carrier (+30 pts), weather (+12 pts)"


# --------------------------------------------------------------- operations
def test_risk_list(model, in_transit, shipments):
    as_of = shipments["ship_date"].max()
    risks = sr.risk_list(model, in_transit, as_of)
    assert len(risks) == len(in_transit) and risks["late_probability"].is_monotonic_decreasing
    overdue = risks[risks["promised_date"] < as_of]
    assert len(overdue) > 0 and (overdue["tier"] == "Already late").all() and (overdue["late_probability"] == 1.0).all()
    rest = risks[risks["tier"] != "Already late"]
    assert (rest.loc[rest["tier"] == "High", "late_probability"] >= 0.5).all()
    assert rest.loc[rest["tier"] == "Low", "late_probability"].max() < 0.25
    assert (risks["expected_date"] >= as_of).all() and (risks["safe_date"] >= risks["expected_date"]).all()
    np.testing.assert_allclose(risks["value_at_risk"], risks["value_usd"] * risks["late_probability"])
    assert (risks["main_reasons"] != "").all()
    strict = sr.risk_list(model, in_transit, as_of, high=0.9, medium=0.8)
    assert (strict["tier"] == "High").sum() <= (risks["tier"] == "High").sum()
    assert sr.risk_list(model, in_transit.iloc[0:0]).empty
    assert "in transit" in sr.risk_headline(risks) and sr.risk_headline(risks.iloc[0:0]) == "No shipments are in transit."


def test_plan_shipment_compares_carriers_on_dates(model):
    o, d, miles = sr.lane_distance("Chicago", "Dallas")
    plan = sr.plan_shipment(model, "2026-07-06", o, d, miles, "truckload")
    assert set(plan["carrier"]) == {"Summit Freight", "Prairie Lines"} and plan["late_probability"].between(0, 1).all()
    assert plan.iloc[0]["carrier"] == "Summit Freight"  # the more reliable truckload carrier
    assert plan["date_90pct"].is_monotonic_increasing and (plan["date_50pct"] <= plan["date_95pct"]).all()
    assert (plan["promised_date"] == pd.Timestamp("2026-07-06") + pd.to_timedelta(plan["promised_days"], unit="D")).all()
    winter = sr.plan_shipment(model, "2026-01-12", o, d, miles, "truckload", weather_risk=0.8)
    calm = sr.plan_shipment(model, "2026-01-12", o, d, miles, "truckload", weather_risk=0.05)
    assert winter["late_probability"].mean() > calm["late_probability"].mean()
    only = sr.plan_shipment(model, "2026-07-06", o, d, miles, "truckload", carriers=["Prairie Lines"])
    assert list(only["carrier"]) == ["Prairie Lines"]
    with pytest.raises(ValueError, match="No carrier"):
        sr.plan_shipment(model, "2026-07-06", o, d, miles, "zeppelin")


def test_carrier_scorecard_adjusts_for_difficulty(shipments):
    card = sr.carrier_scorecard(shipments).set_index("carrier")
    assert card["shipments"].sum() == int(shipments["delivered"].sum()) and card["expected"].between(0, 1).all()
    np.testing.assert_allclose(card["difference"], card["observed"] - card["expected"])
    # Overall, expected and observed agree: the adjustment moves credit between carriers, it does not invent any.
    assert np.average(card["difference"], weights=card["shipments"]) == pytest.approx(0, abs=0.01)
    assert card.loc["Coastal LTL", "verdict"] == "worse than expected" and card.loc["Summit Freight", "verdict"] == "better than expected"
    assert card["difference"].idxmax() == "Coastal LTL"
    # BlueRoute's raw late rate is mid-table, but its freight is hard; adjusted, it ranks near the top.
    assert card.loc["BlueRoute LTL", "adjusted_rank"] < card.loc["BlueRoute LTL", "raw_rank"]
    recent = sr.carrier_scorecard(shipments, since="2026-01-01").set_index("carrier")
    assert recent["shipments"].sum() < card["shipments"].sum()
    assert recent.loc["Coastal LTL", "difference"] > card.loc["Coastal LTL", "difference"]  # worse lately
    lanes = sr.lane_summary(shipments)
    assert lanes["late_rate"].is_monotonic_decreasing and (lanes["shipments"] >= 30).all() and len(lanes) == 16


# ------------------------------------------------------------ reports and CLI
def test_reports(model, in_transit, shipments):
    risks = sr.risk_list(model, in_transit, shipments["ship_date"].max())
    html = sr.build_risk_report(risks, shipments["ship_date"].max())
    for needle in ("Act on these first", "By carrier", risks.iloc[0]["shipment_id"], "Already late", "not a certainty"):
        assert needle in html
    card = sr.build_model_report(model)
    s = ev.summary(model)
    for needle in ("Against simpler alternatives", "Calibration", "Delivery-date confidence", f"{s['roc_auc']:.3f}", "refit regularly"):
        assert needle in card


def test_cli_end_to_end(tmp_path, capsys):
    csv, mdl = tmp_path / "s.csv", tmp_path / "m.joblib"
    assert main(["sample", "--out", str(csv)]) == 0
    assert main(["train", "--input", str(csv), "--model", str(mdl), "--report", str(tmp_path / "model.html")]) == 0
    out = capsys.readouterr().out
    assert "Gradient boosting, calibrated" in out and "Delivery date error" in out and (tmp_path / "model.html").stat().st_size > 4000
    assert main(["score", "--input", str(csv), "--model", str(mdl), "--out", str(tmp_path / "risk.csv"),
                 "--report", str(tmp_path / "risk.html")]) == 0
    assert "in transit" in capsys.readouterr().out and len(pd.read_csv(tmp_path / "risk.csv")) > 50
    assert main(["plan", "--model", str(mdl), "--origin", "Chicago", "--destination", "Dallas", "--mode", "truckload", "--date", "2026-07-06"]) == 0
    assert "Summit Freight" in capsys.readouterr().out
    assert main(["scorecard", "--input", str(csv), "--since", "2026-01-01"]) == 0
    assert "worse than expected" in capsys.readouterr().out
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    assert main(["train", "--input", str(bad)]) == 1 and "Missing required column" in capsys.readouterr().err
    assert main(["plan", "--model", str(mdl), "--origin", "Chicago", "--destination", "Nowhere", "--mode", "truckload", "--date", "2026-07-06"]) == 1
