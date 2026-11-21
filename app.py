"""SupplyLens Shipment Risk dashboard. Run with:  streamlit run app.py"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import shipment_risk as sr
from shipment_risk import evaluate as ev
from shipment_risk.report import money, pct

st.set_page_config(page_title="Shipment Risk", layout="wide")

RED, DARK, AMBER, GREEN, BLUE, GREY = "#d1495b", "#7a1f2b", "#e9a23b", "#2a9d8f", "#3a6ea5", "#9aa5b1"
TIER_COLORS = {"Already late": DARK, "High": RED, "Medium": AMBER, "Low": GREY}
PCT = st.column_config.NumberColumn(format="percent")
USD = st.column_config.NumberColumn(format="dollar")
DAY = st.column_config.DateColumn(format="DD MMM YYYY")


def show(fig: go.Figure, height: int = 340, **kwargs):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return st.plotly_chart(fig, width="stretch", **kwargs)


@st.cache_data(show_spinner="Generating simulated shipments...")
def sample_data() -> pd.DataFrame:
    return sr.generate_sample_shipments()


@st.cache_resource(max_entries=3, show_spinner="Fitting the model on past shipments...")
def fit(key: tuple, _shipments: pd.DataFrame) -> sr.DelayModel:
    return sr.DelayModel().fit(_shipments)


@st.cache_data(max_entries=3, show_spinner="Scoring carriers against the difficulty of their freight...")
def scorecard(key: tuple, _shipments: pd.DataFrame, since) -> pd.DataFrame:
    return sr.carrier_scorecard(_shipments, since=since)


@st.cache_data(max_entries=3, show_spinner="Refitting at earlier dates...")
def backtest(key: tuple, _shipments: pd.DataFrame) -> pd.DataFrame:
    return ev.backtest(_shipments, folds=4, test_days=60)


@st.cache_data(max_entries=3, show_spinner=False)
def importance(key: tuple, _model: sr.DelayModel) -> pd.DataFrame:
    return sr.factor_importance(_model)


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Shipment Risk")
st.sidebar.caption("Delay and delivery-risk prediction")
page = st.sidebar.radio("View", ["Today's risk", "Plan a shipment", "Carriers and lanes", "Model quality", "Monitoring", "Data", "How it works"])

if "shipments" not in st.session_state:
    st.session_state.update(shipments=sample_data(), source="sample")
shipments = st.session_state["shipments"]
key = (len(shipments), str(shipments["ship_date"].min()), str(shipments["ship_date"].max()), float(shipments["distance_miles"].sum()))
if st.session_state["source"] == "sample":
    st.sidebar.info("Using simulated US shipments. Load your own on the **Data** view.")
try:
    model = fit(key, shipments)
except (ValueError, sr.SchemaError) as exc:
    st.error(f"The model could not be fitted on this data: {exc}")
    st.session_state.update(shipments=sample_data(), source="sample")
    st.stop()
delivered, in_transit = sr.split_delivered(shipments)
as_of = shipments["ship_date"].max()
st.sidebar.caption(f"{len(delivered):,} delivered shipments, {len(in_transit)} in transit. Latest date: {as_of:%d %b %Y}.")

# ============================================================== TODAY'S RISK
if page == "Today's risk":
    st.title(f"Shipments in transit: {as_of:%A %d %B %Y}")
    c = st.sidebar
    high = c.slider("High-risk threshold", 0.30, 0.90, 0.50, 0.05)
    medium = c.slider("Medium-risk threshold", 0.10, high, min(0.25, high), 0.05)
    risks = sr.risk_list(model, in_transit, as_of, high=high, medium=medium)
    if risks.empty:
        st.info("No shipments are in transit in this data. Rows with a blank delivery_date are treated as in transit.")
        st.stop()
    st.info(sr.risk_headline(risks))
    counts = risks["tier"].value_counts()
    k = st.columns(5)
    k[0].metric("In transit", len(risks))
    k[1].metric("Already late", int(counts.get("Already late", 0)))
    k[2].metric("High risk", int(counts.get("High", 0)))
    k[3].metric("Medium risk", int(counts.get("Medium", 0)))
    k[4].metric("Value at risk", money(risks["value_at_risk"].sum()), help="Shipment value times late probability")

    left, right = st.columns(2)
    with left:
        by = risks.groupby(["carrier", "tier"]).size().reset_index(name="shipments")
        fig = px.bar(by, x="shipments", y="carrier", color="tier", orientation="h", color_discrete_map=TIER_COLORS,
                     category_orders={"tier": sr.operations.TIERS}, labels={"shipments": "Shipments in transit", "carrier": ""})
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        show(fig, 340)
    with right:
        fig = px.scatter(risks, x="days_to_promise", y="late_probability", size=risks["value_usd"].fillna(1).clip(lower=1), color="tier",
                         hover_name="shipment_id", hover_data={"carrier": True, "mode": True, "value_usd": ":,.0f", "main_reasons": True},
                         color_discrete_map=TIER_COLORS, category_orders={"tier": sr.operations.TIERS},
                         labels={"days_to_promise": "Days until the promised date", "late_probability": "Late probability"})
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        show(fig, 340)
    st.caption("Right: each bubble is a shipment, sized by value. Left of zero means the promised date has already passed.")

    st.subheader("Risk list")
    tiers = st.multiselect("Show", sr.operations.TIERS, default=["Already late", "High", "Medium"])
    view = risks[risks["tier"].isin(tiers)] if tiers else risks
    st.caption("Click a row to see why the shipment is at risk and when it is likely to arrive.")
    cols = ["shipment_id", "tier", "carrier", "mode", "origin_state", "dest_state", "promised_date", "late_probability",
            "expected_date", "safe_date", "value_usd", "main_reasons"]
    event = st.dataframe(view[cols], hide_index=True, on_select="rerun", selection_mode="single-row", key="risk_table", column_config={
        "late_probability": st.column_config.ProgressColumn("Late probability", format="percent", min_value=0, max_value=1),
        "promised_date": DAY, "expected_date": st.column_config.DateColumn("Expected", format="DD MMM"),
        "safe_date": st.column_config.DateColumn("90% sure by", format="DD MMM"), "value_usd": USD})
    if len(view):
        chosen = view.iloc[event.selection.rows[0] if event.selection.rows else 0]
        one = in_transit[in_transit["shipment_id"] == chosen["shipment_id"]]
        st.subheader(f"{chosen['shipment_id']}: {chosen['carrier']}, {chosen['mode']}, {chosen['origin_state']} to {chosen['dest_state']}")
        left, right = st.columns(2)
        with left:
            effects = sr.explain(model, one).iloc[0].sort_values()
            fig = go.Figure(go.Bar(x=effects.values, y=effects.index, orientation="h",
                                   marker_color=[RED if v > 0 else GREEN for v in effects.values]))
            fig.update_layout(xaxis_title="Change in late probability against a typical shipment")
            fig.update_xaxes(tickformat="+.0%")
            show(fig, 320)
            st.caption("Red factors raise this shipment's risk above a typical one; green lower it. Each factor is measured on "
                       "its own, so the bars do not add up exactly to the prediction.")
        with right:
            pred = model.predict(one).iloc[0]
            dates = pd.DataFrame({"confidence": [f"{round(q * 100)}%" for q in model.quantiles],
                                  "date": [max(pred[f"date_q{round(q * 100)}"], as_of) for q in model.quantiles]})
            fig = px.scatter(dates, x="date", y="confidence", color_discrete_sequence=[BLUE], labels={"date": "", "confidence": "Confidence it has arrived"})
            fig.update_traces(marker_size=16)
            fig.add_vline(x=chosen["promised_date"], line_dash="dash", line_color=RED)
            fig.add_annotation(x=chosen["promised_date"], y=1.08, yref="paper", text="promised", showarrow=False, font=dict(color=RED))
            show(fig, 320)
            st.markdown(f"Promised **{chosen['promised_date']:%d %b}**. Late probability **{chosen['late_probability']:.0%}**. "
                        f"Most likely delivery **{chosen['expected_date']:%d %b}**; tell the customer **{chosen['safe_date']:%d %b}** "
                        "to be right nine times in ten.")
    st.subheader("Take it with you")
    d = st.columns(2)
    d[0].download_button("Risk report (HTML)", sr.build_risk_report(risks, as_of), f"risk_report_{as_of:%Y-%m-%d}.html", "text/html",
                         width="stretch", type="primary")
    d[1].download_button("Risk list (CSV)", risks.to_csv(index=False), f"risk_list_{as_of:%Y-%m-%d}.csv", "text/csv", width="stretch")

# =========================================================== PLAN A SHIPMENT
elif page == "Plan a shipment":
    st.title("Plan a shipment: which carrier, and what date to promise")
    states = sorted(set(delivered["origin_state"]) | set(delivered["dest_state"]))
    use_metros = {"origin", "destination"} <= set(delivered.columns) and delivered["origin"].notna().any()
    c = st.columns(4)
    if use_metros:
        metros = sorted(set(delivered["origin"].dropna()) & set(sr.METROS))
        origin = c[0].selectbox("From", metros, index=metros.index("Chicago") if "Chicago" in metros else 0)
        destination = c[1].selectbox("To", [m for m in metros if m != origin])
        o_state, d_state, miles = sr.lane_distance(origin, destination)
    else:
        o_state = c[0].selectbox("From state", states)
        d_state = c[1].selectbox("To state", states, index=min(1, len(states) - 1))
        miles = float(c[1].number_input("Road miles", 50, 4000, 800))
    mode = c[2].selectbox("Mode", sorted(delivered["mode"].unique()))
    service = c[3].selectbox("Service", sorted(delivered["service"].unique()), index=sorted(delivered["service"].unique()).index("standard") if "standard" in set(delivered["service"]) else 0)
    c = st.columns(3)
    ship_date = c[0].date_input("Ship date", (as_of + pd.Timedelta(days=3)).date())
    weight = c[1].number_input("Weight (lbs)", 1, 60000, 1500)
    weather = c[2].slider("Weather forecast severity", 0.0, 1.0, 0.10, 0.05, help="0 is clear; 1 is a severe storm on the route")
    try:
        plan = sr.plan_shipment(model, ship_date, o_state, d_state, miles, mode, service, weight, weather)
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()
    best = plan.iloc[0]
    st.info(f"{o_state} to {d_state}, about {miles:,.0f} road miles. **{best['carrier']}** gets it there soonest with confidence: "
            f"90% sure by **{best['date_90pct']:%a %d %b}** ({best['late_probability']:.0%} chance of missing its own promise of "
            f"{best['promised_date']:%d %b}).")
    long = plan.melt(id_vars="carrier", value_vars=[c for c in plan.columns if c.startswith("date_")], var_name="confidence", value_name="date")
    long["confidence"] = long["confidence"].str.replace("date_", "").str.replace("pct", "%") + " sure"
    fig = px.scatter(long, x="date", y="carrier", color="confidence", labels={"date": "Delivered by", "carrier": ""},
                     color_discrete_sequence=["#a6c8e0", "#6fa3cf", BLUE, "#1f3f66"])
    fig.update_traces(marker_size=15)
    fig.add_scatter(x=plan["promised_date"], y=plan["carrier"], mode="markers", name="carrier's promise",
                    marker=dict(symbol="line-ns-open", size=22, color=RED, line=dict(width=3)))
    show(fig, 130 + 60 * len(plan))
    st.caption("Carriers promise different dates, so compare when the shipment will actually arrive, not only who keeps their promise.")
    st.dataframe(plan, hide_index=True, column_config={"late_probability": PCT, "promised_date": DAY,
                                                       **{c: DAY for c in plan.columns if c.startswith("date_")}})

# ======================================================== CARRIERS AND LANES
elif page == "Carriers and lanes":
    st.title("Carrier performance, adjusted for the difficulty of the freight")
    period = st.radio("Period", ["All history", "Last 180 days", "Last 90 days"], horizontal=True)
    since = None if period == "All history" else as_of - pd.Timedelta(days=180 if "180" in period else 90)
    card = scorecard(key, shipments, since)
    moved = card.assign(change=card["raw_rank"] - card["adjusted_rank"]).sort_values("change", ascending=False).iloc[0]
    worst, top = card.iloc[-1], card.iloc[0]
    text = (f"**{top['carrier']}** does best against the difficulty of its freight ({top['difference'] * 100:+.1f} points against "
            f"expected); **{worst['carrier']}** does worst ({worst['difference'] * 100:+.1f} points).")
    if moved["change"] >= 2:
        text += (f" **{moved['carrier']}** ranks {moved['raw_rank']} on raw late rate but {moved['adjusted_rank']} once its harder "
                 "freight is taken into account.")
    st.info(text)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_bar(x=card["carrier"], y=card["expected"], name="Expected for its freight", marker_color=GREY)
        fig.add_bar(x=card["carrier"], y=card["observed"], name="Observed", marker_color=BLUE)
        fig.update_layout(barmode="group", yaxis_title="Late rate", legend=dict(orientation="h", y=-0.3))
        fig.update_yaxes(tickformat=".0%")
        show(fig, 360)
    with right:
        colors = card["verdict"].map({"better than expected": GREEN, "worse than expected": RED, "as expected": GREY})
        fig = go.Figure(go.Bar(x=card["difference"], y=card["carrier"], orientation="h", marker_color=colors))
        fig.update_layout(xaxis_title="Observed minus expected late rate")
        fig.update_xaxes(tickformat="+.0%")
        fig.update_yaxes(autorange="reversed")
        show(fig, 360)
    st.caption("Expected is what a model that is not told the carrier predicts for each shipment's lane, mode, promise, season and "
               "weather. Green and red mark differences beyond two standard errors.")
    st.dataframe(card, hide_index=True, column_config={"observed": PCT, "expected": PCT, "difference": st.column_config.NumberColumn(format="%+.3f"),
                                                       "z": st.column_config.NumberColumn(format="%+.1f")})
    st.download_button("Download scorecard (CSV)", card.to_csv(index=False), "carrier_scorecard.csv", "text/csv")

    st.subheader("Lanes")
    lanes = sr.lane_summary(shipments)
    grid = lanes.pivot(index="origin_region", columns="dest_region", values="late_rate")
    left, right = st.columns([2, 3])
    with left:
        show(px.imshow(grid, color_continuous_scale="Reds", text_auto=".0%", aspect="auto",
                       labels={"x": "To", "y": "From", "color": "Late rate"}), 320)
    with right:
        st.dataframe(lanes, hide_index=True, column_config={"late_rate": PCT, "average_delay_days": st.column_config.NumberColumn(format="%+.2f"),
                                                            "p90_delay_days": st.column_config.NumberColumn("90% arrive within (days past promise)", format="%+.0f")})

# ============================================================= MODEL QUALITY
elif page == "Model quality":
    st.title("How far the predictions can be trusted")
    s, split = ev.summary(model), model.split_
    st.caption(f"Every figure here is measured on {s['shipments']:,} shipments tendered from {split['test_from']:%d %b %Y}, after the "
               f"model's training data ended ({split['train_rows']:,} shipments delivered before {split['train_until']:%d %b %Y}).")
    acc = ev.eta_accuracy(model)
    lift = ev.lift_table(model)
    k = st.columns(4)
    k[0].metric("ROC AUC", f"{s['roc_auc']:.3f}", help="0.5 is guessing, 1.0 is perfect ranking of late against on-time")
    k[1].metric("Error removed", pct(s["skill"]), help="Share of the error of always quoting the average late rate that the model removes")
    k[2].metric("Late shipments in riskiest fifth", pct(lift["cumulative_capture"].iloc[1]))
    k[3].metric("Delivery date error", f"{acc['mae_model']:.2f} days", f"{acc['mae_model'] - acc['mae_promise']:+.2f} vs carrier's promise", delta_color="inverse")

    left, right = st.columns(2)
    with left:
        st.subheader("Against simpler alternatives")
        comp = ev.compare_models(model)
        fig = px.bar(comp, x="brier_score", y="model", orientation="h", text_auto=".4f", color_discrete_sequence=[BLUE],
                     labels={"brier_score": "Brier score (lower is better)", "model": ""})
        fig.update_yaxes(categoryorder="total descending")
        show(fig, 300)
        st.dataframe(comp, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.4f") for c in comp.columns[1:]})
    with right:
        st.subheader("Calibration")
        cal = ev.calibration_table(model)
        top = float(max(cal["predicted"].max(), cal["observed"].max()) * 1.1)
        fig = px.scatter(cal, x="predicted", y="observed", color_discrete_sequence=[BLUE],
                         labels={"predicted": "Predicted late rate", "observed": "Observed late rate"})
        fig.update_traces(marker_size=12)
        fig.add_scatter(x=[0, top], y=[0, top], mode="lines", line=dict(dash="dash", color=GREY), showlegend=False)
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        show(fig, 300)
        st.caption("Each point is a tenth of the test shipments. On the dashed line, a predicted 30% really is 30%.")

    st.subheader("What is it worth to act on an alert?")
    c = st.columns(3)
    late_cost = c[0].number_input("Cost of a late delivery ($)", 0, 100000, 400, 50)
    action_cost = c[1].number_input("Cost of acting on one shipment ($)", 0, 10000, 60, 10)
    effect = c[2].slider("Chance the action prevents the lateness", 0.05, 1.0, 0.5, 0.05)
    tt = ev.threshold_table(model, late_cost, action_cost, effect, thresholds=np.round(np.arange(0.02, 0.96, 0.02), 2))
    best = tt[tt["best"]].iloc[0]
    left, right = st.columns(2)
    with left:
        fig = px.line(tt, x="threshold", y="net_benefit", labels={"threshold": "Act when late probability is at least", "net_benefit": "Net benefit on the test period ($)"})
        fig.update_traces(line_color=GREEN)
        fig.add_vline(x=best["threshold"], line_dash="dash", line_color=RED)
        fig.update_xaxes(tickformat=".0%")
        show(fig, 300)
    with right:
        fig = go.Figure()
        fig.add_scatter(x=tt["threshold"], y=tt["recall"], name="Late shipments caught", line_color=BLUE)
        fig.add_scatter(x=tt["threshold"], y=tt["precision"], name="Alerts that were right", line_color=GREEN)
        fig.add_scatter(x=tt["threshold"], y=tt["share_flagged"], name="Shipments flagged", line=dict(color=GREY, dash="dot"))
        fig.update_layout(legend=dict(orientation="h", y=-0.3))
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        show(fig, 300)
    if best["net_benefit"] <= 0:
        st.warning("With these costs, acting on alerts would not have paid for itself at any threshold.")
    else:
        st.info(f"Best threshold **{best['threshold']:.0%}** (break-even in theory: {ev.break_even_probability(late_cost, action_cost, effect):.0%}). "
                f"It flags {best['flagged']:.0f} shipments ({best['share_flagged']:.0%}), catches {best['recall']:.0%} of late deliveries, and "
                f"would have been worth **{money(best['net_benefit'])}** over the test period. The three inputs are your assumptions.")

    left, right = st.columns(2)
    with left:
        st.subheader("Delivery-date confidence")
        eta = ev.eta_table(model)
        st.dataframe(eta, hide_index=True, column_config={
            "confidence": PCT, "coverage": st.column_config.NumberColumn("Arrived by the date", format="percent"),
            "coverage_day_before": st.column_config.NumberColumn("Arrived a day earlier", format="percent"),
            "average_days_after_promise": st.column_config.NumberColumn("Days after promise", format="%+.2f"),
            "meets_target": st.column_config.CheckboxColumn("Meets target")})
        st.caption("'Arrived by the date' should be at least the confidence level. 'A day earlier' shows the date is not padded.")
    with right:
        st.subheader("What the model relies on")
        imp = importance(key, model)
        fig = px.bar(imp, x="share", y="factor", orientation="h", color_discrete_sequence=[BLUE], labels={"share": "Share of predictive power", "factor": ""})
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_xaxes(tickformat=".0%")
        show(fig, 300)
    st.download_button("Model report (HTML)", sr.build_model_report(model), "model_report.html", "text/html", type="primary")

# ================================================================ MONITORING
elif page == "Monitoring":
    st.title("Is the model still right?")
    st.subheader("Accuracy over time")
    bt = backtest(key, shipments)
    fig = go.Figure()
    fig.add_scatter(x=bt["cutoff"], y=bt["roc_auc"], name="ROC AUC", mode="lines+markers", line_color=BLUE)
    fig.add_scatter(x=bt["cutoff"], y=1 - bt["brier_score"] / bt["brier_baseline"], name="Error removed", mode="lines+markers", line_color=GREEN)
    fig.update_layout(xaxis_title="Model refitted on shipments delivered before", yaxis_title="", legend=dict(orientation="h", y=-0.3))
    show(fig, 300)
    st.dataframe(bt, hide_index=True, column_config={"cutoff": DAY, "late_rate": PCT, **{c: st.column_config.NumberColumn(format="%.3f")
                                                     for c in ("roc_auc", "average_precision", "brier_score", "log_loss", "brier_baseline")}})
    st.caption("Each row refits the model using only what was known at that date and scores it on the following 60 days.")

    left, right = st.columns(2)
    with left:
        st.subheader("Where predictions are off")
        segment = st.selectbox("By", ["carrier", "mode", "dest_state", "origin_state"])
        seg = ev.by_segment(model, segment)
        fig = go.Figure(go.Bar(x=seg["gap"], y=seg[segment].astype(str), orientation="h",
                               marker_color=[RED if abs(v) > 0.05 else GREY for v in seg["gap"]]))
        fig.update_layout(xaxis_title="Observed minus predicted late rate, test period")
        fig.update_xaxes(tickformat="+.0%")
        fig.update_yaxes(autorange="reversed")
        show(fig, 340)
        off = seg[seg["gap"].abs() > 0.05]
        if len(off):
            names = ", ".join(f"{r[segment]} ({r['gap'] * 100:+.0f} pts)" for _, r in off.iterrows())
            st.warning(f"Predictions are more than 5 points off for: {names}. Something changed after training; refit the model.")
        else:
            st.success("Predicted and observed late rates agree within 5 points for every group.")
    with right:
        st.subheader("Has the data moved?")
        window = st.slider("Compare the last (days) with the training data", 30, 180, 60, 10)
        recent = shipments[shipments["ship_date"] >= as_of - pd.Timedelta(days=window)]
        drift = ev.drift_report(model, recent)
        st.dataframe(drift, hide_index=True, column_config={"psi": st.column_config.NumberColumn("Shift (PSI)", format="%.3f")}, height=340)
        large = drift[drift["status"] == "large shift"]["input"].tolist()
        st.caption("Below 0.10 is stable, 0.10 to 0.25 a moderate shift, above 0.25 a large one. "
                   + (f"Large shifts: {', '.join(large)}." if large else "No large shifts."))

# ====================================================================== DATA
elif page == "Data":
    st.title("Data")
    st.caption("One row per shipment. Leave delivery_date blank for shipments still in transit; those are the ones scored.")
    up = st.file_uploader("Shipments CSV", type="csv")
    c = st.columns(3)
    if up is not None and c[0].button("Use this file", type="primary"):
        try:
            loaded = sr.validate_shipments(pd.read_csv(BytesIO(up.getvalue())))
            if int(loaded["delivered"].sum()) < 500:
                raise sr.SchemaError("at least 500 delivered shipments are needed to fit the model")
        except (sr.SchemaError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not use this file: {exc}")
        else:
            st.session_state.update(shipments=loaded, source="yours")
            st.rerun()
    if c[1].button("Reset to simulated data"):
        st.session_state.update(shipments=sample_data(), source="sample")
        st.rerun()
    raw = shipments.drop(columns=["origin_region", "dest_region", "promised_days", "delivered", "delay_days", "late"])
    c[2].download_button("Download current data (CSV)", raw.to_csv(index=False), "shipments.csv", "text/csv")
    st.subheader("Columns")
    st.dataframe(pd.DataFrame([
        ("shipment_id", "yes", "Unique identifier"), ("ship_date", "yes", "Date tendered to the carrier"),
        ("promised_date", "yes", "Delivery date promised"), ("delivery_date", "no", "Actual delivery date; blank while in transit"),
        ("carrier", "yes", "Carrier name"), ("mode", "yes", "For example truckload, LTL, parcel, air, intermodal rail"),
        ("origin_state, dest_state", "yes", "Two-letter US state codes"), ("distance_miles", "yes", "Road miles"),
        ("service", "no", "For example standard or expedited"), ("weight_lbs", "no", "Shipment weight"),
        ("value_usd", "no", "Shipment value, used for value at risk"),
        ("weather_risk", "no", "Forecast severity on the route at ship time, 0 to 1"),
        ("origin, destination", "no", "City or facility names, for display"),
    ], columns=["column", "required", "meaning"]), hide_index=True)
    st.subheader("Current data")
    k = st.columns(4)
    k[0].metric("Shipments", f"{len(shipments):,}")
    k[1].metric("Late rate", pct(delivered["late"].mean(), 1))
    k[2].metric("Carriers", delivered["carrier"].nunique())
    k[3].metric("Period", f"{shipments['ship_date'].min():%b %Y} to {as_of:%b %Y}")
    monthly = delivered.groupby(pd.Grouper(key="ship_date", freq="MS")).agg(shipments=("late", "size"), late_rate=("late", "mean")).reset_index()
    fig = px.line(monthly, x="ship_date", y="late_rate", markers=True, labels={"ship_date": "", "late_rate": "Late rate by ship month"})
    fig.update_traces(line_color=BLUE)
    fig.update_yaxes(tickformat=".0%")
    show(fig, 280)
    st.dataframe(raw.tail(200), hide_index=True, height=300)

# ============================================================== HOW IT WORKS
else:
    st.title("How it works")
    st.markdown(
        """
**Two predictions for every shipment**

1. *Will it be late?* A probability that delivery falls after the promised date.
2. *When will it arrive?* A delivery date at several confidence levels, so you can quote a date you will keep nine times in ten.

**What the model looks at**, all known when the shipment is tendered: the carrier and its recent late rate, how busy the
carrier is, the lane and its recent late rate, mode and service, how tight the promise is for the distance, day of week,
month, the year-end peak, US federal holidays in transit, the weather forecast, and shipment weight.

**No peeking.** A carrier's "recent late rate" counts only shipments that had already been *delivered* before this one
shipped. The model is trained on older shipments, calibrated on newer ones, and tested on the newest, with every split by date.

**Calibration.** Raw scores are adjusted on held-back shipments so that a predicted 30% means about three in ten are late,
and delivery dates are shifted so that the stated confidence is met.

**Explanations** are measured by experiment: swap one factor for typical values and see how the prediction moves.

**The carrier scorecard** asks a second model, which is not told the carrier, how hard each shipment was. A carrier is then
judged against the expected late rate for its own mix of freight, so taking difficult lanes is not held against it.

**Limits.** The sample data is simulated. On your data, accuracy is whatever the Model quality view shows for your most
recent period, and it changes when carriers or conditions change: watch the Monitoring view and refit. Predictions for
carriers or lanes with little history lean on averages. The model cannot foresee events unlike anything in its history.
"""
    )
