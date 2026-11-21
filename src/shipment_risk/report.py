"""Printable outputs: the daily risk report and the model report."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from . import evaluate
from .explain import factor_importance
from .model import DelayModel


def money(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def pct(x, digits: int = 0) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}%}"


def _date(x) -> str:
    return "" if x is None or pd.isna(x) else f"{pd.Timestamp(x):%d %b %Y}"


_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#1c2430}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{margin:0 0 4px;font-size:26px} h2{margin:34px 0 8px;font-size:19px} p,li{line-height:1.5}
p.note{color:#56627a;font-size:14px;margin:0 0 12px}
.lead{font-size:16px;background:#fff;border:1px solid #dfe3ea;border-left:5px solid #3a6ea5;border-radius:8px;padding:14px 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.kpi{background:#fff;border:1px solid #dfe3ea;border-radius:8px;padding:14px}
.kpi b{display:block;font-size:21px} .kpi span{font-size:13px;color:#56627a}
.scroll{overflow-x:auto;background:#fff;border:1px solid #dfe3ea;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #edf0f4;vertical-align:top}
th{background:#eef1f6;font-weight:600;white-space:nowrap} td.num{text-align:right;white-space:nowrap}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff;white-space:nowrap}
.caveat{background:#fff8e6;border:1px solid #f0d58a;border-radius:8px;padding:12px 14px;font-size:14px}
@media print{body{background:#fff}}
"""
_TIER = {"Already late": "#7a1f2b", "High": "#d1495b", "Medium": "#e9a23b", "Low": "#9aa5b1"}


def _table(df: pd.DataFrame, formats: dict | None = None, empty: str = "Nothing to show.") -> str:
    if df is None or df.empty:
        return f'<p class="note">{escape(empty)}</p>'
    formats = formats or {}
    head = "".join(f"<th>{escape(str(c).replace('_', ' '))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c, v in row.items():
            if c == "tier":
                cells.append(f'<td><span class="tag" style="background:{_TIER.get(v, "#9aa5b1")}">{escape(str(v))}</span></td>')
            elif c in formats:
                cells.append(f'<td class="num">{escape(formats[c](v))}</td>')
            else:
                cells.append(f"<td>{escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _kpis(items) -> str:
    return '<div class="kpis">' + "".join(f'<div class="kpi"><b>{escape(v)}</b><span>{escape(k)}</span></div>' for k, v in items) + "</div>"


def _page(title: str, subtitle: str, body: str) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{escape(title)}</title><style>{_CSS}</style></head><body><main><h1>{escape(title)}</h1>"
            f'<p class="note">{escape(subtitle)}</p>{body}</main></body></html>')


def risk_headline(risks: pd.DataFrame) -> str:
    """One paragraph on today's shipments in transit."""
    if risks.empty:
        return "No shipments are in transit."
    counts = risks["tier"].value_counts()
    late, high, medium = (int(counts.get(k, 0)) for k in ("Already late", "High", "Medium"))
    text = (f"{len(risks)} shipments are in transit, worth {money(risks['value_usd'].sum())}. "
            f"{late} {'has' if late == 1 else 'have'} already missed the promised date, {high} more "
            f"{'is' if high == 1 else 'are'} at high risk and {medium} at medium risk. "
            f"Value at risk is {money(risks['value_at_risk'].sum())}.")
    urgent = risks[risks["tier"].isin(["Already late", "High"])]
    if len(urgent):
        worst = urgent.groupby("carrier").size().sort_values(ascending=False)
        text += f" {worst.index[0]} carries the most of these ({int(worst.iloc[0])})."
    return text


def build_risk_report(risks: pd.DataFrame, as_of, top: int = 40) -> str:
    """The daily risk report as an HTML page: headline, counts, and the shipments to act on first."""
    body = f'<p class="lead">{escape(risk_headline(risks))}</p>'
    if not risks.empty:
        counts = risks["tier"].value_counts()
        body += _kpis([("in transit", f"{len(risks)}"), ("already late", f"{int(counts.get('Already late', 0))}"),
                       ("high risk", f"{int(counts.get('High', 0))}"), ("medium risk", f"{int(counts.get('Medium', 0))}"),
                       ("value at risk", money(risks["value_at_risk"].sum()))])
        cols = ["shipment_id", "tier", "carrier", "mode", "origin_state", "dest_state", "promised_date", "late_probability",
                "expected_date", "safe_date", "value_usd", "main_reasons"]
        body += f"<h2>Act on these first</h2><p class=\"note\">The {min(top, len(risks))} riskiest shipments. "
        body += "Expected date is the median prediction; safe date is the date nine in ten such shipments arrive by.</p>"
        body += _table(risks.head(top)[cols], {"promised_date": _date, "expected_date": _date, "safe_date": _date,
                                               "late_probability": pct, "value_usd": money})
        by = risks.groupby("carrier").agg(shipments=("shipment_id", "size"), high_or_late=("tier", lambda s: int(s.isin(["Already late", "High"]).sum())),
                                          average_risk=("late_probability", "mean"), value_at_risk=("value_at_risk", "sum"))
        body += "<h2>By carrier</h2>" + _table(by.sort_values("value_at_risk", ascending=False).reset_index(),
                                               {"average_risk": pct, "value_at_risk": money})
    body += ('<h2>How to read this</h2><p class="caveat">Late probability is the model\'s estimate that a shipment is delivered '
             "after its promised date, based on past shipments like it. It is a probability, not a certainty: among shipments "
             "given 30%, about three in ten are late. Reasons are the factors that raise this shipment's risk above a typical one.</p>")
    return _page(f"Shipment risk report: {_date(as_of)}", "SupplyLens Shipment Risk", body)


def build_model_report(model: DelayModel) -> str:
    """What the model was trained on and how well it did on shipments it had not seen."""
    s, split = evaluate.summary(model), model.split_
    eta, acc = evaluate.eta_table(model), evaluate.eta_accuracy(model)
    lift = evaluate.lift_table(model)
    lead = (f"On {s['shipments']:,} shipments tendered after training ended, the model separates late from on-time shipments with "
            f"a ROC AUC of {s['roc_auc']:.3f} and removes {s['skill']:.0%} of the error of always quoting the average late rate. "
            f"The riskiest tenth of shipments contains {lift['cumulative_capture'].iloc[0]:.0%} of all late deliveries. "
            f"Its median delivery-date prediction is off by {acc['mae_model']:.2f} days on average, against {acc['mae_promise']:.2f} "
            "days for trusting the carrier's promised date.")
    body = f'<p class="lead">{escape(lead)}</p>'
    body += _kpis([("ROC AUC", f"{s['roc_auc']:.3f}"), ("Brier score", f"{s['brier_score']:.4f}"), ("baseline Brier", f"{s['brier_baseline']:.4f}"),
                   ("late rate, test period", pct(s["late_rate"], 1)), ("test shipments", f"{s['shipments']:,}")])
    body += "<h2>Data and split</h2>"
    body += (f"<p>Trained on {split['train_rows']:,} shipments delivered before {_date(split['train_until'])}; probabilities and "
             f"delivery-date confidence levels calibrated on {split['calibration_rows']:,} later shipments; tested on "
             f"{split['test_rows']:,} shipments tendered from {_date(split['test_from'])}. Splits are by time, and a shipment is "
             "used for training only if it had been delivered before the next period began.</p>")
    num = lambda v: f"{v:.3f}"  # noqa: E731
    body += "<h2>Against simpler alternatives</h2>" + _table(evaluate.compare_models(model), {
        "roc_auc": num, "average_precision": num, "brier_score": lambda v: f"{v:.4f}", "log_loss": lambda v: f"{v:.4f}"})
    body += "<h2>Calibration</h2><p class=\"note\">Predicted and observed late rate in ten equal bands of predicted risk.</p>"
    body += _table(evaluate.calibration_table(model), {"predicted": lambda v: pct(v, 1), "observed": lambda v: pct(v, 1),
                                                       "gap": lambda v: f"{v * 100:+.1f} pts"})
    body += "<h2>Where the late shipments are</h2>" + _table(
        lift[["group", "shipments", "late_rate", "lift", "cumulative_capture"]],
        {"late_rate": lambda v: pct(v, 1), "lift": lambda v: f"{v:.2f}x", "cumulative_capture": pct})
    body += ("<h2>Delivery-date confidence</h2><p class=\"note\">Coverage should be at least the stated confidence. Dates are whole "
             "days, so it is often higher; the day-before column shows what quoting one day earlier would have achieved.</p>")
    body += _table(eta, {"confidence": pct, "coverage": lambda v: pct(v, 1), "coverage_day_before": lambda v: pct(v, 1),
                         "average_days_after_promise": lambda v: f"{v:+.2f}", "meets_target": lambda v: "yes" if v else "no"})
    body += "<h2>What the model relies on</h2>" + _table(
        factor_importance(model)[["factor", "auc_drop", "share"]], {"auc_drop": lambda v: f"{v:.4f}", "share": pct})
    body += ('<h2>Limits</h2><p class="caveat">Accuracy is measured on one held-out period of the data supplied; it can change '
             "when carriers, lanes or conditions change, so refit regularly and watch the drift report. The model predicts from "
             "patterns in past shipments and knows nothing about events absent from that history. Predictions for carriers or "
             "lanes with little history lean on averages.</p>")
    return _page("Model report: shipment delay and delivery risk", "SupplyLens Shipment Risk", body)


def write_html(html: str, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
