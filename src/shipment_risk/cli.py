"""Command line.

    shipment-risk sample --out shipments.csv
    shipment-risk train --input shipments.csv --model model.joblib --report model_report.html
    shipment-risk score --input shipments.csv --model model.joblib --out risk_list.csv --report risk_report.html
    shipment-risk scorecard --input shipments.csv
    shipment-risk plan --model model.joblib --origin Chicago --destination Dallas --mode truckload --date 2026-07-06
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__, evaluate
from .data import SchemaError, generate_sample_shipments, load_shipments, split_delivered
from .model import DelayModel
from .operations import carrier_scorecard, lane_distance, plan_shipment, risk_list
from .report import build_model_report, build_risk_report, risk_headline, write_html


def _show(df: pd.DataFrame, formatters: dict | None = None) -> None:
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(df.to_string(index=False, formatters=formatters or {}))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shipment-risk", description="Shipment delay and delivery-risk prediction.")
    p.add_argument("--version", action="version", version=f"shipment-risk {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="write simulated US shipments to a CSV file")
    s.add_argument("--out", default="shipments.csv")
    s.add_argument("--seed", type=int, default=11)

    t = sub.add_parser("train", help="fit the model and report how it did on the most recent shipments")
    t.add_argument("--input", required=True, help="shipments CSV")
    t.add_argument("--model", default="model.joblib", help="where to save the fitted model")
    t.add_argument("--report", help="write the model report to this HTML file")

    sc = sub.add_parser("score", help="score shipments still in transit")
    sc.add_argument("--input", required=True, help="shipments CSV; rows with a blank delivery_date are scored")
    sc.add_argument("--model", required=True)
    sc.add_argument("--out", help="write the risk list to this CSV file")
    sc.add_argument("--report", help="write the daily risk report to this HTML file")
    sc.add_argument("--as-of", help="today's date, YYYY-MM-DD (default: the last ship date in the file)")

    c = sub.add_parser("scorecard", help="carrier performance adjusted for the difficulty of their shipments")
    c.add_argument("--input", required=True)
    c.add_argument("--since", help="only shipments tendered on or after this date")
    c.add_argument("--out")

    pl = sub.add_parser("plan", help="compare carriers for a planned shipment between two sample metros")
    pl.add_argument("--model", required=True)
    pl.add_argument("--origin", required=True)
    pl.add_argument("--destination", required=True)
    pl.add_argument("--mode", required=True)
    pl.add_argument("--date", required=True)
    pl.add_argument("--service", default="standard")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pc = "{:.1%}".format
    day = lambda d: f"{pd.Timestamp(d):%Y-%m-%d}"  # noqa: E731
    try:
        if args.command == "sample":
            df = generate_sample_shipments(seed=args.seed)
            df.drop(columns=["origin_region", "dest_region", "promised_days", "delivered", "delay_days", "late"]).to_csv(args.out, index=False)
            print(f"Wrote {len(df):,} simulated shipments to {args.out} ({int((~df['delivered']).sum())} still in transit). Not real data.")
        elif args.command == "train":
            model = DelayModel().fit(load_shipments(args.input))
            s, acc = evaluate.summary(model), evaluate.eta_accuracy(model)
            print(f"Tested on {s['shipments']:,} shipments tendered after training ended (late rate {s['late_rate']:.1%}).")
            _show(evaluate.compare_models(model), {"roc_auc": "{:.3f}".format, "average_precision": "{:.3f}".format,
                                                   "brier_score": "{:.4f}".format, "log_loss": "{:.4f}".format})
            print(f"\nDelivery date error: {acc['mae_model']:.2f} days (model) against {acc['mae_promise']:.2f} days (carrier's promise).")
            _show(evaluate.eta_table(model), {"confidence": "{:.0%}".format, "coverage": pc, "coverage_day_before": pc,
                                              "average_days_after_promise": "{:+.2f}".format})
            print(f"\nSaved model to {model.save(args.model)}")
            if args.report:
                print(f"Wrote {write_html(build_model_report(model), args.report)}")
        elif args.command == "score":
            shipments = load_shipments(args.input)
            _, open_ = split_delivered(shipments)
            as_of = pd.Timestamp(args.as_of) if args.as_of else shipments["ship_date"].max()
            risks = risk_list(DelayModel.load(args.model), open_, as_of)
            print(risk_headline(risks))
            if len(risks):
                print()
                _show(risks.head(15)[["shipment_id", "tier", "carrier", "promised_date", "late_probability", "safe_date", "main_reasons"]],
                      {"late_probability": pc, "promised_date": day, "safe_date": day})
            if args.out:
                risks.to_csv(args.out, index=False)
                print(f"\nWrote {args.out}")
            if args.report:
                print(f"Wrote {write_html(build_risk_report(risks, as_of), args.report)}")
        elif args.command == "scorecard":
            card = carrier_scorecard(load_shipments(args.input), since=args.since)
            _show(card, {"observed": pc, "expected": pc, "difference": "{:+.1%}".format, "z": "{:+.1f}".format})
            if args.out:
                card.to_csv(args.out, index=False)
                print(f"\nWrote {args.out}")
        elif args.command == "plan":
            o_state, d_state, miles = lane_distance(args.origin, args.destination)
            table = plan_shipment(DelayModel.load(args.model), args.date, o_state, d_state, miles, args.mode, args.service)
            print(f"{args.origin}, {o_state} to {args.destination}, {d_state}: about {miles:,.0f} road miles, {args.mode}, {args.service}.")
            _show(table, {"late_probability": pc, **{c: day for c in table.columns if c.startswith("date_") or c == "promised_date"}})
    except (SchemaError, ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
