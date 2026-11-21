"""SupplyLens Shipment Risk: shipment delay and delivery-risk prediction."""

__version__ = "0.1.0"

from . import evaluate  # noqa: E402
from .data import SchemaError, generate_sample_shipments, load_shipments, split_delivered, validate_shipments  # noqa: E402
from .explain import explain, explain_features, factor_importance, top_reasons  # noqa: E402
from .features import FACTOR_GROUPS, FEATURES, build_features  # noqa: E402
from .model import DelayModel  # noqa: E402
from .operations import carrier_scorecard, lane_distance, lane_summary, plan_shipment, risk_list  # noqa: E402
from .report import build_model_report, build_risk_report, risk_headline, write_html  # noqa: E402
from .us import CENSUS_REGIONS, METROS, federal_holidays, region_of  # noqa: E402

__all__ = [
    "CENSUS_REGIONS", "DelayModel", "FACTOR_GROUPS", "FEATURES", "METROS", "SchemaError", "build_features",
    "build_model_report", "build_risk_report", "carrier_scorecard", "evaluate", "explain", "explain_features",
    "factor_importance", "federal_holidays", "generate_sample_shipments", "lane_distance", "lane_summary",
    "load_shipments", "plan_shipment", "region_of", "risk_headline", "risk_list", "split_delivered", "top_reasons",
    "validate_shipments", "write_html",
]
