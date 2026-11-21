# SupplyLens Shipment Risk

**Shipment delay and delivery-risk prediction for US freight: which shipments will be late, when they will
really arrive, why, and which carrier to use next time.**

For every shipment the model gives two things:

1. **A late probability**, calibrated so that a predicted 30% means about three in ten such shipments are late.
2. **A delivery date at several confidence levels**, so a planner can quote the date they will keep nine times in ten.

Around those predictions it produces a daily risk list with reasons, a carrier comparison for planned
shipments, a carrier scorecard adjusted for the difficulty of the freight, and the checks needed to know
whether the model can still be trusted.

The included data is simulated. Results on it show that the method works; they are not findings about real carriers.

## What it does

| Question | Where | Function |
|---|---|---|
| Which shipments in transit need attention today, and why? | **Today's risk** | `risk_list`, `explain` |
| When will this shipment actually arrive? | **Today's risk**, shipment detail | `DelayModel.predict` |
| Which carrier should take this load, and what date can I promise? | **Plan a shipment** | `plan_shipment` |
| Which carriers are good once the difficulty of their freight is counted? | **Carriers and lanes** | `carrier_scorecard` |
| How accurate is it, and against what? | **Model quality** | `evaluate.compare_models`, `calibration_table`, `lift_table`, `eta_table` |
| What alert threshold pays for itself? | **Model quality** | `evaluate.threshold_table` |
| Is it still accurate, and has the data changed? | **Monitoring** | `evaluate.backtest`, `by_segment`, `drift_report` |

Outputs: a printable daily risk report and model report (HTML), the risk list and scorecard as CSV, a saved
model file, and a command line for all of it.

## Results on the simulated data

27,133 shipments over 30 months between 20 US freight hubs, eight carriers, five modes. Trained on shipments
delivered before 20 October 2025, tested on 4,068 shipments tendered from 13 February 2026.

| Predictor | ROC AUC | Brier score |
|---|---|---|
| Overall late rate (no model) | | 0.1364 |
| Carrier and lane track record | 0.780 | 0.1194 |
| Logistic regression | 0.837 | 0.1041 |
| **Gradient boosting, calibrated** | **0.863** | **0.0994** |

- The riskiest fifth of shipments contains 60% of all late deliveries.
- The median delivery-date prediction is off by 0.74 days on average, against 1.02 days for trusting the
  carrier's promised date.
- Delivery-date confidence holds: the 90% date was met by 93.8% of test shipments, the 95% date by 97.0%.
- The adjusted scorecard changes the picture: one LTL carrier ranks 5th of 8 on raw late rate and 1st once its
  harder freight is counted.
- Monitoring catches a real change: one carrier's service deteriorates after the training period, and the
  model under-predicts its late rate by 16 points on the test period until it is refitted.

These numbers come from simulated data with built-in patterns, so they are easier to learn than real freight.
Expect lower accuracy on real data, and measure it on your own most recent period.

## Quick start

```bash
git clone https://github.com/MDABDURRAUF/supplylens-shipment-risk.git
cd supplylens-shipment-risk
pip install -e ".[app,dev]"
streamlit run app.py
```

Command line:

```bash
shipment-risk sample --out shipments.csv
shipment-risk train --input shipments.csv --model model.joblib --report model_report.html
shipment-risk score --input shipments.csv --model model.joblib --out risk_list.csv --report risk_report.html
shipment-risk scorecard --input shipments.csv --since 2026-01-01
shipment-risk plan --model model.joblib --origin Chicago --destination Dallas --mode truckload --date 2026-07-06
```

Python:

```python
import shipment_risk as sr
from shipment_risk import evaluate

shipments = sr.load_shipments("shipments.csv")        # or sr.generate_sample_shipments()
model = sr.DelayModel().fit(shipments)
print(evaluate.summary(model))                        # accuracy on the most recent, unseen period

delivered, in_transit = sr.split_delivered(shipments)
risks = sr.risk_list(model, in_transit)               # late probability, expected and safe dates, reasons
print(risks.head())

origin, dest, miles = sr.lane_distance("Chicago", "Dallas")
print(sr.plan_shipment(model, "2026-07-06", origin, dest, miles, mode="truckload"))

print(sr.carrier_scorecard(shipments))
```

Only load model files you created yourself: a saved model is a Python pickle.

## Input data

One CSV file, one row per shipment. Leave `delivery_date` blank for shipments still in transit.

| Column | Required | Meaning |
|---|---|---|
| `shipment_id` | yes | Unique identifier |
| `ship_date`, `promised_date` | yes | Date tendered to the carrier, and the delivery date promised |
| `delivery_date` | no | Actual delivery date; blank while in transit |
| `carrier`, `mode` | yes | Carrier name, and mode such as truckload, LTL, parcel, air, intermodal rail |
| `origin_state`, `dest_state` | yes | Two-letter US state codes |
| `distance_miles` | yes | Road miles |
| `service`, `weight_lbs`, `value_usd` | no | Service level, weight, and value (used for value at risk) |
| `weather_risk` | no | Forecast severity on the route at ship time, from 0 to 1 |
| `origin`, `destination` | no | City or facility names, for display |

Files are validated on load; problems are reported with row counts, and nothing is dropped silently.
At least 500 delivered shipments are needed to fit the model.

## How it works

**Inputs**, all known when a shipment is tendered: the carrier and its late rate over the previous 60 days; how
busy the carrier is against its usual volume; the lane and its late rate over the previous 180 days; mode and
service; how tight the promise is for the distance; day of week, month, the year-end peak and US federal
holidays in transit; the weather forecast; and weight. States are grouped into the four US Census regions.

**No leakage.** A history feature for a shipment counts only shipments that had already been *delivered* before
it shipped, because a shipment on the road has no outcome yet. The data is split by date into training,
calibration and test periods, and a shipment enters training only if it was delivered before the calibration
period began. A test checks that changing later outcomes leaves earlier features untouched.

**Late probability.** Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`), then isotonic
calibration on the calibration period.

**Delivery dates.** One quantile regression per confidence level predicts the delay against the promised date.
Each is then shifted using the calibration period so that the stated share of shipments really arrive by the
predicted date (a split conformal adjustment). Deliveries happen on whole days, so "at most 1.6 days late"
becomes "at most 1 day late".

**Explanations.** Factors are groups of related inputs (carrier, lane, timing, promise tightness, weather and
so on). For one shipment, each factor is swapped for typical values from the training data and the change in
late probability is recorded. Overall importance is the loss of accuracy when a factor is shuffled. Both are
measured one factor at a time and are approximate.

**Carrier scorecard.** A second model, which is not told the carrier, estimates how likely each shipment was to
be late from its lane, mode, promise, season and weather, using cross-fitting so that no shipment is predicted
by a model that saw it. Each carrier's observed late rate is compared with that expected rate for its own
freight; differences beyond two standard errors are flagged.

**Alert economics.** Given the cost of a late delivery, the cost of acting on a shipment and the chance the
action works, the threshold table shows the net benefit of each alert threshold on the test period. Those
three numbers are business assumptions.

**Monitoring.** A backtest refits the model at earlier dates and scores each on the 60 days that followed.
The drift report measures how far recent inputs have moved from the training data (population stability
index). Predicted and observed late rates are compared by carrier, mode and state.

## Limits

- The sample data is simulated, with patterns that are cleaner than real freight.
- Accuracy is measured on one held-out period. It changes when carriers, lanes or conditions change, which is
  what the monitoring view is for.
- The model learns from history and cannot foresee an event unlike anything in it.
- Carriers and lanes with little history are pulled toward averages.
- Delivery-date confidence is met on average across shipments, not guaranteed for each kind of shipment.
- Explanations are one-factor-at-a-time approximations and do not add up exactly to the prediction.
- The scorecard adjusts for what is in the data; anything unrecorded that makes a carrier's freight harder is
  not adjusted for.

## Tests

```bash
pytest
```

Twenty tests: history features checked against a hand-worked example, no leakage from later outcomes, time-based
splitting, accuracy against simpler alternatives, calibration, delivery-date coverage, the scorecard, the risk
list, the planner, reports and the command line.

## Related projects

- [SupplyLens](https://github.com/MDABDURRAUF/supplylens): supply chain disruption analytics on purchase orders
- [SupplyLens Control Tower](https://github.com/MDABDURRAUF/supplylens-control-tower): track decisions and outcomes
- [SupplyLens SME Sourcing Planner](https://github.com/MDABDURRAUF/supplylens-sme-sourcing): landed-cost sourcing for small businesses
- [SupplyLens Network Simulator](https://github.com/MDABDURRAUF/supplylens-network-simulator): disruption scenarios for service and goods networks

## Author

Md Abdur Rauf — ORCID [0000-0002-5105-9892](https://orcid.org/0000-0002-5105-9892)

## License

MIT. See [LICENSE](LICENSE).
