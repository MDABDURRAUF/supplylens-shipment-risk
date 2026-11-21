"""The delay model: the chance a shipment arrives late, and how late it is likely to be.

Two kinds of prediction come from one ``DelayModel``:

* **late probability**: gradient-boosted trees, then calibrated so that a
  predicted 30% means about 30% of such shipments are late;
* **delay quantiles**: how many days after (or before) the promised date the
  shipment will arrive, at several confidence levels, adjusted on held-back
  data so that the stated confidence is actually met.

All splits are by time. A shipment is used for training only if it had been
*delivered* before the calibration period starts, and for calibration only if
delivered before the test period starts, so no outcome is used before it
could have been known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import split_delivered, validate_shipments
from .features import CATEGORICAL, FEATURES, build_features


def _boosted(estimator, features: list[str]) -> Pipeline:
    """Gradient boosting on the given features, with text columns handled as categories."""
    cats = [c for c in features if c in CATEGORICAL]
    nums = [c for c in features if c not in CATEGORICAL]
    pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cats),
                             ("num", "passthrough", nums)])
    estimator.set_params(categorical_features=list(range(len(cats))))
    return Pipeline([("pre", pre), ("model", estimator)])


def _logistic(features: list[str]) -> Pipeline:
    cats = [c for c in features if c in CATEGORICAL]
    nums = [c for c in features if c not in CATEGORICAL]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                             ("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums)])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])


@dataclass
class DelayModel:
    """Fit with ``fit(shipments)``, then ``predict(new_shipments)``.

    ``quantiles``        confidence levels for the delay prediction
    ``train_fraction``   share of the time span used for training
    ``calib_fraction``   share used to calibrate probabilities and quantiles
    ``exclude``          feature names to leave out (see ``operations.carrier_scorecard``)

    The remaining share of the time span is the held-out test period; its
    predictions are kept in ``holdout_`` for every check in ``evaluate``.
    """

    quantiles: tuple = (0.5, 0.8, 0.9, 0.95)
    train_fraction: float = 0.70
    calib_fraction: float = 0.15
    exclude: tuple = ()
    random_state: int = 0
    features_: list = field(default_factory=list, repr=False)
    prior_: float = 0.2
    history_: pd.DataFrame | None = field(default=None, repr=False)
    holdout_: pd.DataFrame | None = field(default=None, repr=False)
    background_: pd.DataFrame | None = field(default=None, repr=False)
    train_features_: pd.DataFrame | None = field(default=None, repr=False)
    split_: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ fit
    def fit(self, shipments: pd.DataFrame) -> "DelayModel":
        if "delivered" not in shipments.columns:
            shipments = validate_shipments(shipments)
        delivered, _ = split_delivered(shipments)
        delivered = delivered.sort_values("ship_date").reset_index(drop=True)
        if len(delivered) < 500:
            raise ValueError("At least 500 delivered shipments are needed to fit the model")
        if not 0 < self.train_fraction < 1 or not 0 < self.calib_fraction < 1 or self.train_fraction + self.calib_fraction > 1:
            raise ValueError("train_fraction and calib_fraction must be positive and sum to at most 1")
        ship = delivered["ship_date"]
        t1 = ship.quantile(self.train_fraction, interpolation="nearest")
        t2 = ship.quantile(min(1.0, self.train_fraction + self.calib_fraction), interpolation="nearest")
        has_test = self.train_fraction + self.calib_fraction < 1
        train = (ship < t1) & (delivered["delivery_date"] < t1)
        if has_test:
            calib = (ship >= t1) & (ship < t2) & (delivered["delivery_date"] < t2)
            test = ship >= t2
        else:
            calib, test = ship >= t1, pd.Series(False, index=ship.index)
        if delivered.loc[train, "late"].nunique() < 2 or delivered.loc[calib, "late"].nunique() < 2:
            raise ValueError("The training and calibration periods must each contain late and on-time shipments")

        self.prior_ = float(delivered.loc[train, "late"].mean())
        self.features_ = [f for f in FEATURES if f not in self.exclude]
        X = build_features(delivered, prior=self.prior_)[self.features_]
        y, delay = delivered["late"].astype(int), delivered["delay_days"].astype(float)

        self.classifier_ = _boosted(
            HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
                                           random_state=self.random_state), self.features_).fit(X[train], y[train])
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            self.classifier_.predict_proba(X[calib])[:, 1], y[calib])
        self.logistic_ = _logistic(self.features_).fit(X[train], y[train])

        self.regressors_, self.offsets_ = {}, {}
        for q in self.quantiles:
            reg = _boosted(HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, learning_rate=0.06,
                                                         max_leaf_nodes=15, random_state=self.random_state), self.features_)
            self.regressors_[q] = reg.fit(X[train], delay[train])
            # Shift the prediction so that the stated share of held-back shipments really arrive by it.
            resid = np.sort(delay[calib].to_numpy() - reg.predict(X[calib]))
            rank = min(len(resid) - 1, int(np.ceil((len(resid) + 1) * q)) - 1)
            self.offsets_[q] = float(resid[rank])

        self.history_ = delivered
        self.train_features_ = X[train]
        self.background_ = X[train].sample(min(80, int(train.sum())), random_state=self.random_state)
        self.split_ = {"train_rows": int(train.sum()), "calibration_rows": int(calib.sum()), "test_rows": int(test.sum()),
                       "train_until": t1, "test_from": t2 if has_test else None,
                       "first_ship_date": ship.min(), "last_ship_date": ship.max(), "train_late_rate": self.prior_}
        if has_test and test.sum():
            h = delivered.loc[test, ["shipment_id", "ship_date", "promised_date", "delivery_date", "carrier", "mode",
                                     "origin_state", "dest_state", "value_usd", "late", "delay_days"]].copy()
            Xt = X[test]
            h["p_constant"] = self.prior_
            h["p_history"] = (Xt["carrier_late_rate"] + Xt["lane_late_rate"]).to_numpy() / 2 if {"carrier_late_rate", "lane_late_rate"} <= set(self.features_) else self.prior_
            h["p_logistic"] = self.logistic_.predict_proba(Xt)[:, 1]
            h["p_boosted_raw"] = self.classifier_.predict_proba(Xt)[:, 1]
            h["p"] = self.calibrator_.predict(h["p_boosted_raw"].to_numpy())
            for q, col in zip(self.quantiles, self._delay_matrix(Xt).T):
                h[f"q{round(q * 100)}"] = col
            self.holdout_ = h.reset_index(drop=True)
            self.holdout_features_ = Xt.reset_index(drop=True)
        else:
            self.holdout_, self.holdout_features_ = None, None
        return self

    # -------------------------------------------------------------- predict
    def _check(self) -> None:
        if not self.features_:
            raise RuntimeError("Call fit() first")

    def features(self, shipments: pd.DataFrame) -> pd.DataFrame:
        """Feature table for new shipments, using the training history for the history features."""
        self._check()
        if "delivered" not in shipments.columns:
            shipments = validate_shipments(shipments)
        return build_features(shipments, history=self.history_, prior=self.prior_)[self.features_]

    def _delay_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted delay (days after the promised date) per quantile, never decreasing across quantiles."""
        cols = [self.regressors_[q].predict(X) + self.offsets_[q] for q in self.quantiles]
        return np.maximum.accumulate(np.column_stack(cols), axis=1) if cols else np.empty((len(X), 0))

    def proba_from_features(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated late probability for an already-built feature table."""
        return self.calibrator_.predict(self.classifier_.predict_proba(X[self.features_])[:, 1])

    def predict_proba(self, shipments: pd.DataFrame) -> pd.Series:
        """Calibrated probability that each shipment is delivered after its promised date."""
        X = self.features(shipments)
        return pd.Series(self.proba_from_features(X), index=shipments.index, name="late_probability")

    def predict(self, shipments: pd.DataFrame) -> pd.DataFrame:
        """Late probability and predicted delivery dates for each shipment.

        ``delay_qNN`` is the delay in days (negative means early) that NN% of
        similar shipments do not exceed; ``date_qNN`` is the corresponding
        delivery date, so ``date_q90`` is the date to quote if you want to be
        right nine times in ten.
        """
        if "delivered" not in shipments.columns:
            shipments = validate_shipments(shipments)
        X = self.features(shipments)
        out = pd.DataFrame({"late_probability": self.proba_from_features(X)}, index=shipments.index)
        for q, col in zip(self.quantiles, self._delay_matrix(X).T):
            name = round(q * 100)
            out[f"delay_q{name}"] = col
            # Deliveries happen on whole days, so "no more than 1.6 days late" means "no more than 1 day late".
            out[f"date_q{name}"] = shipments["promised_date"] + pd.to_timedelta(np.floor(col), unit="D")
        return out

    # ----------------------------------------------------------------- files
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "DelayModel":
        model = joblib.load(path)
        if not isinstance(model, DelayModel):
            raise ValueError("The file does not contain a DelayModel")
        return model
