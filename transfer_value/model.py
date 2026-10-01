"""Candidate models, time-based cross-validation and evaluation."""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import BASIC_FEATURES, FEATURES, make_features

BASELINE = "Median baseline"


def _log_target(regressor):
    # Market values are heavily right-skewed (a few 100m+ stars), so every
    # model learns log(1 + value); predict() converts back to euros.
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def candidate_models(seed=42, n_estimators=500):
    """Return {name: (unfitted estimator, feature columns)}."""
    return {
        BASELINE: (_log_target(DummyRegressor(strategy="median")), FEATURES),
        "Ridge regression": (
            _log_target(make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
            FEATURES,
        ),
        "Random Forest (basic features)": (
            _log_target(
                RandomForestRegressor(
                    n_estimators=n_estimators, min_samples_leaf=3, random_state=seed, n_jobs=-1
                )
            ),
            BASIC_FEATURES,
        ),
        "Random Forest": (
            _log_target(
                RandomForestRegressor(
                    n_estimators=n_estimators, min_samples_leaf=3, random_state=seed, n_jobs=-1
                )
            ),
            FEATURES,
        ),
        "Gradient Boosting": (
            _log_target(
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=seed,
                )
            ),
            FEATURES,
        ),
    }


def time_splits(seasons):
    """Expanding-window splits: train on all earlier seasons, test on the next.

    Yields (train_mask, test_mask, test_season). The model never sees the
    future, which a random K-fold split would allow.
    """
    seasons = pd.Series(seasons)
    ordered = sorted(seasons.unique())
    for test_season in ordered[1:]:
        yield (seasons < test_season).values, (seasons == test_season).values, test_season


def score(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return {
        "mae_eur": float(mean_absolute_error(actual, predicted)),
        "rmse_eur": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "r2": float(r2_score(actual, predicted)),
        "r2_log": float(r2_score(np.log1p(actual), np.log1p(predicted))),
    }


def compare_models(df, seed=42, n_estimators=500):
    """Cross-validate every candidate, then score each on the held-out season.

    The latest season is the hold-out. Model selection uses only the earlier
    seasons (expanding-window CV), so the hold-out score is an honest estimate.

    Returns (comparison DataFrame, best model name, hold-out predictions of
    the best model as a Series aligned with the hold-out rows).
    """
    X, y = make_features(df), df[config.TARGET]
    holdout_season = df["season"].max()
    dev = (df["season"] < holdout_season).values
    test = ~dev
    if df.loc[dev, "season"].nunique() < 2:
        raise ValueError("need at least three seasons: two for cross-validation, one hold-out")

    rows, predictions = [], {}
    for name, (estimator, columns) in candidate_models(seed, n_estimators).items():
        fold_mae = []
        for train_mask, test_mask, _ in time_splits(df.loc[dev, "season"]):
            X_dev, y_dev = X.loc[dev, columns], y[dev]
            estimator.fit(X_dev[train_mask], y_dev[train_mask])
            fold_mae.append(
                mean_absolute_error(y_dev[test_mask], estimator.predict(X_dev[test_mask]))
            )

        estimator.fit(X.loc[dev, columns], y[dev])
        predictions[name] = pd.Series(estimator.predict(X.loc[test, columns]), index=df.index[test])
        row = {"model": name, "n_features": len(columns), "cv_mae_eur": float(np.mean(fold_mae))}
        row.update({"holdout_" + k: v for k, v in score(y[test], predictions[name]).items()})
        rows.append(row)
        print("  {:<32} CV MAE €{:5.2f}m   hold-out MAE €{:5.2f}m".format(
            name, row["cv_mae_eur"] / 1e6, row["holdout_mae_eur"] / 1e6))

    comparison = pd.DataFrame(rows)
    best = comparison.sort_values("cv_mae_eur").iloc[0]["model"]
    return comparison, best, predictions[best]


def fit_final(df, name, seed=42, n_estimators=500):
    """Refit the chosen model on every season, for serving."""
    estimator, columns = candidate_models(seed, n_estimators)[name]
    estimator.fit(make_features(df)[columns], df[config.TARGET])
    return {
        "name": name,
        "model": estimator,
        "features": columns,
        "seasons": sorted(int(s) for s in df["season"].unique()),
    }


def predict(artifact, raw):
    """Predict euros for a DataFrame of raw inputs (see features.RAW_*)."""
    return artifact["model"].predict(make_features(raw)[artifact["features"]])


def save(artifact, path=config.MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load(path=config.MODEL_PATH):
    return joblib.load(path)
