"""Run the full pipeline: download, build features, compare models, plot, save.

Usage:
    python -m transfer_value.train
    python -m transfer_value.train --leagues GB1 --seasons 2022 2023 2024 2025
"""

import argparse
import json

import pandas as pd
from sklearn.inspection import permutation_importance

from . import config, data, model, plots
from .features import build_dataset, make_features


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=config.DEFAULT_SEASONS,
        help="Season start years; the latest one is held out as the test set.",
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=list(config.LEAGUES),
        choices=list(config.LEAGUES),
        help="Transfermarkt competition ids to include.",
    )
    parser.add_argument("--min-minutes", type=int, default=config.DEFAULT_MIN_MINUTES,
                        help="Drop player-seasons with fewer league minutes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true", help="Re-download raw data.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tables = data.load_raw(force=args.refresh)
    df = build_dataset(tables, args.seasons, args.leagues, args.min_minutes)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_DIR / "player_seasons.csv", index=False)
    print("Built {} player-seasons ({} seasons, {} leagues)\n".format(
        len(df), df["season"].nunique(), df["league"].nunique()))

    print("Comparing models (expanding-window CV, then hold-out)...")
    comparison, best, predicted = model.compare_models(df, seed=args.seed)
    holdout_season = int(df["season"].max())
    test = df.loc[predicted.index].copy()
    test["predicted_value_eur"] = predicted
    best_row = comparison.set_index("model").loc[best]
    print("\nSelected by CV: {}".format(best))

    # Error by league, to show where the model is strong and weak.
    by_league = test.groupby("league").apply(
        lambda g: (g["predicted_value_eur"] - g[config.TARGET]).abs().mean()
    )

    # Permutation importance of the selected model on the hold-out season.
    artifact = model.fit_final(df[df["season"] < holdout_season], best, seed=args.seed)
    X_test = make_features(test)[artifact["features"]]
    result = permutation_importance(
        artifact["model"], X_test, test[config.TARGET],
        scoring="neg_mean_absolute_error", n_repeats=5, random_state=args.seed, n_jobs=-1,
    )
    importances = pd.Series(result.importances_mean, index=artifact["features"])

    comparison.to_csv(config.OUTPUT_DIR / "model_comparison.csv", index=False, float_format="%.4f")
    columns = ["name", "league", "position", "sub_position", "age", "minutes_played", "goals",
               "assists", config.TARGET, "predicted_value_eur"]
    test[columns].sort_values(config.TARGET, ascending=False).to_csv(
        config.OUTPUT_DIR / "test_predictions.csv", index=False, float_format="%.1f"
    )
    metrics = {
        "best_model": best,
        "holdout_season": holdout_season,
        "train_rows": int((df["season"] < holdout_season).sum()),
        "holdout_rows": int(len(test)),
        "holdout": {k: float(v) for k, v in best_row.items() if k.startswith("holdout_")},
        "holdout_mae_by_league": {k: float(v) for k, v in by_league.items()},
        # Relative error is easier to interpret than euros across a 100k-200m range.
        "holdout_median_pct_error": float(
            ((test["predicted_value_eur"] - test[config.TARGET]).abs() / test[config.TARGET])
            .median() * 100
        ),
    }
    with open(config.OUTPUT_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    title = "{}/{} season, {}: actual vs. predicted value\nMAE {}  |  R² (log scale) {:.2f}".format(
        holdout_season, str(holdout_season + 1)[-2:], best,
        plots.millions(round(best_row["holdout_mae_eur"], -5)), best_row["holdout_r2_log"],
    )
    plots.actual_vs_predicted(test, title, config.OUTPUT_DIR / "actual_vs_predicted.png")
    plots.model_comparison(comparison, best, config.OUTPUT_DIR / "model_comparison.png")
    plots.feature_importance(importances, config.OUTPUT_DIR / "feature_importance.png")

    # The served model is refit on every season, including the hold-out.
    model.save(model.fit_final(df, best, seed=args.seed))
    print("Saved outputs to {}/ and model to {}".format(
        config.OUTPUT_DIR.name, config.MODEL_PATH.relative_to(config.ROOT)))


if __name__ == "__main__":
    main()
