"""Matplotlib charts for the README and for error analysis."""

import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from . import config

POSITION_COLORS = {
    "Goalkeeper": "#7f7f7f",
    "Defender": "#1f77b4",
    "Midfield": "#2ca02c",
    "Attack": "#d62728",
}


def millions(value, _pos=None):
    return "€{:g}m".format(value / 1e6)


def actual_vs_predicted(test, title, path):
    """Scatter of actual vs. predicted value; expects a `predicted_value_eur` column."""
    actual, predicted = test[config.TARGET], test["predicted_value_eur"]
    fig, ax = plt.subplots(figsize=(8, 8))
    for position in config.POSITIONS:
        subset = test[test["position"] == position]
        ax.scatter(
            subset[config.TARGET],
            subset["predicted_value_eur"],
            s=14,
            alpha=0.5,
            color=POSITION_COLORS[position],
            label=position,
        )

    low = min(actual.min(), predicted.min()) * 0.8
    high = max(actual.max(), predicted.max()) * 1.25
    ax.plot([low, high], [low, high], "k--", linewidth=1, label="Perfect prediction")

    # Label the five biggest misses so the chart tells a story.
    for idx in (predicted - actual).abs().nlargest(5).index:
        ax.annotate(
            test.loc[idx, "name"],
            (actual[idx], predicted[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_xlabel("Actual market value (Transfermarkt)")
    ax.set_ylabel("Predicted market value")
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def model_comparison(comparison, best, path):
    """Horizontal bars of hold-out MAE per model, best model highlighted."""
    ordered = comparison.sort_values("holdout_mae_eur", ascending=False)
    colors = ["#d62728" if name == best else "#9aa7b4" for name in ordered["model"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(ordered["model"], ordered["holdout_mae_eur"] / 1e6, color=colors)
    for bar in bars:
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            "€{:.2f}m".format(bar.get_width()),
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, ordered["holdout_mae_eur"].max() / 1e6 * 1.15)
    ax.set_xlabel("Mean absolute error on the held-out season (lower is better)")
    ax.set_title("Model comparison")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def feature_importance(importances, path, top=15):
    """Bar chart of a pandas Series of permutation importances."""
    top_features = importances.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top_features.index, top_features.values / 1e6, color="#1f77b4")
    ax.set_xlabel("Increase in hold-out MAE when the feature is shuffled (€m)")
    ax.set_title("Permutation feature importance (top {})".format(top))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
