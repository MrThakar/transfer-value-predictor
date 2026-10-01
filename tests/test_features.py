import pandas as pd
import pytest

from transfer_value import config
from transfer_value.features import (
    FEATURES,
    build_dataset,
    club_strength,
    label_season,
    make_features,
    season_stats,
)


def test_label_season_splits_on_july():
    seasons = label_season(["2023-08-12", "2024-05-19", "2024-06-30", "2024-07-01"])
    assert list(seasons) == [2023, 2023, 2023, 2024]


def test_season_stats_aggregates_league_games_only(tables):
    stats = season_stats(tables["appearances"], ["GB1"]).set_index(["player_id", "season"])
    row = stats.loc[(1, 2023)]
    assert row["appearances"] == 10
    assert row["minutes_played"] == 900
    assert row["goals"] == 10  # the 3 cup goals are excluded


def test_season_stats_assigns_club_with_most_minutes(tables):
    stats = season_stats(tables["appearances"], ["GB1"]).set_index(["player_id", "season"])
    row = stats.loc[(2, 2023)]
    assert row["minutes_played"] == 810  # both clubs combined
    assert row["club_id"] == 2


def test_club_strength_points_and_goal_difference(tables):
    strength = club_strength(tables["games"], ["GB1"]).set_index(["club_id", "season"])
    # 2023: club 1 won 3-0 at home then drew 1-1 away -> 4 points from 2 games.
    assert strength.loc[(1, 2023), "club_points_per_game"] == 2.0
    assert strength.loc[(1, 2023), "club_goal_diff_per_game"] == 1.5
    assert strength.loc[(2, 2023), "club_points_per_game"] == 0.5
    assert strength.loc[(2, 2022), "club_points_per_game"] == 3.0


def test_build_dataset_joins_target_age_and_previous_season(tables):
    df = build_dataset(tables, seasons=[2022, 2023], leagues=["GB1"], min_minutes=450)
    df = df.set_index(["player_id", "season"])
    assert len(df) == 3

    first, second = df.loc[(1, 2022)], df.loc[(1, 2023)]
    assert first[config.TARGET] == 20e6
    assert second[config.TARGET] == 40e6
    assert first["age"] == pytest.approx(23.0, abs=0.01)
    assert first["prev_minutes"] == 0
    assert second["prev_minutes"] == 900
    assert second["prev_goals"] == 5
    assert second["club_points_per_game"] == 2.0


def test_build_dataset_applies_minutes_filter(tables):
    df = build_dataset(tables, seasons=[2022, 2023], leagues=["GB1"], min_minutes=850)
    assert set(df["player_id"]) == {1}


def test_build_dataset_drops_rows_without_a_nearby_valuation(tables):
    tables["player_valuations"] = tables["player_valuations"].assign(
        date=lambda v: v["date"].where(v["player_id"] != 2, pd.Timestamp("2025-01-01"))
    )
    df = build_dataset(tables, seasons=[2022, 2023], leagues=["GB1"], min_minutes=450)
    assert 2 not in set(df["player_id"])


def test_make_features_columns_rates_and_one_hots(tables):
    df = build_dataset(tables, seasons=[2022, 2023], leagues=["GB1"], min_minutes=450)
    X = make_features(df)
    assert list(X.columns) == FEATURES
    assert not X.isna().any().any()

    striker = X[(df["player_id"] == 1) & (df["season"] == 2023)].iloc[0]
    assert striker["goals_per_90"] == pytest.approx(1.0)
    assert striker["position_Attack"] == 1
    assert striker["position_Defender"] == 0
    assert striker["sub_position_Centre-Forward"] == 1
    assert striker["league_GB1"] == 1
    assert striker["played_prev_season"] == 1

    back = X[df["player_id"] == 2].iloc[0]
    assert back["height_in_cm"] == config.DEFAULT_HEIGHT_CM  # missing height filled
    assert back["played_prev_season"] == 0


def test_make_features_handles_zero_minutes():
    raw = pd.DataFrame([{
        "age": 25, "height_in_cm": 180, "appearances": 0, "minutes_played": 0, "goals": 0,
        "assists": 0, "yellow_cards": 0, "red_cards": 0, "prev_minutes": 0, "prev_goals": 0,
        "prev_assists": 0, "club_points_per_game": 1.5, "club_goal_diff_per_game": 0.0,
        "position": "Midfield", "sub_position": None, "league": "ES1",
    }])
    X = make_features(raw)
    assert X.loc[0, "goals_per_90"] == 0
    assert not X.isna().any().any()
