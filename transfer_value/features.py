"""Turn raw match-level tables into one modelling row per player-season."""

import numpy as np
import pandas as pd

from . import config

# Columns a caller must supply to get a prediction (training table or API).
RAW_NUMERIC = [
    "age",
    "height_in_cm",
    "appearances",
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "prev_minutes",
    "prev_goals",
    "prev_assists",
    "club_points_per_game",
    "club_goal_diff_per_game",
]
RAW_CATEGORICAL = ["position", "sub_position", "league"]

POSITION_FEATURES = ["position_" + p for p in config.POSITIONS]
SUB_POSITION_FEATURES = ["sub_position_" + p for p in config.SUB_POSITIONS]
LEAGUE_FEATURES = ["league_" + code for code in config.LEAGUES]

# The first version of the model used only these; kept for the ablation.
BASIC_FEATURES = [
    "age",
    "appearances",
    "minutes_played",
    "goals",
    "assists",
    "goals_per_90",
    "assists_per_90",
    "yellow_cards",
    "red_cards",
] + POSITION_FEATURES

FEATURES = (
    BASIC_FEATURES
    + [
        "height_in_cm",
        "prev_minutes",
        "prev_goals",
        "prev_assists",
        "played_prev_season",
        "club_points_per_game",
        "club_goal_diff_per_game",
    ]
    + SUB_POSITION_FEATURES
    + LEAGUE_FEATURES
)


def label_season(dates):
    """Label a date with the year its season starts in (Aug 2023 - May 2024 -> 2023)."""
    dates = pd.to_datetime(pd.Series(dates))
    return pd.Series(
        np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1), index=dates.index
    )


def season_stats(appearances, leagues):
    """Aggregate league appearances to one row per player per season.

    A player who moved mid-season gets their combined stats, attributed to the
    club and league where they played the most minutes.
    """
    app = appearances[appearances["competition_id"].isin(leagues)].copy()
    app["season"] = label_season(app["date"])

    totals = (
        app.groupby(["player_id", "season"])
        .agg(
            appearances=("minutes_played", "size"),
            minutes_played=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
        )
        .reset_index()
    )
    by_club = app.groupby(
        ["player_id", "season", "player_club_id", "competition_id"], as_index=False
    )["minutes_played"].sum()
    primary = (
        by_club.sort_values("minutes_played")
        .drop_duplicates(["player_id", "season"], keep="last")
        .rename(columns={"player_club_id": "club_id", "competition_id": "league"})
    )
    return totals.merge(primary[["player_id", "season", "club_id", "league"]])


def club_strength(games, leagues):
    """Points and goal difference per game for every club-season."""
    games = games[games["competition_id"].isin(leagues)]
    sides = []
    for own, other in (("home", "away"), ("away", "home")):
        scored = games["{}_club_goals".format(own)]
        conceded = games["{}_club_goals".format(other)]
        sides.append(
            pd.DataFrame(
                {
                    "club_id": games["{}_club_id".format(own)],
                    "season": games["season"],
                    "points": np.select([scored > conceded, scored == conceded], [3, 1], 0),
                    "goal_diff": scored - conceded,
                }
            )
        )
    table = pd.concat(sides).groupby(["club_id", "season"]).agg(
        club_points_per_game=("points", "mean"),
        club_goal_diff_per_game=("goal_diff", "mean"),
    )
    return table.reset_index()


def build_dataset(tables, seasons, leagues=None, min_minutes=config.DEFAULT_MIN_MINUTES):
    """Return one row per player-season with raw inputs and the target value."""
    leagues = list(leagues or config.LEAGUES)
    stats = season_stats(tables["appearances"], leagues)

    # Previous-season output (0 if the player wasn't in these leagues).
    prev = stats[["player_id", "season", "minutes_played", "goals", "assists"]].copy()
    prev["season"] += 1
    prev.columns = ["player_id", "season", "prev_minutes", "prev_goals", "prev_assists"]
    df = stats.merge(prev, on=["player_id", "season"], how="left")
    prev_columns = ["prev_minutes", "prev_goals", "prev_assists"]
    df[prev_columns] = df[prev_columns].fillna(0)

    df = df[df["season"].isin(seasons) & (df["minutes_played"] >= min_minutes)]
    df = df.merge(tables["players"], on="player_id", how="inner")
    df = df[df["position"].isin(config.POSITIONS)].dropna(subset=["date_of_birth"])

    strength = club_strength(tables["games"], leagues)
    df = df.merge(strength, on=["club_id", "season"], how="left")
    for column in ("club_points_per_game", "club_goal_diff_per_game"):
        df[column] = df[column].fillna(strength[column].median())

    # Target: the Transfermarkt valuation closest to the end of that season
    # (15 June), so the value reflects the season the stats describe.
    df["season_end"] = pd.to_datetime((df["season"] + 1).astype(str) + "-06-15")
    df["age"] = (df["season_end"] - df["date_of_birth"]).dt.days / 365.25
    valuations = tables["player_valuations"].dropna().rename(columns={"date": "valuation_date"})
    df = pd.merge_asof(
        df.sort_values("season_end"),
        valuations.sort_values("valuation_date"),
        left_on="season_end",
        right_on="valuation_date",
        by="player_id",
        direction="nearest",
        tolerance=pd.Timedelta(days=90),
    )
    df = df.dropna(subset=[config.TARGET])
    return df.sort_values(["season", "player_id"]).reset_index(drop=True)


def make_features(df):
    """Build the model's feature matrix from raw inputs.

    Used for both training and serving so the two can never drift apart.
    """
    X = pd.DataFrame(index=df.index)
    for column in RAW_NUMERIC:
        X[column] = pd.to_numeric(df[column], errors="coerce")

    # A handful of heights in the source are missing or obviously wrong.
    height = X["height_in_cm"].where(X["height_in_cm"].between(150, 215))
    X["height_in_cm"] = height.fillna(config.DEFAULT_HEIGHT_CM)

    nineties = (X["minutes_played"] / 90).replace(0, np.nan)
    X["goals_per_90"] = (X["goals"] / nineties).fillna(0)
    X["assists_per_90"] = (X["assists"] / nineties).fillna(0)
    X["played_prev_season"] = (X["prev_minutes"] > 0).astype(int)

    for position in config.POSITIONS:
        X["position_" + position] = (df["position"] == position).astype(int)
    for sub_position in config.SUB_POSITIONS:
        X["sub_position_" + sub_position] = (df["sub_position"] == sub_position).astype(int)
    for league in config.LEAGUES:
        X["league_" + league] = (df["league"] == league).astype(int)

    return X[FEATURES]
