"""Small synthetic tables shaped like the raw Transfermarkt data (no network)."""

import numpy as np
import pandas as pd
import pytest

from transfer_value import config


def make_appearance(player_id, date, minutes=90, goals=0, assists=0, club_id=1,
                    competition_id="GB1"):
    return {
        "player_id": player_id,
        "player_club_id": club_id,
        "date": pd.Timestamp(date),
        "competition_id": competition_id,
        "goals": goals,
        "assists": assists,
        "minutes_played": minutes,
        "yellow_cards": 0,
        "red_cards": 0,
    }


@pytest.fixture
def tables():
    """Two players, two seasons (2022 and 2023), one league."""
    appearances = pd.DataFrame(
        # Player 1: 10 full games in each season, 5 goals in 2022, 10 in 2023.
        [make_appearance(1, "2022-09-01", goals=1 if i < 5 else 0) for i in range(10)]
        + [make_appearance(1, "2023-09-01", goals=1) for _ in range(10)]
        # Player 2: 2023 only, split across two clubs (more minutes at club 2).
        + [make_appearance(2, "2023-10-01", club_id=1) for _ in range(3)]
        + [make_appearance(2, "2024-02-01", club_id=2, assists=1) for _ in range(6)]
        # A cup game that must be ignored.
        + [make_appearance(1, "2023-11-01", goals=3, competition_id="FAC")]
    )
    players = pd.DataFrame(
        {
            "player_id": [1, 2],
            "name": ["Striker One", "Back Two"],
            "date_of_birth": pd.to_datetime(["2000-06-15", "1995-06-15"]),
            "position": ["Attack", "Defender"],
            "sub_position": ["Centre-Forward", "Left-Back"],
            "height_in_cm": [185.0, np.nan],
        }
    )
    valuations = pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "date": pd.to_datetime(["2023-06-10", "2024-06-01", "2024-06-20"]),
            "market_value_in_eur": [20e6, 40e6, 5e6],
        }
    )
    games = pd.DataFrame(
        {
            "competition_id": ["GB1", "GB1", "GB1"],
            "season": [2023, 2023, 2022],
            "home_club_id": [1, 2, 1],
            "away_club_id": [2, 1, 2],
            "home_club_goals": [3, 1, 0],
            "away_club_goals": [0, 1, 2],
        }
    )
    return {
        "appearances": appearances,
        "players": players,
        "player_valuations": valuations,
        "games": games,
    }


@pytest.fixture
def modelling_frame():
    """200 synthetic player-seasons over four seasons with a learnable target."""
    rng = np.random.RandomState(0)
    n = 200
    minutes = rng.randint(450, 3400, n)
    goals = rng.poisson(4, n)
    age = rng.uniform(18, 35, n)
    return pd.DataFrame(
        {
            "name": ["Player {}".format(i) for i in range(n)],
            "season": np.repeat([2021, 2022, 2023, 2024], n // 4),
            "age": age,
            "height_in_cm": rng.normal(182, 6, n),
            "appearances": minutes // 80,
            "minutes_played": minutes,
            "goals": goals,
            "assists": rng.poisson(3, n),
            "yellow_cards": rng.poisson(3, n),
            "red_cards": 0,
            "prev_minutes": rng.randint(0, 3400, n),
            "prev_goals": rng.poisson(4, n),
            "prev_assists": rng.poisson(3, n),
            "club_points_per_game": rng.uniform(0.7, 2.4, n),
            "club_goal_diff_per_game": rng.uniform(-1, 1.5, n),
            "position": rng.choice(config.POSITIONS, n),
            "sub_position": rng.choice(config.SUB_POSITIONS, n),
            "league": rng.choice(list(config.LEAGUES), n),
            config.TARGET: 1e6 * (1 + goals) * minutes / 1000 * (36 - age) / 10,
        }
    )
