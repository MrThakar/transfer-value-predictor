"""Paths and constants shared across the package."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "model.joblib"

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ["players", "appearances", "player_valuations", "games"]

# Transfermarkt competition ids for the top five European leagues.
LEAGUES = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "IT1": "Serie A",
    "L1": "Bundesliga",
    "FR1": "Ligue 1",
}
DEFAULT_SEASONS = [2021, 2022, 2023, 2024, 2025]  # season start years
DEFAULT_MIN_MINUTES = 450

POSITIONS = ["Goalkeeper", "Defender", "Midfield", "Attack"]
SUB_POSITIONS = [
    "Goalkeeper",
    "Centre-Back",
    "Left-Back",
    "Right-Back",
    "Defensive Midfield",
    "Central Midfield",
    "Attacking Midfield",
    "Left Midfield",
    "Right Midfield",
    "Left Winger",
    "Right Winger",
    "Second Striker",
    "Centre-Forward",
]
DEFAULT_HEIGHT_CM = 182.0  # dataset median, used when a height is missing

TARGET = "market_value_in_eur"

# Detailed positions grouped by broad position (used by the front end's form).
POSITION_GROUPS = {
    "Goalkeeper": ["Goalkeeper"],
    "Defender": ["Centre-Back", "Left-Back", "Right-Back"],
    "Midfield": [
        "Defensive Midfield",
        "Central Midfield",
        "Attacking Midfield",
        "Left Midfield",
        "Right Midfield",
    ],
    "Attack": ["Left Winger", "Right Winger", "Second Striker", "Centre-Forward"],
}
FRONTEND_DIR = ROOT / "frontend" / "dist"
