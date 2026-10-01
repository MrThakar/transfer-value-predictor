"""Download and load the raw Transfermarkt tables."""

import pandas as pd
import requests

from . import config

COLUMNS = {
    "players": ["player_id", "name", "date_of_birth", "position", "sub_position", "height_in_cm"],
    "appearances": [
        "player_id",
        "player_club_id",
        "date",
        "competition_id",
        "goals",
        "assists",
        "minutes_played",
        "yellow_cards",
        "red_cards",
    ],
    "player_valuations": ["player_id", "date", "market_value_in_eur"],
    "games": [
        "competition_id",
        "season",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
    ],
}
DATE_COLUMNS = {
    "players": ["date_of_birth"],
    "appearances": ["date"],
    "player_valuations": ["date"],
    "games": [],
}


def download_table(name, force=False):
    """Download one gzipped CSV to data/raw (skipped if already cached)."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / "{}.csv.gz".format(name)
    if path.exists() and not force:
        print("  using cached {}".format(path.name))
        return path

    url = "{}/{}.csv.gz".format(config.BASE_URL, name)
    print("  downloading {}".format(url))
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(path)
    return path


def load_raw(force=False):
    """Return a dict of DataFrames keyed by table name."""
    print("Fetching data...")
    tables = {}
    for name in config.TABLES:
        path = download_table(name, force)
        tables[name] = pd.read_csv(
            path, usecols=COLUMNS[name], parse_dates=DATE_COLUMNS[name]
        )
    return tables
