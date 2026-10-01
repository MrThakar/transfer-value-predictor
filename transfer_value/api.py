"""FastAPI service that serves the trained model and the React front end.

Run with:
    uvicorn transfer_value.api:app --reload
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, model

app = FastAPI(
    title="Transfer Value Predictor",
    description="Estimate a footballer's market value from one season of statistics.",
)
# The Vite dev server runs on a different port from the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlayerSeason(BaseModel):
    age: float = Field(..., ge=15, le=45, description="Age at the end of the season")
    position: Literal[tuple(config.POSITIONS)]
    sub_position: Optional[Literal[tuple(config.SUB_POSITIONS)]] = None
    league: Literal[tuple(config.LEAGUES)] = Field(
        "GB1", description="Transfermarkt competition id"
    )
    appearances: int = Field(..., ge=1, le=60)
    minutes_played: int = Field(..., ge=1, le=5400)
    goals: int = Field(0, ge=0)
    assists: int = Field(0, ge=0)
    yellow_cards: int = Field(0, ge=0)
    red_cards: int = Field(0, ge=0)
    prev_minutes: int = Field(0, ge=0, le=5400, description="League minutes last season")
    prev_goals: int = Field(0, ge=0)
    prev_assists: int = Field(0, ge=0)
    club_points_per_game: float = Field(1.37, ge=0, le=3)
    club_goal_diff_per_game: float = Field(0.0, ge=-5, le=5)
    height_in_cm: Optional[float] = Field(None, ge=150, le=215)


class Prediction(BaseModel):
    predicted_value_eur: float
    model: str


class Comparable(BaseModel):
    name: str
    league: str
    position: str
    age: float
    goals: int
    assists: int
    market_value_in_eur: float


COMPARABLE_COLUMNS = ["name", "league", "position", "age", "goals", "assists", config.TARGET]


def _output_path(filename):
    return Path(os.environ.get("TVP_OUTPUT_DIR", str(config.OUTPUT_DIR))) / filename


@lru_cache(maxsize=1)
def get_artifact():
    path = Path(os.environ.get("TVP_MODEL_PATH", str(config.MODEL_PATH)))
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="No trained model found. Run `python -m transfer_value.train` first.",
        )
    return model.load(path)


@lru_cache(maxsize=1)
def get_reference_players():
    """Real players from the hold-out season, used to put a prediction in context."""
    path = _output_path("test_predictions.csv")
    if not path.exists():
        return pd.DataFrame(columns=COMPARABLE_COLUMNS)
    return pd.read_csv(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    """Form options and headline model facts for the front end."""
    artifact = get_artifact()
    info = {
        "model": artifact["name"],
        "seasons": artifact["seasons"],
        "leagues": config.LEAGUES,
        "positions": config.POSITION_GROUPS,
        "holdout_median_pct_error": None,
        "holdout_season": None,
    }
    metrics_path = _output_path("metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        info["holdout_median_pct_error"] = metrics.get("holdout_median_pct_error")
        info["holdout_season"] = metrics["holdout_season"]
    return info


@app.post("/predict", response_model=Prediction)
def predict(player: PlayerSeason):
    artifact = get_artifact()
    payload = player.model_dump() if hasattr(player, "model_dump") else player.dict()
    value = model.predict(artifact, pd.DataFrame([payload]))[0]
    return Prediction(predicted_value_eur=round(float(value), -3), model=artifact["name"])


@app.get("/comparables", response_model=List[Comparable])
def comparables(
    value: float = Query(..., gt=0, description="A market value in euros"),
    position: Optional[Literal[tuple(config.POSITIONS)]] = None,
    limit: int = Query(5, ge=1, le=20),
):
    """Real players whose actual market value is closest to `value`."""
    players = get_reference_players()
    if position is not None:
        players = players[players["position"] == position]
    nearest = (players[config.TARGET] - value).abs().sort_values().index[:limit]
    return players.loc[nearest, COMPARABLE_COLUMNS].to_dict("records")


# Serve the built React app from the same service (must be mounted last so the
# API routes above take priority).
if config.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
