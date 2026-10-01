import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from transfer_value import api, model

PLAYER = {
    "age": 24.5,
    "position": "Attack",
    "sub_position": "Centre-Forward",
    "league": "GB1",
    "appearances": 34,
    "minutes_played": 2900,
    "goals": 18,
    "assists": 6,
}


@pytest.fixture
def client(modelling_frame, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    model.save(model.fit_final(modelling_frame, "Random Forest", n_estimators=10), path)
    monkeypatch.setenv("TVP_MODEL_PATH", str(path))
    api.get_artifact.cache_clear()
    yield TestClient(api.app)
    api.get_artifact.cache_clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_a_positive_value(client):
    response = client.post("/predict", json=PLAYER)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_value_eur"] > 0
    assert body["model"] == "Random Forest"


def test_predict_rejects_invalid_input(client):
    assert client.post("/predict", json=dict(PLAYER, position="Striker")).status_code == 422
    assert client.post("/predict", json=dict(PLAYER, age=7)).status_code == 422
    missing = {k: v for k, v in PLAYER.items() if k != "minutes_played"}
    assert client.post("/predict", json=missing).status_code == 422


def test_predict_without_a_trained_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("TVP_MODEL_PATH", str(tmp_path / "missing.joblib"))
    api.get_artifact.cache_clear()
    response = TestClient(api.app).post("/predict", json=PLAYER)
    api.get_artifact.cache_clear()
    assert response.status_code == 503


def test_metadata_lists_form_options(client):
    body = client.get("/metadata").json()
    assert body["model"] == "Random Forest"
    assert body["leagues"]["GB1"] == "Premier League"
    assert "Centre-Forward" in body["positions"]["Attack"]


def test_comparables_returns_nearest_players(tmp_path, monkeypatch):
    rows = [
        "name,league,position,age,goals,assists,market_value_in_eur",
        "Cheap Keeper,GB1,Goalkeeper,30.0,0,0,1000000",
        "Mid Winger,ES1,Attack,24.0,8,6,30000000",
        "Star Striker,GB1,Attack,25.0,25,7,150000000",
    ]
    (tmp_path / "test_predictions.csv").write_text("\n".join(rows))
    monkeypatch.setenv("TVP_OUTPUT_DIR", str(tmp_path))
    api.get_reference_players.cache_clear()
    client = TestClient(api.app)

    nearest = client.get("/comparables", params={"value": 28e6, "limit": 2}).json()
    assert [p["name"] for p in nearest] == ["Mid Winger", "Cheap Keeper"]

    attackers = client.get("/comparables", params={"value": 1e6, "position": "Attack"}).json()
    assert [p["name"] for p in attackers] == ["Mid Winger", "Star Striker"]

    assert client.get("/comparables", params={"value": -5}).status_code == 422
    api.get_reference_players.cache_clear()
