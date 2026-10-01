import numpy as np
import pytest

from transfer_value import config, model


def test_time_splits_never_train_on_the_future():
    seasons = np.array([2021, 2021, 2022, 2023, 2023])
    splits = list(model.time_splits(seasons))
    assert [season for _, _, season in splits] == [2022, 2023]
    for train_mask, test_mask, season in splits:
        assert seasons[train_mask].max() < season
        assert (seasons[test_mask] == season).all()
        assert not (train_mask & test_mask).any()


def test_score_is_perfect_for_exact_predictions():
    values = np.array([1e6, 5e6, 20e6])
    metrics = model.score(values, values)
    assert metrics["mae_eur"] == 0
    assert metrics["r2"] == 1


def test_compare_models_beats_the_baseline(modelling_frame):
    comparison, best, predicted = model.compare_models(modelling_frame, n_estimators=30)

    assert set(comparison["model"]) == set(model.candidate_models())
    assert best != model.BASELINE
    mae = comparison.set_index("model")["holdout_mae_eur"]
    assert mae[best] < mae[model.BASELINE]

    holdout = modelling_frame[modelling_frame["season"] == 2024]
    assert list(predicted.index) == list(holdout.index)
    assert (predicted > 0).all()


def test_compare_models_needs_three_seasons(modelling_frame):
    two_seasons = modelling_frame[modelling_frame["season"] >= 2023]
    with pytest.raises(ValueError):
        model.compare_models(two_seasons, n_estimators=5)


def test_saved_model_round_trips(modelling_frame, tmp_path):
    artifact = model.fit_final(modelling_frame, "Random Forest", n_estimators=10)
    path = tmp_path / "model.joblib"
    model.save(artifact, path)
    loaded = model.load(path)

    raw = modelling_frame.drop(columns=[config.TARGET]).head(5)
    np.testing.assert_allclose(model.predict(loaded, raw), model.predict(artifact, raw))
    assert loaded["seasons"] == [2021, 2022, 2023, 2024]
