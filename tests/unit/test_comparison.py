import sys
from unittest.mock import mock_open, patch

import numpy as np
import pytest

sys.path.insert(0, ".")

from src.models.comparison import EnsembleVoting, ModelComparison


@pytest.mark.unit
def test_select_best_model_by_sharpe():
    comparison = ModelComparison()
    comparison.results = {
        "AAPL_lstm": [
            {"metrics": {"test_accuracy": 0.6, "sharpe_ratio": 1.2, "max_drawdown": -0.2}},
            {"metrics": {"test_accuracy": 0.7, "sharpe_ratio": 0.8, "max_drawdown": -0.1}},
        ],
        "AAPL_rf": [
            {"metrics": {"test_accuracy": 0.55, "sharpe_ratio": 1.5, "max_drawdown": -0.3}},
            {"metrics": {"test_accuracy": 0.65, "sharpe_ratio": 1.1, "max_drawdown": -0.25}},
        ],
    }

    best = comparison.select_best_model(metric="avg_sharpe")

    assert best["ticker"] == "AAPL"
    assert best["model"] == "rf"
    assert best["avg_sharpe"] == pytest.approx(1.3)
    assert best["avg_accuracy"] == pytest.approx(0.6)
    assert best["num_steps"] == 2


@pytest.mark.unit
def test_select_best_model_by_max_drawdown_least_negative():
    comparison = ModelComparison()
    comparison.results = {
        "MSFT_lstm": [
            {"metrics": {"test_accuracy": 0.6, "sharpe_ratio": 0.9, "max_drawdown": -0.3}},
            {"metrics": {"test_accuracy": 0.7, "sharpe_ratio": 1.0, "max_drawdown": -0.2}},
        ],
        "MSFT_rf": [
            {"metrics": {"test_accuracy": 0.55, "sharpe_ratio": 0.8, "max_drawdown": -0.1}},
            {"metrics": {"test_accuracy": 0.65, "sharpe_ratio": 0.7, "max_drawdown": -0.15}},
        ],
    }

    best = comparison.select_best_model(metric="avg_max_drawdown")

    assert best["ticker"] == "MSFT"
    assert best["model"] == "rf"
    assert best["avg_max_drawdown"] == pytest.approx(-0.125)


@pytest.mark.unit
def test_save_production_config_writes_expected_payload():
    comparison = ModelComparison()
    best_model = {
        "ticker": "AAPL",
        "model": "lstm",
        "avg_accuracy": 0.64,
        "avg_sharpe": 1.12,
        "avg_max_drawdown": -0.18,
    }

    fixed_timestamp = "2026-02-04T12:00:00"

    with patch("src.models.comparison.os.makedirs") as makedirs_mock:
        with patch("src.models.comparison.pd.Timestamp.now") as now_mock:
            now_mock.return_value.isoformat.return_value = fixed_timestamp
            with patch("src.models.comparison.open", mock_open(), create=True) as open_mock:
                with patch("src.models.comparison.json.dump") as dump_mock:
                    comparison.save_production_config(best_model, config_path="config/prod.json")

    makedirs_mock.assert_called_once()
    mk_args, mk_kwargs = makedirs_mock.call_args
    assert mk_kwargs == {"exist_ok": True}
    assert str(mk_args[0]).endswith("config")
    open_mock.assert_called_once_with("config/prod.json", "w")
    dump_args, _ = dump_mock.call_args
    written = dump_args[0]

    assert written["production_model"]["ticker"] == "AAPL"
    assert written["production_model"]["model_type"] == "lstm"
    assert written["production_model"]["performance"]["avg_sharpe"] == pytest.approx(1.12)
    assert written["production_model"]["updated_at"] == fixed_timestamp


@pytest.mark.unit
def test_ensemble_vote_majority_uses_vstack():
    predictions = [
        np.array([1, 0, 1, 0]),
        np.array([1, 1, 0, 0]),
        np.array([0, 1, 1, 0]),
    ]
    expected = np.array([1, 1, 1, 0])
    original_vstack = np.vstack

    with patch("src.models.comparison.np.vstack", side_effect=original_vstack) as vstack_mock:
        result = EnsembleVoting.vote(predictions)

    vstack_mock.assert_called_once()
    assert result.tolist() == expected.tolist()
