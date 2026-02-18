import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.init_db import create_tables
from src.database.dal import DataAccessLayer


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["JOBLIB_MULTIPROCESSING"] = "0"
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result


def _seed_price_history(dal: DataAccessLayer, ticker: str, rows: int = 1200, phase: float = 0.0) -> None:
    dates = pd.date_range("2019-01-01", periods=rows, freq="D")
    trend = np.linspace(100.0, 160.0, rows)
    seasonal = 2.0 * np.sin((np.arange(rows) / 15.0) + phase)
    close = trend + seasonal
    records = []
    for i, dt in enumerate(dates):
        c = float(close[i])
        records.append(
            {
                "ticker": ticker,
                "date": dt.strftime("%Y-%m-%d"),
                "open": c - 0.3,
                "high": c + 0.8,
                "low": c - 0.9,
                "close": c,
                "volume": int(100000 + (i % 5000)),
                "adj_close": c,
            }
        )
    dal.bulk_insert_prices(records)


def _find_single(paths: list[Path], label: str) -> Path:
    assert paths, f"No {label} artifact found."
    return sorted(paths)[-1]


@pytest.mark.integration
def test_model_scripts_smoke_db_mode(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[2]
    db_path = tmp_path / "smoke.db"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    docs_dir = tmp_path / "docs"
    config_dir = tmp_path / "config"
    ticker = "TEST"
    ticker_2 = "ALT"

    conn = sqlite3.connect(db_path)
    create_tables(conn)
    conn.close()

    dal = DataAccessLayer(db_path=db_path)
    dal.insert_ticker(ticker, "Smoke Test Corp")
    dal.insert_ticker(ticker_2, "Alt Smoke Test Corp")
    _seed_price_history(dal, ticker=ticker)
    _seed_price_history(dal, ticker=ticker_2, phase=0.9)

    _run(
        [
            sys.executable,
            "scripts/train_random_forest.py",
            "--no-tune",
            "--output-dir",
            str(models_dir),
            "--data-source",
            "db",
            "--db-path",
            str(db_path),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2023-12-31",
        ],
        cwd=project_root,
    )
    rf_model = models_dir / "random_forest_global.pkl"
    assert rf_model.exists()
    rf_manifest = rf_model.with_suffix(".manifest.json")
    assert rf_manifest.exists()

    _run(
        [
            sys.executable,
            "scripts/train_xgboost.py",
            "--no-tune",
            "--output-dir",
            str(models_dir),
            "--data-source",
            "db",
            "--db-path",
            str(db_path),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2023-12-31",
        ],
        cwd=project_root,
    )
    xgb_model = models_dir / "xgboost_global.json"
    assert xgb_model.exists()
    xgb_manifest = xgb_model.with_suffix(".manifest.json")
    assert xgb_manifest.exists()

    _run(
        [
            sys.executable,
            "scripts/train_lstm.py",
            "--ticker",
            ticker,
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--sequence-length",
            "20",
            "--output-dir",
            str(models_dir),
            "--data-source",
            "db",
            "--db-path",
            str(db_path),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2023-12-31",
        ],
        cwd=project_root,
    )
    lstm_model = _find_single(list(models_dir.glob(f"lstm_{ticker.lower()}_*.keras")), "lstm model")
    lstm_manifest = lstm_model.with_suffix(".manifest.json")
    lstm_scalers = Path(str(lstm_model).replace(".keras", "_scalers.joblib"))
    assert lstm_manifest.exists()
    assert lstm_scalers.exists()

    _run(
        [
            sys.executable,
            "scripts/run_walkforward.py",
            "--ticker",
            ticker,
            "--model",
            "rf",
            "--train-years",
            "1",
            "--val-months",
            "1",
            "--test-months",
            "1",
            "--step-months",
            "3",
            "--output-dir",
            str(results_dir),
            "--data-source",
            "db",
            "--db-path",
            str(db_path),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2023-12-31",
        ],
        cwd=project_root,
    )
    wf_json = results_dir / f"wf_results_{ticker}_rf.json"
    assert wf_json.exists()

    for model_path, scaler_arg in [
        (rf_model, []),
        (xgb_model, []),
        (lstm_model, ["--scaler", str(lstm_scalers)]),
    ]:
        _run(
            [
                sys.executable,
                "scripts/backtest_models.py",
                "--ticker",
                ticker,
                "--model",
                str(model_path),
                "--start-date",
                "2022-01-01",
                "--end-date",
                "2022-12-31",
                "--output-dir",
                str(results_dir),
                "--data-source",
                "db",
                "--db-path",
                str(db_path),
                *scaler_arg,
            ],
            cwd=project_root,
        )
    assert (results_dir / "backtest_TEST_2022.csv").exists()

    comparison_csv = results_dir / "comparison_summary.csv"
    comparison_md = docs_dir / "model_selection.md"
    production_cfg = config_dir / "production_model.json"
    _run(
        [
            sys.executable,
            "scripts/compare_models.py",
            "--results_dir",
            str(results_dir),
            "--output_report",
            str(comparison_md),
            "--output_csv",
            str(comparison_csv),
            "--production_config",
            str(production_cfg),
        ],
        cwd=project_root,
    )
    assert comparison_csv.exists()
    assert comparison_md.exists()
    assert production_cfg.exists()
