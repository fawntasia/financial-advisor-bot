import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATHS = [
    "scripts/train_lstm.py",
    "scripts/train_random_forest.py",
    "scripts/train_xgboost.py",
    "scripts/run_walkforward.py",
    "scripts/backtest_models.py",
    "scripts/compare_models.py",
]


@pytest.mark.unit
@pytest.mark.parametrize("script_relpath", SCRIPT_PATHS)
def test_script_help_exits_zero(script_relpath: str):
    script_path = PROJECT_ROOT / script_relpath
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"--help failed for {script_relpath}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
