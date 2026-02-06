#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/venv"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
INIT_DB_SCRIPT="$ROOT_DIR/scripts/init_db.py"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: Python is not installed or not on PATH. Please install Python 3.9+." >&2
  exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: requirements.txt not found at $REQUIREMENTS_FILE" >&2
  exit 1
fi

if [ ! -f "$INIT_DB_SCRIPT" ]; then
  echo "Error: init_db.py not found at $INIT_DB_SCRIPT" >&2
  exit 1
fi

if [ -d "$VENV_DIR" ]; then
  echo "Found existing virtual environment at $VENV_DIR"
else
  echo "Creating virtual environment in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing dependencies from requirements.txt"
python -m pip install --upgrade pip
python -m pip install -r "$REQUIREMENTS_FILE"

echo "Initializing database"
python "$INIT_DB_SCRIPT"

echo
echo "Setup complete. Next steps:"
echo "1) Activate the virtual environment: source venv/bin/activate"
echo "2) Run the app: streamlit run app.py"
