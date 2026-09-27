#!/usr/bin/env bash
set -euo pipefail

# Run the AECS-SDC unit test suite.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HOME}/venvs/aecs-supervisor"

if [[ ! -d "${VENV}" ]]; then
    echo "Virtual environment not found at ${VENV}."
    exit 1
fi

source "${VENV}/bin/activate"
cd "${REPO_ROOT}"

python -m pytest tests/ -q "$@"
