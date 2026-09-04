#!/usr/bin/env bash
set -euo pipefail

# One-time environment setup for the AECS-SDC benchmark.
# This installs the local package and dev dependencies.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HOME}/venvs/aecs-supervisor"

python3 -m venv "${VENV}"
source "${VENV}/bin/activate"

pip install --upgrade pip setuptools wheel
pip install -e "${REPO_ROOT}[dev]"

echo "Environment ready. Activate with: source ${VENV}/bin/activate"
