#!/usr/bin/env bash
set -euo pipefail

# Run the AECS-SDC offline benchmark.
# Usage: scripts/run_benchmark.sh [subset-size]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HOME}/venvs/aecs-supervisor"

if [[ ! -d "${VENV}" ]]; then
    echo "Virtual environment not found at ${VENV}."
    echo "Create it first: python3 -m venv ${VENV}"
    exit 1
fi

source "${VENV}/bin/activate"
cd "${REPO_ROOT}"

SUBSET_FLAG=""
if [[ $# -ge 1 ]]; then
    SUBSET_FLAG="--subset-size $1"
fi

python benchmark.py --config configs/benchmark.yaml ${SUBSET_FLAG}
