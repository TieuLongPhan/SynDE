#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
cd "${ROOT_DIR}"

if (( $# )); then
    paths=("$@")
else
    paths=(synde test Experiment/scripts scripts doc/conf.py)
fi

"${PYTHON_BIN}" scripts/check_python_file_size.py
"${PYTHON_BIN}" scripts/check_docstring_style.py
"${PYTHON_BIN}" -m flake8 "${paths[@]}" \
    --count \
    --max-complexity=13 \
    --max-line-length=120 \
    --extend-ignore=E203 \
    --per-file-ignores="\
__init__.py:F401,F403,\
Experiment/scripts/_helpers.py:C901,\
Experiment/scripts/09_diagnostics.py:E402,\
Experiment/scripts/_fit_energy.py:C901,\
Experiment/scripts/06_evaluate.py:C901,\
Experiment/scripts/01_prepare_ord.py:C901,\
Experiment/scripts/03_run_xtb.py:C901,\
Experiment/scripts/02_select_cohorts.py:C901,\
synde/energy/energy_predictor.py:C901,\
synde/energy/interpretable_two_d_v2.py:C901,\
synde/energy/interpretable_two_d_v3.py:C901,\
synde/energy/theory_energy.py:C901,\
synde/energy/truncated_scc_energy.py:C901,\
synde/geometry/xtb/xtb_minimize.py:C901" \
    --exclude="\
venv,\
.venv,\
__pycache__,\
.git,\
.pytest_cache,\
data/ord-data,\
docs,\
doc/_build" \
    --statistics
