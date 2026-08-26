#!/usr/bin/env bash
set -euo pipefail

# Workstation launcher for the fair SynDE/fingerprint/Chemprop comparison.
# Override any setting as an environment variable, for example:
#   SYNDE_ACCELERATOR=cpu SYNDE_NUM_WORKERS=0 bash Experiment/run_global_comparators.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${SYNDE_PYTHON:-python}"
WORK_DIR="${SYNDE_COMPARATOR_WORK_DIR:-/tmp/synde-external-global-comparators-v1}"
BOOTSTRAPS="${SYNDE_BOOTSTRAPS:-2000}"
EPOCHS="${SYNDE_CHEMPROP_EPOCHS:-30}"
PATIENCE="${SYNDE_CHEMPROP_PATIENCE:-8}"
BATCH_SIZE="${SYNDE_BATCH_SIZE:-128}"
NUM_WORKERS="${SYNDE_NUM_WORKERS:-4}"
ACCELERATOR="${SYNDE_ACCELERATOR:-gpu}"
DEVICES="${SYNDE_DEVICES:-1}"
TRAINING_SOURCE="${SYNDE_TRAINING_SOURCE:-data/ord_training_xtb.csv}"
EXTERNAL_SOURCE="${SYNDE_EXTERNAL_SOURCE:-data/ord_test_xtb.csv}"
CONNECTIVITY_MODEL="${SYNDE_CONNECTIVITY_MODEL:-synde/models/synde_frozen_model.json}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  printf 'ERROR: Python executable not found: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi

missing=0
for input in "${TRAINING_SOURCE}" "${EXTERNAL_SOURCE}" "${CONNECTIVITY_MODEL}" Experiment/calibration_seed.json; do
  if [[ ! -f "${input}" ]]; then
    printf 'ERROR: missing required input: %s\n' "${input}" >&2
    missing=1
  fi
done
if (( missing )); then
  printf 'Generate missing xTB label CSVs with: python Experiment/scripts/03_run_xtb.py\n' >&2
  exit 2
fi

mkdir -p "${WORK_DIR}"

echo "[1/4] Building/reusing the training and external-validation caches"
"${PYTHON_BIN}" Experiment/scripts/07_train_energy.py \
  --training-source "${TRAINING_SOURCE}" \
  --external-source "${EXTERNAL_SOURCE}" \
  --connectivity-model "${CONNECTIVITY_MODEL}" \
  --work-dir "${WORK_DIR}" \
  --rebuild-training-cache \
  --rebuild-external-cache \
  --bootstrap-replicates "${BOOTSTRAPS}"

echo "[2/4] Training ECFP4, RDK5, MACCS, and five-fold Chemprop comparators"
"${PYTHON_BIN}" Experiment/scripts/08_compare_models.py \
  --training-cache "${WORK_DIR}/training.joblib" \
  --external-cache "${WORK_DIR}/external.joblib" \
  --synde-result "${WORK_DIR}/synde.json" \
  --work-dir "${WORK_DIR}" \
  --output "${WORK_DIR}/benchmark.json" \
  --models fingerprints chemprop \
  --bootstrap-replicates "${BOOTSTRAPS}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --accelerator "${ACCELERATOR}" \
  --devices "${DEVICES}"

echo "[3/4] Validating the completed work-directory benchmark"
"${PYTHON_BIN}" Experiment/scripts/13_validate.py \
  --synde-result "${WORK_DIR}/synde.json" \
  --comparator-result "${WORK_DIR}/benchmark.json" \
  --artifact-dir "${WORK_DIR}/artifacts" \
  --energy-model "${WORK_DIR}/synde_energy_model.json"

echo "[4/4] Finished"
echo "Full record: ${WORK_DIR}/benchmark.json"
echo "SynDE record: ${WORK_DIR}/synde.json"
echo "Global table: ${WORK_DIR}/tables/global_metrics.csv"
echo "Local table: ${WORK_DIR}/tables/local_metrics.csv"
