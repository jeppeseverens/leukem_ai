#!/bin/bash
# Option B: score 6-study final models on each held-out study (known cohort only).

set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONPATH="${PWD}/python${PYTHONPATH:+:${PYTHONPATH}}"

FS_METHOD="eta2"
echo "Running deploy-loso calibration scoring (fs_method=${FS_METHOD})..."
python python/run_deploy_calibration_loso.py --fs_method "${FS_METHOD}" --model_type SVM

deactivate
echo "Deploy-loso Python scoring done. Next: Rscript R/build_deploy_loso_fold_matrices.R"
