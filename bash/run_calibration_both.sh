#!/bin/bash
# Run both calibration tracks: selection-LOSO (existing) and Option B deploy-loso.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Track A: selection-LOSO calibration (existing)"
echo "=============================================="
echo "Requires: R/train_test_analysis.R + left-out augmentation already completed."
Rscript R/calibration_reject_models_final.R

echo ""
echo "=============================================="
echo "Track B: Option B deploy-loso calibration"
echo "=============================================="
bash bash/run_deploy_calibration_loso.sh
Rscript R/build_deploy_loso_fold_matrices.R
Rscript R/calibration_reject_models_deploy_b.R

echo ""
echo "Both calibration tracks complete."
echo "Compare external predictions with:"
echo "  python python/predict_new_samples.py --cutoff_source both --input_file ... --output_dir ..."
