#!/bin/bash
set -euo pipefail

# Run MLL lab predictions for all calibration combinations:
# - calibration methods:   univariate, multivariate
# - calibration settings:  known_only, known_only_logit, ood_aware, ood_aware_logit, two_head, two_head_postcal
# - ensemble methods:      product, weighted
#
# Defaults:
# - input file: data/MLL_lab/STAR_AML_MLLlab.csv
# - output base: data/out/final_models/predictions_mll_lab_4way
# - max accepted risk: 5 (%)
# - cutoff curve split: cv (options: cv, loso, both)
#
# Usage:
#   bash bash/run_mll_lab_4way.sh [INPUT_CSV] [OUTPUT_BASE_DIR] [MAX_ACCEPTED_RISK_PCT] [CUTOFF_CURVE_SPLIT]

PROJECT_ROOT="/Users/jsevere2/leukem_ai"
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
fi

INPUT_FILE="${1:-$PROJECT_ROOT/data/MLL_lab/STAR_AML_MLLlab.csv}"
OUTPUT_BASE_DIR="${2:-$PROJECT_ROOT/data/out/final_models/predictions_mll_lab_4way}"
MAX_RISK_PCT="${3:-5}"
CUTOFF_CURVE_SPLIT_MODE="${4:-cv}"

if [ ! -f "$INPUT_FILE" ]; then
  echo "ERROR: Input file not found: $INPUT_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_BASE_DIR"

run_combo() {
  local calibration_method="$1"
  local calibration_setting="$2"
  local ensemble_method="$3"
  local cutoff_curve_split="$4"
  local combo_id="${calibration_method}__${calibration_setting}__${ensemble_method}__${cutoff_curve_split}"
  local combo_out_dir="${OUTPUT_BASE_DIR}/${combo_id}"

  echo "============================================================"
  echo "Running combo: ${combo_id}"
  echo "Input: ${INPUT_FILE}"
  echo "Output: ${combo_out_dir}"
  echo "Max accepted risk (%): ${MAX_RISK_PCT}"
  echo "Cutoff curve split: ${cutoff_curve_split}"
  echo "============================================================"

  python python/predict_new_samples.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$combo_out_dir" \
    --merged_only \
    --calibration_method "$calibration_method" \
    --calibration_setting "$calibration_setting" \
    --ensemble_method "$ensemble_method" \
    --max_accepted_risk_pct "$MAX_RISK_PCT" \
    --cutoff_curve_split "$cutoff_curve_split"

  echo "Completed combo: ${combo_id}"
  echo
}

SPLIT_LIST=()
case "$CUTOFF_CURVE_SPLIT_MODE" in
  cv|loso)
    SPLIT_LIST=("$CUTOFF_CURVE_SPLIT_MODE")
    ;;
  both)
    SPLIT_LIST=("cv" "loso")
    ;;
  *)
    echo "ERROR: Invalid CUTOFF_CURVE_SPLIT: $CUTOFF_CURVE_SPLIT_MODE (use: cv, loso, or both)"
    exit 1
    ;;
esac

for method in "univariate" "multivariate"; do
  for setting in "known_only" "known_only_logit" "ood_aware" "ood_aware_logit" "two_head" "two_head_postcal"; do
    for ensemble in "product" "weighted"; do
      for split in "${SPLIT_LIST[@]}"; do
        run_combo "$method" "$setting" "$ensemble" "$split"
      done
    done
  done
done

echo "All MLL lab prediction combos completed."

if command -v deactivate >/dev/null 2>&1; then
  deactivate || true
fi
