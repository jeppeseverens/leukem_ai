#!/bin/bash

set -euo pipefail

# Backfill KNN reject-feature columns into all existing outer-CV CSV outputs
# (SVM/XGBOOST/NN, CV + LOSO, with and without left-out variants) without
# rerunning model inference.

PROJECT_ROOT="/Users/jsevere2/leukem_ai"
PYTHON_BIN="python"
FS_METHOD="${FS_METHOD:-eta2}"
KNN_N_GENES="${KNN_N_GENES:-500}"
CACHE_DIR="${CACHE_DIR:-}"
# Default to in-place replacement so downstream auto-discovery in R picks KNN-complete files.
OVERWRITE="${OVERWRITE:-1}"

cd "${PROJECT_ROOT}"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

latest_match() {
  # Return lexicographically latest existing file for the given glob pattern.
  # Timestamped filenames use YYYYMMDD_HHMM so lexicographic ordering is stable.
  local pattern="$1"
  local latest=""
  shopt -s nullglob
  for f in ${pattern}; do
    if [ -z "${latest}" ] || [[ "${f}" > "${latest}" ]]; then
      latest="${f}"
    fi
  done
  shopt -u nullglob
  printf '%s' "${latest}"
}

run_backfill() {
  local csv_path="$1"
  local fold_type="$2"

  if [ -z "${csv_path}" ]; then
    return 0
  fi

  echo "Backfilling: ${csv_path}"
  local cmd=(
    "${PYTHON_BIN}" "python/backfill_outer_cv_knn.py"
    --outer_csv "${csv_path}"
    --fold_type "${fold_type}"
    --fs_method "${FS_METHOD}"
    --knn_n_genes "${KNN_N_GENES}"
  )

  if [ -n "${CACHE_DIR}" ]; then
    cmd+=(--cache_dir "${CACHE_DIR}")
  fi
  if [ "${OVERWRITE}" = "1" ]; then
    cmd+=(--overwrite)
  fi

  "${cmd[@]}"
}

process_model() {
  local model_dir="$1"
  local model_prefix="$2"
  local multi="$3"

  local base_dir="data/out/outer_cv/${model_dir}"
  if [ ! -d "${base_dir}" ]; then
    echo "Skipping ${model_dir}: directory not found (${base_dir})"
    return 0
  fi

  local cv_file
  local loso_file
  local cv_leftout_file
  local loso_leftout_file

  cv_file="$(latest_match "${base_dir}/${model_prefix}_outer_cv_CV_${multi}_fs_eta_*.csv")"
  loso_file="$(latest_match "${base_dir}/${model_prefix}_outer_cv_loso_${multi}_fs_eta_*.csv")"
  cv_leftout_file="$(latest_match "${base_dir}/${model_prefix}_outer_cv_CV_${multi}_leftout_fs_eta_*.csv")"
  loso_leftout_file="$(latest_match "${base_dir}/${model_prefix}_outer_cv_loso_${multi}_leftout_fs_eta_*.csv")"

  if [ -n "${cv_file}" ]; then
    run_backfill "${cv_file}" "CV"
  else
    echo "No CV file found for ${model_dir}"
  fi
  if [ -n "${loso_file}" ]; then
    run_backfill "${loso_file}" "loso"
  else
    echo "No LOSO file found for ${model_dir}"
  fi
  if [ -n "${cv_leftout_file}" ]; then
    run_backfill "${cv_leftout_file}" "CV"
  else
    echo "No CV left-out file found for ${model_dir}"
  fi
  if [ -n "${loso_leftout_file}" ]; then
    run_backfill "${loso_leftout_file}" "loso"
  else
    echo "No LOSO left-out file found for ${model_dir}"
  fi
}

echo "Starting KNN backfill for all outer-CV files..."
echo "FS_METHOD=${FS_METHOD}, KNN_N_GENES=${KNN_N_GENES}, OVERWRITE=${OVERWRITE}"

process_model "SVM_n10_fs_eta" "SVM" "OvR"
process_model "XGBOOST_n10_fs_eta" "XGBOOST" "OvR"
process_model "NN_n10_fs_eta" "NN" "standard"

echo "Completed KNN backfill for discovered files."

if [ -f ".venv/bin/activate" ]; then
  deactivate || true
fi
