#!/bin/bash

# Set the working directory to the python folder
cd /Users/jsevere2/leukem_ai

# Activate the Python virtual environment
source .venv/bin/activate

echo "Starting all outer CV experiments (with left-out predictions)..."
echo "=================================="

# Feature selection method for outer CV:
#   - "mad"  : intersecting MVGs (default)
#   - "eta2" : eta2_subtype - eta2_study
FS_METHOD="eta2"

# Function to run outer CV for a specific model and strategy
run_outer_cv() {
    local model_type=$1
    local multi_type=$2
    local fold_type=$3
    local best_params_file=$4

    echo "Running outer CV for: $model_type - $multi_type - $fold_type"
    echo "Using best params from: $best_params_file"
    echo "----------------------------------------"

    python -u python/run_outer_cv.py \
        --model_type "$model_type" \
        --multi_type "$multi_type" \
        --fold_type "$fold_type" \
        --best_params_path "$best_params_file" \
        --fs_method "$FS_METHOD" \
        --include_leftout

    echo "Completed: $model_type - $multi_type - $fold_type"
    echo ""
}

# SVM experiments
run_outer_cv "SVM" "OvR" "CV" "data/out/inner_cv/inner_cv_best_params/SVM_10feb26/SVM_best_param_cv.csv"
run_outer_cv "SVM" "OvR" "loso" "data/out/inner_cv/inner_cv_best_params/SVM_10feb26/SVM_best_param_loso.csv"

# XGBOOST experiments
run_outer_cv "XGBOOST" "OvR" "CV" "data/out/inner_cv/inner_cv_best_params/XGBOOST_10feb26/XGBOOST_best_param_cv.csv"
run_outer_cv "XGBOOST" "OvR" "loso" "data/out/inner_cv/inner_cv_best_params/XGBOOST_10feb26/XGBOOST_best_param_loso.csv"

# NN experiments
run_outer_cv "NN" "standard" "CV" "data/out/inner_cv/inner_cv_best_params/NN_10feb26/NEURAL_NET_best_param_cv.csv"
run_outer_cv "NN" "standard" "loso" "data/out/inner_cv/inner_cv_best_params/NN_10feb26/NEURAL_NET_best_param_loso.csv"

echo "All outer CV experiments completed!"
echo "=================================="

# Deactivate the virtual environment
deactivate