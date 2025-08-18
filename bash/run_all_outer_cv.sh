#!/bin/bash

# Set the working directory to the python folder
cd /Users/jsevere2/Documents/AML_PhD/leukem_ai

# Activate the Python virtual environment
source .venv/bin/activate

echo "Starting all outer CV experiments..."
echo "=================================="

# Function to run outer CV for a specific model and strategy
run_outer_cv() {
    local model_type=$1
    local multi_type=$2
    local fold_type=$3
    local best_params_file=$4

    echo "Running outer CV for: $model_type - $multi_type - $fold_type"
    echo "Using best params from: $best_params_file"
    echo "----------------------------------------"

    python python/run_outer_cv.py \
        --model_type "$model_type" \
        --multi_type "$multi_type" \
        --fold_type "$fold_type" \
        --best_params_path "$best_params_file"

    echo "Completed: $model_type - $multi_type - $fold_type"
    echo ""
}

# SVM experiments
# echo "Running SVM outer CV experiments..."
# run_outer_cv "SVM" "OvR" "CV" "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/SVM_15aug/SVM_best_param_cv.csv"
# run_outer_cv "SVM" "OvR" "loso" "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/SVM_15aug/SVM_best_param_loso.csv"

# # XGBOOST experiments
# echo "Running XGBOOST outer CV experiments..."
# run_outer_cv "XGBOOST" "OvR" "CV" "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/XGBOOST_15aug/XGBOOST_best_param_cv.csv"
# run_outer_cv "XGBOOST" "OvR" "loso" "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/XGBOOST_15aug/XGBOOST_best_param_loso.csv"

echo "Running NN outer CV experiments..."
run_outer_cv "NN" "standard" "CV" "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/NN_15aug/NEURAL_NET_best_param_cv.csv"
run_outer_cv "NN" "standard" "loso" "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/NN_15aug/NEURAL_NET_best_param_loso.csv"

echo "All outer CV experiments completed!"
echo "=================================="

# Deactivate the virtual environment
deactivate