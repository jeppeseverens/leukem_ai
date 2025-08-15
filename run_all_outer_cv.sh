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
# run_outer_cv "SVM" "standard" "CV" "inner_cv_best_params/SVM/SVM_best_param_standard.csv"
# run_outer_cv "SVM" "OvR" "CV" "inner_cv_best_params_n10/SVM/SVM_best_param_OvR.csv"
# run_outer_cv "SVM" "OvO" "CV" "inner_cv_best_params/SVM/SVM_best_param_OvO.csv"
# run_outer_cv "SVM" "standard" "loso" "inner_cv_best_params/SVM/SVM_best_param_standard_loso.csv"
# run_outer_cv "SVM" "OvR" "loso" "inner_cv_best_params_n10/SVM/SVM_best_param_OvR_loso.csv"
# run_outer_cv "SVM" "OvO" "loso" "inner_cv_best_params/SVM/SVM_best_param_OvO_loso.csv"

# XGBOOST experiments
# echo "Running XGBOOST outer CV experiments..."
# run_outer_cv "XGBOOST" "standard" "CV" "inner_cv_best_params/XGBOOST/XGBOOST_best_param_standard.csv"
# run_outer_cv "XGBOOST" "OvR" "CV" "inner_cv_best_params_n10/XGBOOST/XGBOOST_best_param_OvR.csv"
# run_outer_cv "XGBOOST" "OvO" "CV" "inner_cv_best_params/XGBOOST/XGBOOST_best_param_OvO.csv"
# run_outer_cv "XGBOOST" "standard" "loso" "inner_cv_best_params/XGBOOST/XGBOOST_best_param_standard_loso.csv"
# run_outer_cv "XGBOOST" "OvR" "loso" "inner_cv_best_params_n10/XGBOOST/XGBOOST_best_param_OvR_loso.csv"
# run_outer_cv "XGBOOST" "OvO" "loso" "inner_cv_best_params/XGBOOST/XGBOOST_best_param_OvO_loso.csv"

echo "Running NN outer CV experiments..."
run_outer_cv "NN" "standard" "CV" "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/NN/NEURAL_NET_best_param_cv.csv"
run_outer_cv "NN" "standard" "loso" "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/NN/NEURAL_NET_best_param_loso.csv"

echo "All outer CV experiments completed!"
echo "=================================="

# Deactivate the virtual environment
deactivate 