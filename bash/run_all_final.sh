#!/bin/bash

# Set the working directory to the python folder
cd /Users/jsevere2/leukem_ai

# Activate the Python virtual environment
source .venv/bin/activate

echo "Starting all final model training experiments..."
echo "=================================="

# Function to run outer CV for a specific model and strategy
run_final_model_train() {
    local model_type=$1
    local multi_type=$2
    local fold_type=$3
    local best_params_file=$4

    echo "Running final model training for: $model_type - $multi_type - $fold_type"
    echo "Using best params from: $best_params_file"
    echo "----------------------------------------"

    python python/run_final_train.py \
        --model_type "$model_type" \
        --multi_type "$multi_type" \
        --fold_type "$fold_type" \
        --best_params_path "$best_params_file"

    echo "Completed: $model_type - $multi_type - $fold_type"
    echo ""
}

# SVM experiments
# echo "Running SVM..."
run_final_model_train "SVM" "OvR" "CV" "data/out/final_train_test/best_params/SVM/SVM_best_param_cv.csv"
# run_final_model_train "SVM" "OvR" "loso" "/Users/jsevere2/Documents/AML_PhD/leukem_ai/data/out/final_train_test/best_params/SVM/SVM_best_param_loso.csv"

# # XGBOOST experiments
# echo "Running XGBOOST..."
run_final_model_train "XGBOOST" "OvR" "CV" "data/out/final_train_test/best_params/XGBOOST/XGBOOST_best_param_cv.csv"
# run_final_model_train "XGBOOST" "OvR" "loso" "/Users/jsevere2/Documents/AML_PhD/leukem_ai/data/out/final_train_test/best_params/XGBOOST/XGBOOST_best_param_loso.csv"

echo "Running NN..."
run_final_model_train "NN" "standard" "CV" "data/out/final_train_test/best_params/NN/NEURAL_NET_best_param_cv.csv"
# run_final_model_train "NN" "standard" "loso" "/Users/jsevere2/Documents/AML_PhD/leukem_ai/data/out/final_train_test/best_params/NN/NEURAL_NET_best_param_loso.csv"

echo "All final model training experiments completed!"
echo "=================================="

# Deactivate the virtual environment
deactivate