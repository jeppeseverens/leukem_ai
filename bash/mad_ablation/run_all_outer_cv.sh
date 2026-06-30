#!/bin/bash

# MAD-global ablation: outer CV using best params from inner_cv_analysis_mad_global.R

cd /Users/jsevere2/leukem_ai

source .venv/bin/activate

echo "Starting MAD-global ablation outer CV experiments (no KNN, no left-out predictions)..."
echo "=================================="

FS_METHOD="mad_global"

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
        --skip_knn

    echo "Completed: $model_type - $multi_type - $fold_type"
    echo ""
}

run_outer_cv "SVM" "OvR" "CV" "data/out/inner_cv/inner_cv_best_params/SVM_mad_global/SVM_best_param_cv.csv"
run_outer_cv "SVM" "OvR" "loso" "data/out/inner_cv/inner_cv_best_params/SVM_mad_global/SVM_best_param_loso.csv"

run_outer_cv "XGBOOST" "OvR" "CV" "data/out/inner_cv/inner_cv_best_params/XGBOOST_mad_global/XGBOOST_best_param_cv.csv"
run_outer_cv "XGBOOST" "OvR" "loso" "data/out/inner_cv/inner_cv_best_params/XGBOOST_mad_global/XGBOOST_best_param_loso.csv"

run_outer_cv "NN" "standard" "CV" "data/out/inner_cv/inner_cv_best_params/NN_mad_global/NEURAL_NET_best_param_cv.csv"
run_outer_cv "NN" "standard" "loso" "data/out/inner_cv/inner_cv_best_params/NN_mad_global/NEURAL_NET_best_param_loso.csv"

echo "All MAD-global outer CV experiments completed!"
echo "=================================="

deactivate
