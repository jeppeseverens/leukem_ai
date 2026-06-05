#!/bin/bash

# Set the working directory to the python folder
cd /Users/jsevere2/leukem_ai

# Activate the Python virtual environment
source .venv/bin/activate

# Remove cached pipelines so they are rebuilt from current data (get_or_create_pipeline
# does not check if data changed, so stale pipelines would otherwise be reused)
PIPELINES_DIR="data/out/final_models/pipelines"
if [ -d "$PIPELINES_DIR" ]; then
  echo "Removing cached pipelines in $PIPELINES_DIR so they are rebuilt from current data."
  rm -rf "$PIPELINES_DIR"
fi

echo "Starting all final model training experiments (LOSO)..."
echo "=================================="

# Feature selection method for final models:
#   - "mad"  : intersecting MVGs (default)
#   - "eta2" : eta2_subtype - eta2_study
FS_METHOD="eta2"

# Function to run final model training for a specific model and strategy
run_final_model_train() {
    local model_type=$1
    local multi_type=$2
    local best_params_file=$3

    echo "Running final model training for: $model_type - $multi_type - loso"
    echo "Using best params from: $best_params_file"
    echo "----------------------------------------"

    python python/run_final_train.py \
        --model_type "$model_type" \
        --multi_type "$multi_type" \
        --fold_type "loso" \
        --best_params_path "$best_params_file" \
        --fs_method "$FS_METHOD" \
        --include_leftout

    echo "Completed: $model_type - $multi_type - loso"
    echo ""
}

run_final_model_train "SVM" "OvR" "data/out/final_train_test/best_params/SVM/SVM_best_param_loso.csv"
run_final_model_train "XGBOOST" "OvR" "data/out/final_train_test/best_params/XGBOOST/XGBOOST_best_param_loso.csv"
run_final_model_train "NN" "standard" "data/out/final_train_test/best_params/NN/NEURAL_NET_best_param_loso.csv"

echo "All final model training experiments completed!"
echo "=================================="

# Deactivate the virtual environment
deactivate
