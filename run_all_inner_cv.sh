#!/bin/bash

set -e

function shutdown_on_error {
 sudo shutdown now -h
}

trap shutdown_on_error ERR

cd /home/ubuntu

# Update system packages
sudo apt update 

# Install required tools
sudo apt install python3 python3-venv python3-pip -y
sudo apt install -y unzip curl -y
sudo apt install git -y 

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Clone repository and download data
git clone https://github.com/jeppeseverens/leukem_ai.git
cd leukem_ai

aws s3 cp --recursive s3://jfseverens/leukem_ai/data/ ./data

# Install Python packages
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting all inner CV experiments..."
echo "=================================="

# # Function to run inner CV for a specific model and fold type
run_inner_cv() {
    local model_type=$1
    local fold_type=$2
    
    echo "Running inner CV for: $model_type - $fold_type"
    echo "----------------------------------------"
    
    python python/run_inner_cv.py \
        --model_type "$model_type" \
        --n_jobs 32 \
        --k_out 5 \
        --k_in 5 \
        --n_max_param 128 \
        --fold_type "$fold_type"
    
    echo "Completed: $model_type - $fold_type"
    
    # Sync results to S3 after each experiment
    echo "Syncing results to S3..."
    aws s3 sync /home/ubuntu/leukem_ai/out/ s3://jfseverens/inner_cv_results/
    echo "S3 sync completed for $model_type - $fold_type"
    echo ""
}

# Run all experiments
# echo "Running SVM experiments..."
# run_inner_cv "SVM" "CV"
# run_inner_cv "SVM" "loso"

# echo "Running XGBOOST experiments..."
# run_inner_cv "XGBOOST" "CV"
# run_inner_cv "XGBOOST" "loso"

# echo "Running NN experiments..."
# run_inner_cv "NN" "CV"
run_inner_cv "NN" "loso"

echo "All inner CV experiments completed!"
echo "=================================="

deactivate

# Final sync to S3 (in case any files were missed)
echo "Performing final S3 sync..."
aws s3 sync /home/ubuntu/leukem_ai/out/ s3://jfseverens/inner_cv_results/

sudo shutdown now -h