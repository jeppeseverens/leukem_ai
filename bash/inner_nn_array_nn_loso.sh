#!/bin/bash
#SBATCH -J run_inner_nn_array_loso
#SBATCH --array=0-59              # 96 hyperparameter combinations (0-indexed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         # 1 core per hyperparameter combination
#SBATCH --time=16:00:00           # Shorter time per job
#SBATCH --error=job_output/loso/job_array_loso.%A_%a.err
#SBATCH --output=job_output/loso/job_array_loso.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=j.f.severens@lumc.nl
#SBATCH --mem=12G                  

cd /exports/me-lcco-aml-hpc/Jeppe2/leukem_ai

# Create job_output directory if it doesn't exist
mkdir -p job_output/loso

source venv/bin/activate

export TF_CPP_MIN_LOG_LEVEL=2

echo "Starting inner CV experiment (LOSO fold type) on SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Processing hyperparameter combination $SLURM_ARRAY_TASK_ID out of 60"
echo "=================================="

# Define your arguments for each array job
MODEL_TYPE="NN"

# Run LOSO fold type
python python/run_inner_cv_array.py \
    --model_type "$MODEL_TYPE" \
    --param_index $SLURM_ARRAY_TASK_ID \
    --k_out 5 \
    --k_in 5 \
    --n_max_param 60 \
    --fold_type "loso" \
    --run_name "loso_30dec25_all"

deactivate 