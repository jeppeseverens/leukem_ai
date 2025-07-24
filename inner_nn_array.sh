#!/bin/bash
#SBATCH -J run_inner_nn_array
#SBATCH --array=0-95              # 96 hyperparameter combinations (0-indexed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         # 1 core per hyperparameter combination
#SBATCH --time=24:00:00           # Shorter time per job
#SBATCH --error=job_output/job_array.%A_%a.err
#SBATCH --output=job_output/job_array.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=j.f.severens@lumc.nl
#SBATCH --mem=4G                  # Much smaller memory per job

cd /exports/me-lcco-aml-hpc/Jeppe2/leukem_ai

# Create job_output directory if it doesn't exist
mkdir -p job_output

source venv/bin/activate

export TF_CPP_MIN_LOG_LEVEL=2

echo "Starting inner CV experiment on SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Processing hyperparameter combination $SLURM_ARRAY_TASK_ID out of 96"
echo "=================================="

# Define your arguments for each array job
MODEL_TYPE="NN"

# Run CV fold type
python python/run_inner_cv_array.py \
    --model_type "$MODEL_TYPE" \
    --param_index $SLURM_ARRAY_TASK_ID \
    --k_out 5 \
    --k_in 5 \
    --n_max_param 96 \
    --fold_type "CV"

# Run LOSO fold type
python python/run_inner_cv_array.py \
    --model_type "$MODEL_TYPE" \
    --param_index $SLURM_ARRAY_TASK_ID \
    --k_out 5 \
    --k_in 5 \
    --n_max_param 96 \
    --fold_type "loso"

deactivate 