#!/bin/bash
#SBATCH -J final_selection_nn_loso
#SBATCH --array=0-59              # Adjust based on parameter combinations
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         # 1 core per hyperparameter combination
#SBATCH --time=8:00:00           
#SBATCH --error=job_output/final_loso/job_array_final_loso.%A_%a.err
#SBATCH --output=job_output/final_loso/job_array_final_loso.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=j.f.severens@lumc.nl
#SBATCH --mem=8G                  

cd /exports/me-lcco-aml-hpc/Jeppe2/leukem_ai

# Create job_output directory if it doesn't exist
mkdir -p job_output/final_loso

source venv/bin/activate

export TF_CPP_MIN_LOG_LEVEL=2

echo "Starting final hyperparameter selection (LOSO fold type) on SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Processing hyperparameter combination $SLURM_ARRAY_TASK_ID"
echo "=================================="

# Define your arguments for each array job
MODEL_TYPE="NN"

# Run LOSO fold type for final selection
python python/run_final_selection_array.py \
    --model_type "$MODEL_TYPE" \
    --param_index $SLURM_ARRAY_TASK_ID \
    --k_folds 5 \
    --n_max_param 60 \
    --fold_type "loso" \
    --run_name "final_loso_nn"

deactivate
