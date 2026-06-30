#!/bin/bash
#SBATCH -J withnos_inner_xgb_loso
#SBATCH --array=0-119              # 120 hyperparameter combinations (0-indexed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --error=job_output/with_nos/loso/job_array_loso.%A_%a.err
#SBATCH --output=job_output/with_nos/loso/job_array_loso.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=j.f.severens@lumc.nl
#SBATCH --mem=12G

cd /exports/me-lcco-aml-hpc/Jeppe2/leukem_ai

mkdir -p job_output/with_nos/loso

source venv/bin/activate

export TF_CPP_MIN_LOG_LEVEL=2

echo "With-NOS inner CV (XGBOOST, LOSO) SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "=================================="

MODEL_TYPE="XGBOOST"
FS_METHOD="eta2"

python python/run_inner_cv_array.py \
    --model_type "$MODEL_TYPE" \
    --param_index $SLURM_ARRAY_TASK_ID \
    --k_out 5 \
    --k_in 5 \
    --n_max_param 120 \
    --fold_type "loso" \
    --run_name "loso_10feb26_eta2_withnos" \
    --fs_method "$FS_METHOD" \
    --include_nos

deactivate
