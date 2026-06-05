#!/bin/bash
#SBATCH -J mad_ablation_inner_xgb_loso
#SBATCH --array=0-119
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --error=job_output/mad_ablation/loso/job_array_loso.%A_%a.err
#SBATCH --output=job_output/mad_ablation/loso/job_array_loso.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=j.f.severens@lumc.nl
#SBATCH --mem=12G

cd /exports/me-lcco-aml-hpc/Jeppe2/leukem_ai

mkdir -p job_output/mad_ablation/loso

source venv/bin/activate

export TF_CPP_MIN_LOG_LEVEL=2

echo "MAD-global ablation inner CV (XGBOOST, LOSO) SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "=================================="

MODEL_TYPE="XGBOOST"
FS_METHOD="mad_global"

python python/run_inner_cv_array.py \
    --model_type "$MODEL_TYPE" \
    --param_index $SLURM_ARRAY_TASK_ID \
    --k_out 5 \
    --k_in 5 \
    --n_max_param 120 \
    --fold_type "loso" \
    --run_name "loso_mad_global" \
    --fs_method "$FS_METHOD"

deactivate
