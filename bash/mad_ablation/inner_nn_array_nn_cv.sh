#!/bin/bash
#SBATCH -J mad_ablation_inner_nn_cv
#SBATCH --array=0-79
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --error=job_output/mad_ablation/cv/job_array_cv.%A_%a.err
#SBATCH --output=job_output/mad_ablation/cv/job_array_cv.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=j.f.severens@lumc.nl
#SBATCH --mem=8G

cd /exports/me-lcco-aml-hpc/Jeppe2/leukem_ai

mkdir -p job_output/mad_ablation/cv

source venv/bin/activate

export TF_CPP_MIN_LOG_LEVEL=2

echo "MAD-global ablation inner CV (NN, CV) SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "=================================="

MODEL_TYPE="NN"
FS_METHOD="mad_global"

python python/run_inner_cv_array.py \
    --model_type "$MODEL_TYPE" \
    --param_index $SLURM_ARRAY_TASK_ID \
    --k_out 5 \
    --k_in 5 \
    --n_max_param 80 \
    --fold_type "CV" \
    --run_name "cv_mad_global" \
    --fs_method "$FS_METHOD"

deactivate
