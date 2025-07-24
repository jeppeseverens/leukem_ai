
#!/bin/bash
#SBATCH -J run_inner_nn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --error=job.%A_%a.err
#SBATCH --output=job.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=j.f.severens@lumc.nl
#SBATCH --mem=64G

cd /exports/me-lcco-aml-hpc/Jeppe2/leukem_ai

source venv/bin/activate

export TF_CPP_MIN_LOG_LEVEL=2

echo "Starting inner CV experiment on SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "=================================="

# Define your arguments for each array job
MODEL_TYPE="NN"

python python/run_inner_cv.py \
    --model_type "$MODEL_TYPE" \
    --n_jobs 32 \
    --k_out 5 \
    --k_in 5 \
    --n_max_param 96 \
    --fold_type "loso"

python python/run_inner_cv.py \
  --model_type "$MODEL_TYPE" \
  --n_jobs 32 \
  --k_out 5 \
  --k_in 5 \
  --n_max_param 96 \
  --fold_type "CV"

deactivate