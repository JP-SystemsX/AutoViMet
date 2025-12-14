#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --nodes=1
#SBATCH --mem=160G # Working Memory
#SBATCH --time=12:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_llm_timeseries
#SBATCH --job-name=cache2db
#SBATCH --output=hpc/logs/cache2db-%j-%a.out  # Output Address 
#SBATCH --error=hpc/logs/cache2db-%j-%a.err  # Output Address
#SBATCH --array=0

source ./hpc/modules.sh
srun python3 ./analysis/move_cache.py