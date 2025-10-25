#!/bin/bash
#SBATCH --cpus-per-task=4 # CPU Count
#SBATCH --nodes=1
#SBATCH --mem=160G # Working Memory
#SBATCH --time=1-00:00:00  # Runtime HH:MM:SS
#SBATCH --account=p_llm_timeseries
#SBATCH --job-name=ctr23
#SBATCH --output=hpc/logs/ctr23-%j-%a.out  # Output Address 
#SBATCH --error=hpc/logs/ctr23-%j-%a.err  # Output Address
#SBATCH --array=0-1155%50 #35 datasets * (29 Single Models + 4 AutoGluon Configs)  
# Load all Modules

# Prepare file list
FILES=(configs/search_spaces/* configs/automl_configs/*)

# Total files
NUM_FILES=${#FILES[@]}

# Derive parameters
DATA_ID=$(( SLURM_ARRAY_TASK_ID % 35 ))           # number between 0–35
FILE_INDEX=$(( SLURM_ARRAY_TASK_ID / 35 ))    # index into file list
FILE=${FILES[$FILE_INDEX]}

# Determine source folder
if [[ $FILE == configs/search_spaces/* ]]; then
    FOLDER="single_model"
    BASENAME=$(basename "$FILE")
    MODEL_NAME="${BASENAME%.*}"
    SEARCH_ALGO="HEBO"
elif [[ $FILE == configs/automl_configs/* ]]; then
    FOLDER="automl"
    MODEL_NAME="AutoGluon"
    SEARCH_ALGO="automl"
fi

source ./hpc/modules.sh
srun python3 ./src/main.py --model-name "$MODEL_NAME" --eval-config-adr "configs/eval_configs/standard_eval.yml" --data-config-adr "configs/data_configs/ctr23.yml" --search-space-adr "$FILE" --data-id "$DATA_ID" --search-algo "$SEARCH_ALGO"