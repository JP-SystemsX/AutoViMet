#!/bin/bash
#BSUB -R select"[type=RHEL8_64]"
#BSUB -n 2
#BSUB -R "rusage[mem=16000]"
#BSUB -q hpc
export OMP_NUM_THREADS=2
#export python=/proj/tdtcad/pjungmann/Virtual_Metrology/deployment_simulation/virtual_environments/autoML3/bin/python3
export python=/proj/tdtcad2/pjungman/Virtual_Metrology/deployment_simulation/LOTSE_setup4/AutoViMet/AutoViMet_env/bin/python3
 
project_root_folder="/proj/tdtcad2/pjungman/Virtual_Metrology/deployment_simulation/LOTSE_setup4/AutoViMet"
benchmark_name="BD_Bench"
cd $project_root_folder

# (28+6) Models * 102 Datasets = 3468 Evaluations
INDEX=0 # TODO Increase for every batch up to 3468/50=70 (-1 because of starting at 0)
 
 

# Prepare file list
FILES=(configs/search_spaces/* configs/automl_configs/*)
 
# Total files
NUM_FILES=${#FILES[@]}
NUM_DATASETS=102 # TODO Read out automatically


mkdir -p "./tmp/logs"
 
start=$((INDEX * 50))
end=$(( (INDEX + 1) * 50 ))

for ((i = start; i <= end; i++));
do
    # Derive parameters
    DATA_ID=$(( i % NUM_DATASETS ))           # number between 0–35
    FILE_INDEX=$(( i / NUM_DATASETS ))    # index into file list
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
 
    lsf_bsub -J "dep_sim" \
        -n $OMP_NUM_THREADS \
        -R "span[hosts=1]" \
        -R "select[type=RHEL8_64]" \
        -o ./tmp/logs/$benchmark_name-$DATA_ID-$MODEL_NAME-$i-out.txt \
        -e ./tmp/logs/$benchmark_name-$DATA_ID-$MODEL_NAME-$i-err.txt \
        "$python ./src/main.py --model-name $MODEL_NAME --eval-config-adr configs/eval_configs/standard_eval.yml --data-config-adr configs/data_configs/glofo_bench.yml --search-space-adr \"$FILE\" --data-id $DATA_ID --search-algo $SEARCH_ALGO"
    echo '---'
 
done
 
 
echo 'DONE'
