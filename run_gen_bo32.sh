#!/bin/bash
eval "$(conda shell.bash hook)"

conda activate azr
export PROJ_DIR=/weka/scratch/dkhasha1/jzhan237/scratch_files/repos/safety-control
export PYTHONPATH=/weka/scratch/dkhasha1/jzhan237/scratch_files/repos/safety-control
export MODEL_DIR=/weka/scratch/dkhasha1/jzhan237/scratch_files/models

##################################################################

DATAPATHS=("data/nq/dev")

##################################################################

# MODEL_DISPLAY_NAME="gemma2-9b-it"
# MODEL="$MODEL_DIR/$MODEL_DISPLAY_NAME"

MODEL_DISPLAY_NAME="llama3.1-8b-instruct"
MODEL="$MODEL_DIR/$MODEL_DISPLAY_NAME"

##################################################################

ts=$(date +%F_%T)
echo "Starting inference at $ts"
for DATAPATH in "${DATAPATHS[@]}"
do
    echo "DATAPATH: $DATAPATH"
    python $PROJ_DIR/src/oai_inference.py \
        ${@} \
        --model $MODEL --model_display_name $MODEL_DISPLAY_NAME \
        --input_dataset $DATAPATH \
        -p prompt_templates/concise_gr_nosys.json \
        --kwargs $PROJ_DIR/src/kwargs/bestof32.json \
        | tee -a "logs/oai_inference_$ts.log"
done

echo "log file: logs/oai_inference_$ts.log"

