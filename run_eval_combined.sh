#!/bin/bash
eval "$(conda shell.bash hook)"

export PROJ_DIR=/weka/scratch/dkhasha1/jzhan237/scratch_files/repos/safety-control
export PYTHONPATH=/weka/scratch/dkhasha1/jzhan237/scratch_files/repos/safety-control
export MODEL_DIR=/weka/scratch/dkhasha1/jzhan237/scratch_files/models

##################################################################

DATAPATH="data/nq/dev"

##################################################################

PROMPT_TEMPLATE_NAME="concise_gr"

# llama 3.1
# MODEL="/weka/scratch/jzhan237/repos/safety-control/dpo/runs/jzhan237/llama3.1-8b-instruct-DPO-qt_llama3.1-8b-inst_bo32_dq0.10_dl0.05-concise_sysp-beta0.3_2024-08-14_06-48-20_899352/LATEST/policy"
# MODEL_DISPLAY_NAME="qt_llama3.1-8b-inst_bo32_dq0.10_dl0.05-concise_sysp-beta0.3-20k"
# MODEL="/weka/scratch/jzhan237/models/llama3.1-8b-instruct"
# MODEL_DISPLAY_NAME="llama3.1-8b-instruct"
# negate beta0.1
# MODEL="/weka/scratch/dkhasha1/jzhan237/scratch_files/repos/safety-control/dpo/runs/jzhan237/llama3.1-8b-instruct-DPO-qt_llama3.1-8b-instruct_bo32_dq0.10_dl0.10-concise_sysp_NEG-beta0.1-lr5e-7_2024-11-03_13-40-29_964514/LATEST/policy"
# MODEL_DISPLAY_NAME="qt_llama3.1-8b-instruct_bo32_dq0.10_dl0.10-concise_sysp_NEG-beta0.1-20k"
# negate beta0.3
# MODEL="/weka/scratch/dkhasha1/jzhan237/scratch_files/repos/safety-control/dpo/runs/jzhan237/llama3.1-8b-instruct-DPO-qt_llama3.1-8b-instruct_bo32_dq0.10_dl0.10-concise_sysp_NEG-beta0.3-lr5e-7_2024-11-03_15-57-16_501634/LATEST/policy"
# MODEL_DISPLAY_NAME="qt_llama3.1-8b-instruct_bo32_dq0.10_dl0.10-concise_sysp_NEG-beta0.3-20k"
# negate beta0.5
MODEL="/weka/scratch/dkhasha1/jzhan237/scratch_files/repos/safety-control/dpo/runs/jzhan237/llama3.1-8b-instruct-DPO-qt_llama3.1-8b-instruct_bo32_dq0.10_dl0.10-concise_sysp_NEG-beta0.5-lr5e-7_2024-11-03_20-43-15_102577/LATEST/policy"
MODEL_DISPLAY_NAME="qt_llama3.1-8b-instruct_bo32_dq0.10_dl0.10-concise_sysp_NEG-beta0.5-20k"

# gemma2
# no system prompt for gemma2
# PROMPT_TEMPLATE_NAME="concise_gr_nosys"
# MODEL="$MODEL_DIR/gemma2-9b-it"
# MODEL_DISPLAY_NAME="gemma2-9b-it"
# MODEL="/weka/scratch/dkhasha1/jzhan237/scratch_files/repos/safety-control/dpo/runs/jzhan237/gemma2-9b-it-DPO-qt_gemma2-9b-it-inst_bo32_dq0.10_dl0.10-concise_sysp-beta0.1_2024-10-14_01-56-49_872904/LATEST/policy"
# MODEL_DISPLAY_NAME="qt_gemma2-9b-it_bo32_dq0.10_dl0.10-concise_sysp-beta0.1-20k"

# starling-7b-beta
# MODEL="$MODEL_DIR/starling-7b-beta"
# MODEL_DISPLAY_NAME="starling-7b-beta"
# MODEL="/weka/scratch/dkhasha1/jzhan237/scratch_files/repos/safety-control/dpo/runs/jzhan237/starling-7b-beta-DPO-qt_starling-7b-beta_bo32_dq0.10_dl0.10-concise_sysp-beta0.5-lr5e-7_2024-10-14_17-35-00_426473/LATEST/policy"
# MODEL_DISPLAY_NAME="qt_starling-7b-beta_bo32_dq0.10_dl0.10-concise_sysp-beta0.5-lr5e-7-20k"

##################################################################

# STEP 1: Generate on eval data
# NOTE: need to start vllm server before running this

ts=$(date +%F_%T)
echo "Starting inference at $ts"
# for DATAPATH in "${DATAPATHS[@]}"
# do
echo "DATAPATH: $DATAPATH"
conda run -n azr python $PROJ_DIR/src/oai_inference.py \
    ${@} \
    --model $MODEL --model_display_name $MODEL_DISPLAY_NAME \
    --input_dataset $DATAPATH \
    -p prompt_templates/$PROMPT_TEMPLATE_NAME.json \
    | tee -a "logs/oai_inference_$ts.log" # --kwargs $PROJ_DIR/src/kwargs/p095.json \
# done
echo "log file: logs/oai_inference_$ts.log"

##################################################################

# STEP 2: Evaluate generated data
# NOTE: REQUIRES GPU; make sure to STOP VLLM SERVER to clear vllm GPU memory before running this part.

OUTPATH=${DATAPATH}_model-${MODEL_DISPLAY_NAME}_${PROMPT_TEMPLATE_NAME}

export PYTHONPATH=$PWD
conda activate py310

# # conda run -n py310 
python run_metric_on_gen.py \
    $OUTPATH \
    --modes quip bartscore

# conda run -n py310 
python evaluation.py \
    "${OUTPATH}_quip_bartscore.json" \
    --tokenizer $MODEL \
    --nq_short_answer_acc \
    --ppl $MODEL_DIR/mistral-7b-v0.1