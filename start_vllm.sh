#!/bin/bash
eval "$(conda shell.bash hook)"
if [ -n "$IS_DSTI" ]; then
    export OUTLINES_CACHE_DIR=~/tmp/.outlines # bug fix: https://github.com/vllm-project/vllm/issues/4193#issue-2252298694
    export PROJ_DIR=$PWD
    export MODEL_DIR=/weka/scratch/jzhan237/models
    export DATA_DIR=$PWD/data
    export OUTPUT_DIR=$PWD/dpo/runs
    export PYTHONPATH=$PROJ_DIR
    conda activate vllm310
else
    conda activate vllm # py310
fi

# BUG FIX: If "too many open files": add --disable-frontend-multiprocessing (https://github.com/vllm-project/vllm/issues/7290)

model=$1 # "microsoft/Phi-3-mini-4k-instruct" "meta-llama/Meta-Llama-3-70B-Instruct"
size=${2:-8} # default is 8
port=8000

# convenient short names
if [ "$model" == "llama3-70b" ]; then
    model="meta-llama/Meta-Llama-3-70B-Instruct"
elif [ "$model" == "phi-3-mini" ]; then
    model="microsoft/Phi-3-mini-4k-instruct"
fi

echo "Starting vLLM API server with model $model on port $port with tensor parallel size $size"
python -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port $port \
    --dtype auto \
    --trust-remote-code \
    --api-key token-abc123 \
    --tensor-parallel-size $size \
    ${@:3}
    # --chat-template chat_templates/phi-3.jinja