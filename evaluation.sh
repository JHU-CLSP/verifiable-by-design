#!/bin/bash
source activate py310

written_data_name="/weka/scratch/jzhan237/repos/preference-quoting/gens/nq-dev_policy_2024-04-23_16-43-24_841642_bartscore.json /weka/scratch/jzhan237/repos/preference-quoting/gens/nq-dev_policy_2024-04-23_16-43-06_711131_bartscore.json /weka/scratch/jzhan237/repos/preference-quoting/gens/nq-dev_policy_2024-04-23_19-47-20_098987_bartscore.json /weka/scratch/jzhan237/repos/preference-quoting/gens/nq-dev_policy_2024-04-23_20-38-39_120509_bartscore.json" # "/weka/scratch/jzhan237/repos/preference-quoting/data/wiki_2024/wiki_2024_new_entries_cite_removed_len128.json"
EVAL_DATA="${1:-$written_data_name}"

# accelerate launch --multi_gpu --mixed_precision=bf16 
  # TODO: remember to change tokenizer meta-llama/Llama-2-7b-chat-hf \
python evaluation.py \
    $EVAL_DATA \
    --tokenizer meta-llama/Llama-2-7b-chat-hf \
    --ppl meta-llama/Llama-2-7b-hf \
    --nq_short_answer_acc 
    # --rerank_by_quip

    # QA setting
    # --tokenizer meta-llama/Llama-2-7b-chat-hf \
    # --ppl meta-llama/Llama-2-7b-hf \
    # --mauve \
    # --nq_short_answer_acc 

    # wiki pile setting
    # --tokenizer meta-llama/Llama-2-7b-hf \
    # --ppl mistralai/Mistral-7B-v0.1

    # --rerank_by_quip --ppl meta-llama/Llama-2-7b-hf --ppl mistralai/Mistral-7B-v0.1 \
