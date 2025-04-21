#!/bin/bash
source activate py310

# CKPT_PATH=/weka/scratch/jzhan237/repos/preference-quoting/runs/jzhan237/paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp0.25_bsval0.1_37k_dpo_2024-02-20_20-38-10_377562/LATEST/policy.pt

CKPT_PATH=$1 #/weka/scratch/jzhan237/repos/preference-quoting/runs/jzhan237/paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp0.25_bsval0.1_dpo_2024-02-20_20-50-46_861147/LATEST/policy.pt

#/weka/scratch/jzhan237/repos/preference-quoting/runs/jzhan237/llama2-7b-chat_nq_sft_2024-02-07_21-08-30_637448/LATEST/policy.pt # e.g. /weka/scratch/jzhan237/repos/preference-quoting/runs/jzhan237/llama2-7b-chat_nq_sft_2024-02-07_21-08-30_637448/LATEST/policy.pt

python checkpoint_pt_to_hf.py meta-llama/Llama-2-7b-chat-hf $CKPT_PATH
