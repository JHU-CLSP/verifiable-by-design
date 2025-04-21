#!/bin/bash
source activate py310

# CHAT model nq sft with system prompt
# ulimit -n 64000; python train.py model=llama2-7b-chat datasets=[nq_sft] loss=sft exp_name=llama2-7b-chat_nq_sft trainer=FSDPTrainer sample_during_eval=false gradient_accumulation_steps=2 eval_every=10000 'prompt_before="<s>[INST] <<SYS>>\nYou will be given a question. You need to produce a short paragraph that answers the question. Remember to be concise, accurate, and on-topic.\n<</SYS>>\n\n"' 'prompt_after=" [/INST]"'

# BASE model wiki sft: no system prompt
ulimit -n 64000; python train.py model=llama2-7b datasets=[wiki_pile_sft] loss=sft exp_name=llama2-7b-wiki_pile_sft trainer=FSDPTrainer sample_during_eval=false gradient_accumulation_steps=2 eval_every=5000 'prompt_before=""' 'prompt_after=""'






# temp: basic trainer for debug
# ulimit -n 64000; python train.py model=llama2-7b-chat datasets=[nq_sft] loss=sft exp_name=llama2-7b-chat_nq_sft trainer=BasicTrainer sample_during_eval=false gradient_accumulation_steps=2 eval_every=10000 'prompt_before="<s>[INST] <<SYS>>\nYou will be given a question. You need to produce a short paragraph that answers the question. Remember to be concise, accurate, and on-topic.\n<</SYS>>\n\n"' 'prompt_after=" [/INST]"'
