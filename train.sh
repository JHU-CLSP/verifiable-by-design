#!/bin/bash
#SBATCH --exclude=c[001-003]

source activate py310

# NQ no BS filter
# ulimit -n 64000; python -u train.py model=llama2-7b-chat datasets=[paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0] loss=dpo loss.beta=0.01 exp_name=paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0_dpo_beta0.01 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=2816 'prompt_before="<s>[INST] <<SYS>>\nYou will be given a question. You need to produce a short paragraph that answers the question. Remember to be concise, accurate, and on-topic.\n<</SYS>>\n\n"' 'prompt_after=" [/INST]"'

# NQ with BS filter
# ulimit -n 64000; python -u train.py model=llama2-7b-chat datasets=[paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp0.25_bsval0.1] loss=dpo loss.beta=0.1 exp_name=paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp0.25_bsval0.1_dpo trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=2496 'prompt_before="<s>[INST] <<SYS>>\nYou will be given a question. You need to produce a short paragraph that answers the question. Remember to be concise, accurate, and on-topic.\n<</SYS>>\n\n"' 'prompt_after=" [/INST]"'

# WIKI no BS filter
# ulimit -n 64000; python -u train.py model=llama2-7b datasets=[paired_wiki_20k_llama2-7b_bo32_dq0.1_dl0.1_bsp1.0] loss=dpo loss.beta=0.01 exp_name=paired_wiki_20k_llama2-7b_bo32_dq0.1_dl0.1_bsp1.0_dpo_beta0.01 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=3200 'prompt_before=""' 'prompt_after=""'

ulimit -n 64000; python -u train.py model=llama2-7b-quote-nq datasets=[paired_nq_llama2-7b-chat-quote_bo32_dq0.1_dl0.1_bsp1.0] loss=dpo loss.beta=0.1 exp_name=paired_nq_llama2-7b-chat-quote_bo32_dq0.1_dl0.1_bsp1.0_dpo_beta0.1 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=2816 'prompt_before="<s>[INST] <<SYS>>\nYou will be given a question. You need to produce a short paragraph that answers the question. Remember to be concise, accurate, and on-topic.\n<</SYS>>\n\n"' 'prompt_after=" [/INST]"'


# TEMP: basic trainer for debug
# ulimit -n 64000; python -u train.py model=llama2-7b-chat datasets=[paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0] loss=dpo loss.beta=0.1 exp_name=paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0_dpo trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=2816 'prompt_before="<s>[INST] <<SYS>>\nYou will be given a question. You need to produce a short paragraph that answers the question. Remember to be concise, accurate, and on-topic.\n<</SYS>>\n\n"' 'prompt_after=" [/INST]"'
