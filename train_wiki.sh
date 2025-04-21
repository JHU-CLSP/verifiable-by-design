#!/bin/bash
source activate py310

ulimit -n 64000; python -u train.py model=llama2-7b-chat datasets=[paired_wiki_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0] loss=dpo loss.beta=0.1 exp_name=paired_wiki_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0_dpo trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=2848 'prompt_before="<s>[INST] <<SYS>>\nYou will be given a snippet of text from Wikipedia. You need to produce a short continuation of that paragraph by reciting Wikipedia exactly. Remember to be concise and accurate. Start with your answer directly.\n<</SYS>>\n\n"' 'prompt_after=" [/INST]"'

# IMPORTANT NOTE: the prompt above is different from NQ



# TEMP: basic trainer for debug
# ulimit -n 64000; python -u train.py model=llama2-7b-chat datasets=[paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0] loss=dpo loss.beta=0.1 exp_name=paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0_dpo trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=2816 'prompt_before="<s>[INST] <<SYS>>\nYou will be given a question. You need to produce a short paragraph that answers the question. Remember to be concise, accurate, and on-topic.\n<</SYS>>\n\n"' 'prompt_after=" [/INST]"'
