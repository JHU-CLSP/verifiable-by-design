#!/bin/bash
source activate py310


ulimit -n 64000; python -u train.py model=llama2-7b-chat datasets=[paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0] loss=dpo loss.beta=0.1 exp_name=paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0_dpo trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=2816
