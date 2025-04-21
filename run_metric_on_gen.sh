#!/bin/bash
source activate py310

written_data_name="/weka/scratch/jzhan237/repos/preference-quoting/gens/nq-dev_policy_2024-04-23_16-43-24_841642.json /weka/scratch/jzhan237/repos/preference-quoting/gens/nq-dev_policy_2024-04-23_16-43-06_711131.json"
EVAL_DATA="${1:-$written_data_name}"

python run_metric_on_gen.py \
    $EVAL_DATA \
    -m bartscore
