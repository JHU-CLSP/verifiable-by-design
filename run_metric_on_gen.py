from jack_utils import load_json, save_json, add_suffix_before_ext, flatten_lol, proj_dir
from quip_api import batch_quip_api
import argparse
import numpy as np
from tqdm import tqdm
from BARTScore.bart_score import BARTScorer
from typing import List
import evaluate
import torch
from datasets import load_from_disk
import os

def load_data(gen_file: str):
    if gen_file.endswith('.json'):
        return load_json(gen_file)
    else:
        # assume it's a huggingface dataset
        gen = load_from_disk(gen_file)
        pred = gen['response']
        if isinstance(pred[0], str):
            pred = [[p] for p in pred]
        # reformat as json
        return {
            'pred': pred,
            'refs': gen['reference'],
            'args': {'prompt_before': '', 'prompt_after': ''}, # PLACEHOLDER
            'full_prompts': gen['prompt']
        }

def metric_name_mapping(metric: str) -> str:
    if metric == 'bartscore':
        return 'bartscore'
    elif metric == 'quip':
        return 'quip_reports'
    elif metric.startswith('ppl_'):
        return metric
    elif metric == 'mauve':
        return metric
    else:
        return None

def run_quip(ctx: List[str], preds: List[str], refs: List[List[str]], batch_size=500, simple_report_format=True, **kwargs):
    print('Running QUIP API')
    quip_reports = batch_quip_api(preds, batch_size=batch_size, simple_report_format=simple_report_format, **kwargs)
    assert quip_reports is not None, "error during running QUIP API"
    return quip_reports

def run_bartscore(ctx: List[str], preds: List[str], refs: List[List[str]], batch_size=32, **kwargs):
    bart_scorer = BARTScorer(device=kwargs.get('device', 'cuda'), checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path=os.path.join(os.getcwd(), 'BARTScore/bart_score.pth'))
    min_num_ref = min([len(ref) for ref in refs])
    refs = [ref[:min_num_ref] for ref in refs] # truncate refs to the same length
    print('Calculating BARTScore, number of refs:', len(refs[0]))
    return bart_scorer.multi_ref_score(preds, refs, agg="max", batch_size=batch_size)

def run_ppl(ctx: List[str], preds: List[str], refs: List[List[str]], batch_size=4, **kwargs):
    # batch size here need to be smaller since model is potentially large
    assert 'model' in kwargs, 'model is not provided'
    perplexity = evaluate.load("./cond_perplexity.py",  module_type= "measurement")
    ppl_results = perplexity.compute(context=ctx, continuation=preds, model_id=kwargs['model'], batch_size=batch_size, torch_dtype=torch.bfloat16) # NOTE: using bfloat16 to save memory
    print(f'mean perplexity on model {kwargs["model"]}: {ppl_results["mean_perplexity"]}')
    return ppl_results['perplexities']

def run_mauve(ctx: List[str], preds: List[str], refs: List[List[str]], **kwargs):
    mauve = evaluate.load('mauve')
    mauve_results = mauve.compute(predictions=preds, references=[ref[0] for ref in refs], device_id=0) # since mauve only supports single reference
    return mauve_results.mauve

def run_metric_on_gen_batch(gen_file: str, metrics: List[str], suffix: str):
    # prepare pred and refs
    gen = load_data(gen_file)
    filepath = gen_file
    if not filepath.endswith('.json'):
        filepath += '.json'
    for metric in metrics:
        n = len(gen['pred'][0])
        for preds in gen['pred']: assert len(preds) == n, 'preds have different lengths'
        pred_flatten = [pred for preds in gen['pred'] for pred in preds] # list of prediction strings
        ref_expanded = flatten_lol([[ref] * n for ref in gen['refs']]) # list of list of references
        full_prompts_flattent = flatten_lol([[prompt] * n for prompt in gen['full_prompts']]) # list of prompts
        ctx_flatten = [gen['args']['prompt_before'] + p + gen['args']['prompt_after'] for p in full_prompts_flattent] # list of context strings

        kwargs = {}
        if metric == 'quip':
            score_fun = run_quip
        elif metric == 'bartscore':
            score_fun = run_bartscore
        elif metric.startswith('ppl_'):
            score_fun = run_ppl
            kwargs['model'] = '_'.join(metric.split('_')[1:])
        elif metric == 'mauve':
            score_fun = run_mauve
        else:
            raise NotImplementedError
        scores = score_fun(ctx_flatten, pred_flatten, ref_expanded, **kwargs)

        assert len(scores) == len(pred_flatten), 'scores and preds have different lengths'
        scores_unflatten = [scores[i:i+n] for i in range(0, len(scores), n)]
        gen[metric_name_mapping(metric)] = scores_unflatten

        filepath = add_suffix_before_ext(filepath, f'_{metric}')
    filepath = add_suffix_before_ext(filepath, suffix)
    print(f'Saving to {filepath}')
    save_json(gen, filepath)

def check_modes(modes: List[str]):
    '''
    for perplexity, the mode should be ppl_{model_name}
    '''
    for mode in modes:
        assert metric_name_mapping(mode) is not None, f"mode {mode} is not supported"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs='+', help='generation file')
    parser.add_argument('--modes', '-m', type=str, nargs='+', help='metric mode')
    parser.add_argument('--outname_suffix', '-s', type=str, default='', help='suffix of output file name')
    # parser.add_argument('--run_on_data', '-d', action='store_true', help='run on data file instead of gen_file')
    args = parser.parse_args()

    print(args)
    for input_file in tqdm(args.input_files, desc='Files'):
        print(f'Processing {input_file}')
        run_metric_on_gen_batch(input_file, args.modes, args.outname_suffix)

    