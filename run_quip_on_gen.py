from jack_utils import load_json, save_json, add_suffix_before_ext
from quip_api import batch_quip_api
import argparse
import numpy as np
from tqdm import tqdm

def run_quip_on_gen(gen_file):
    gen = load_json(gen_file)
    n = len(gen['pred'][0])
    for preds in gen['pred']: assert len(preds) == n, 'preds have different lengths'
    pred_flatten = [pred for preds in gen['pred'] for pred in preds]
    quip_reports = batch_quip_api(pred_flatten)
    assert len(quip_reports) == len(pred_flatten), 'quip reports have different lengths'
    quip_reports_unflatten = [quip_reports[i:i+n] for i in range(0, len(quip_reports), n)]
    gen['quip_reports'] = quip_reports_unflatten
    save_json(gen, add_suffix_before_ext(gen_file, '_quip'))

def run_quip_on_data(input_file):
    data = load_json(input_file)
    length_list = [len(refs) for refs in data['refs']]
    refs_flatten = [ref for refs in data['refs'] for ref in refs]
    quip_reports = batch_quip_api(refs_flatten)
    assert len(quip_reports) == len(refs_flatten), 'quip reports have different lengths'
    length_prefix_sum = [0] + list(np.cumsum(length_list))
    quip_reports_unflatten = [quip_reports[i:j] for i, j in zip(length_prefix_sum[:-1], length_prefix_sum[1:])]
    for i, quip_reports in enumerate(quip_reports_unflatten):
        assert len(quip_reports) == length_list[i], 'quip reports have different lengths'
    data['quip_reports'] = quip_reports_unflatten
    save_json(data, add_suffix_before_ext(input_file, '_quip'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs='+', help='generation file')
    parser.add_argument('--run_on_data', '-d', action='store_true', help='run on data file instead of gen_file')
    args = parser.parse_args()

    run_func = run_quip_on_data if args.run_on_data else run_quip_on_gen
    for input_file in tqdm(args.input_files, desc='Files'):
        run_func(input_file)

    