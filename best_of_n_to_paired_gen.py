import argparse
from jack_utils import load_json, save_json, add_suffix_before_ext, proj_dir
import random
import os
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs='+', help='generation file')
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--delta_quip', type=float, default=0.1, help='min accetable quip difference')
    parser.add_argument('--delta_len', type=float, default=0.1, help='max accetable length difference ratio, i.e., abs(len1 - len2) / min(len1, len2)')

    parser.add_argument('--negate_score', action='store_true', help='add a negative sign to all raw quip scores. This will lead to Unquote-Tuning!')
    parser.add_argument('--num_pairs', type=int, default=1, help='max number of pairs to generate for each prompt')

    ### BELOW: optional... these seems to be not very useful ###
    parser.add_argument('--bartscore_percentile', type=float, default=None, help='only consider generation whose bartscore is in the top this percentile')
    parser.add_argument('--bartscore_rank', action='store_true', help='enforce that y_win must have higher bartscore than y_lose')
    parser.add_argument('--bartscore_val', type=float, default=None, help='enforce that |BS(y_win) - BS(y_lose)| < THRESH * min(|BS(y_win)|, |BS(y_lose)|)')
    parser.add_argument('--select_high_quip_random_response_when_failed', type=int, default=None, help='If not None, when failed to find a pair, select a high quip response (higher than current y_win) from some othre random prompts as y_lose. The input is number of trials. If still cannot find a pair, then just skip this prompt. Hopefully this teach model to prefer relevance over just quoting.')
    #############################

    parser.add_argument('--output_dir', type=str, default=f'{proj_dir()}/paired_gens')
    parser.add_argument('--seed', '-s', type=int, default=0, help='random seed')
    args = parser.parse_args()
    return args

def quip_constraint(delta_quip: float, q1: float, q2: float):
    return q1 - q2 >= delta_quip

def length_constraint(delta_len: float, l1: int, l2: int):
    return abs(l1 - l2)/min(l1, l2) <= delta_len

def bartscore_percentile_constraint(bartscore_percentile: float, bscores: List[float], bscore_to_test: float):
    if bartscore_percentile is None:
        return True
    return bscore_to_test >= np.percentile(bscores, 100*(1-bartscore_percentile))

def bartscore_ranking_constraint(bs1: float, bs2: float, enforced: bool = True):
    return (bs1 >= bs2) if enforced else True

def bartscore_value_constraint(delta_bs_value: float, bs1: float, bs2: float, enforced: bool = True):
    return abs(bs1-bs2) < delta_bs_value * min(abs(bs1), abs(bs2)) if enforced else True

if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for input_file in args.input_files:
        data = load_json(input_file)
        collected = []
        ii_idxs = []
        jj_idxs = []
        quip_win = []
        quip_diffs = []
        length_diffs = []
        n_after_bartscore_filter = []

        for i in tqdm(range(len(data['pred']))):
            prompt = data['full_prompts'][i]
            gold = data['refs'][i] # use first reference as gold
            preds = data['pred'][i]
            quips = [r['quip_25_beta'] if (r is not None and 'quip_25_beta' in r) else None for r in data['quip_reports'][i]]
            if args.negate_score:
                quips = [-q if q is not None else None for q in quips]
            if args.bartscore_percentile is None and not args.bartscore_rank and args.bartscore_val is None:
                # create a dummy bartscore list
                bartscores = [0] * len(preds)
            else:
                bartscores = data['bartscore'][i]
            lens = [len(tokenizer.encode(pred)) for pred in preds]
            # assert len(preds) == len(quips), 'preds and quips have different lengths'
            
            pred_quip_l = [
                (pred, quip, l, bs) for pred, quip, l, bs in zip(preds, quips, lens, bartscores) 
                if (quip is not None) and bartscore_percentile_constraint(args.bartscore_percentile, bartscores, bs)]
            n_after_bartscore_filter.append(len(pred_quip_l))
            if len(pred_quip_l) < 2:
                continue # no success, because less than 2 elements left
            pred_quip_l.sort(key=lambda x: x[1], reverse=True) # descending order by quip

            pairs = []
            for ii in range(len(pred_quip_l) - 1):
                j_range = list(range(ii + 1, len(pred_quip_l)))
                random.shuffle(j_range)
                for jj in j_range:
                    if quip_constraint(args.delta_quip, pred_quip_l[ii][1], pred_quip_l[jj][1]) and \
                        length_constraint(args.delta_len, pred_quip_l[ii][2], pred_quip_l[jj][2]) and \
                        bartscore_ranking_constraint(pred_quip_l[ii][3], pred_quip_l[jj][3], args.bartscore_rank) and \
                        bartscore_value_constraint(args.bartscore_val, pred_quip_l[ii][3], pred_quip_l[jj][3], args.bartscore_val is not None):
                        pair = (ii, jj)
                        # break # found pair, break immediately
                        pairs.append(pair)
                        if len(pairs) >= args.num_pairs: break
                # if pair is not None: break
                if len(pairs) >= args.num_pairs: break
            
            # if pair is not None:
            if len(pairs) > 0:
                # success
                for pair in pairs:
                    ii, jj = pair
                    collected.append({
                        'full_prompt': prompt, 'gold': gold,
                        'chosen': pred_quip_l[ii][0], 'rejected': pred_quip_l[jj][0],
                        'chosen_quip': pred_quip_l[ii][1], 'rejected_quip': pred_quip_l[jj][1],
                        'chosen_len': pred_quip_l[ii][2], 'rejected_len': pred_quip_l[jj][2],
                        'chosen_bartscore': pred_quip_l[ii][3], 'rejected_bartscore': pred_quip_l[jj][3],
                    })
                    ii_idxs.append(ii)
                    jj_idxs.append(jj)
                    quip_win.append(pred_quip_l[ii][1])
                    quip_diffs.append(pred_quip_l[ii][1] - pred_quip_l[jj][1])
                    length_diffs.append(abs(pred_quip_l[ii][2] - pred_quip_l[jj][2]))
            elif args.select_high_quip_random_response_when_failed is not None:
                # select a high quip response (higher than current y_win) from some othre random prompts as y_lose
                # ust skip this prompt. Hopefully this teach model to prefer relevance over just quoting.
                success = False
                for _ in range(args.select_high_quip_random_response_when_failed):
                    j=i
                    while j==i: j = random.randint(0, len(data['pred'])-1)
                    pred_i_top = data['pred'][i][0]
                    quip_i_top = data['quip_reports'][i][0]['quip_25_beta']
                    len_i_top = len(tokenizer.encode(pred_i_top))
                    pred_j_top = data['pred'][j][0]
                    quip_j_top = data['quip_reports'][j][0]['quip_25_beta']
                    len_j_top = len(tokenizer.encode(pred_j_top))
                    if quip_i_top is None or quip_j_top is None: continue

                    # need quip j - quip i >= delta_quip
                    if quip_constraint(args.delta_quip, quip_j_top, quip_i_top) \
                        and length_constraint(args.delta_len, len_i_top, len_j_top):
                        success = True
                        break
                if success:
                    collected.append({
                            'full_prompt': prompt, 'gold': gold,
                            'chosen': pred_i_top, 'rejected': pred_j_top,
                            'chosen_quip': quip_i_top, 'rejected_quip': quip_j_top,
                            'chosen_len': len_i_top, 'rejected_len': len_j_top,
                            'chosen_bartscore': pred_quip_l[ii][3], 'rejected_bartscore': pred_quip_l[jj][3],
                        })
                    ii_idxs.append(0)
                    jj_idxs.append(0)
                    quip_win.append(quip_i_top)
                    quip_diffs.append(quip_i_top - quip_j_top)
                    length_diffs.append(abs(len_i_top - len_j_top))

        success_rate = len(collected) / len(data['pred'])
        print(f'input file: {input_file}')
        print(f'success rate: {success_rate:.4f} (old len: {len(data["pred"])}, new len: {len(collected)})')
        print(f'stat,mean,std')
        for name, l in [('ii_idx', ii_idxs), ('jj_idx', jj_idxs), ('quip_win', quip_win), ('quip diff', quip_diffs), ('length diff', length_diffs), ('n_after_bartscore_filter', n_after_bartscore_filter)]:
            print(f'{name},{np.mean(l):.2f},{np.std(l):.2f}')
        
        extra_name = ''
        if args.negate_score:
            extra_name += '_NEG'
        if args.bartscore_rank:
            extra_name += '_bsrank'
        if args.bartscore_val is not None:
            extra_name += f'_bsval{args.bartscore_val:.2f}'
        if args.select_high_quip_random_response_when_failed is not None:
            extra_name += f'_shq{args.select_high_quip_random_response_when_failed}'
        if args.num_pairs > 1:
            extra_name += f'_np{args.num_pairs}'
        output_name = add_suffix_before_ext(os.path.basename(input_file), f'_dq{args.delta_quip:.2f}_dl{args.delta_len:.2f}_tok{args.tokenizer.split("/")[-1]}{extra_name}')
        if args.bartscore_percentile is not None:
            output_name += f'_bsp{args.bartscore_percentile:.2f}'
        output_path = os.path.join(args.output_dir, output_name)
        print(f'output file: {output_path}')
        out_data = {
            'args': vars(args),
            'stats': {
                'tokenizer': args.tokenizer,
                'success_rate': success_rate,
                'delta_quip': args.delta_quip,
                'delta_len': args.delta_len,
                'best_of': len(data['pred'][0]),
                'ii_idx': {'mean': np.mean(ii_idxs), 'std': np.std(ii_idxs)},
                'jj_idx': {'mean': np.mean(jj_idxs), 'std': np.std(jj_idxs)},
                'quip_win': {'mean': np.mean(quip_win), 'std': np.std(quip_win)},
                'quip_diff': {'mean': np.mean(quip_diffs), 'std': np.std(quip_diffs)},
                'length_diff': {'mean': np.mean(length_diffs), 'std': np.std(length_diffs)},
                'n_after_bartscore_filter': {'mean': np.mean(n_after_bartscore_filter), 'std': np.std(n_after_bartscore_filter)},
            },
            'data': collected
        }
        save_json(out_data, output_path)


