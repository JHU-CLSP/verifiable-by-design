# from kilt.eval_downstream import _metric_max_over_ground_truths, _exact_match_score, _f1_score, _rougel_score
import re
import string
from rouge import Rouge
from collections import Counter
from sacrebleu import corpus_bleu
from typing import List, Dict, Callable
import argparse
from jack_utils import load_json, save_json, list_of_list_to_csv, proj_dir, get_timestamp_now, path_no_ext
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from evaluate import load
import os
import torch

### START copfied from kilt/eval_downstream.py
def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

# answer nomalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

# F1 score definition
def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# ROUGEL score definition
def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except (ValueError, RecursionError):  # "Hypothesis is empty."
        # breakpoint()
        return 0.0
    return scores["rouge-l"]["f"]
### END copfied from kilt/eval_downstream.py


def do_kilt_eval(full_prompts: List[str], outputs: List[str], refs: List[List[str]], short_answer_dict: Dict[str, List[str]] = None, prompt_transform_f: Callable[[str], str] = None): #, dataset_name: str, output_format: str):
    # adapted from https://github.com/facebookresearch/KILT/blob/main/kilt/eval_downstream.py

    short_answer_correct = 0
    have_short_answer = 0
        
    total_count = 0

    # downstream metrics
    accuracy = 0
    normalized_em = 0
    normalized_f1 = 0
    rougel = 0

    em_list = []
    f1_list = []
    rougel_list = []
    bleu_list = []
    short_answer_correct_list = [] # -1: no gold short answer available, 0: short answer available but not seen, 1: short answer available and seen in generation

    explanations = []
    for (full_prompt, guess_answer, gold_candidate_answers) in tqdm(zip(full_prompts, outputs, refs), total=len(outputs), desc='Examples'):

        # if dataset_name not in ["ELI5_HF", "HumanEval"]:
        #     if output_format.lower() == "json":
        #         # evaluate as JSON
        #         import ast
        #         try:
        #             all_answer = ast.literal_eval(guess_answer)
        #             answer_only = all_answer["answer"]
        #             explanation = all_answer["explanation"]
        #         except Exception as e:
        #             print("Failed parse JSON")
        #             answer_only = guess_answer
        #             explanation = guess_answer
        #     elif output_format.lower() == "newline":
        #         answer_only = guess_answer.lower().split("explanation")[0].replace("answer:", "").replace("\n", "")
        #         explanation = " ".join(guess_answer.lower().split("explanation")[1:]).replace("explanation:", "").replace("\n", "")
        #     else:
        #         answer_only = guess_answer.split(";")[0]
        #         explanation = guess_answer.split(";")[1] if ";" in guess_answer else guess_answer
        # else:
        #     answer_only = guess_answer
        #     explanation = guess_answer

        explanation = guess_answer
        answer_only = '' # NOTE: no answer only for ELI5 for now
        explanations.append(explanation)
        total_count += 1
        

        # # 0. accuracy = strict exact match
        # local_accuracy = 0
        # if guess_answer in gold_candidate_answers:
        #     local_accuracy = 1
        # accuracy += local_accuracy

        # 1. normalized exact match
        local_em = _metric_max_over_ground_truths(
            _exact_match_score, answer_only, gold_candidate_answers
        )
        normalized_em += local_em
        em_list.append(local_em)

        # 2. normalized f1
        local_f1 = _metric_max_over_ground_truths(
            _f1_score, answer_only, gold_candidate_answers
        )
        normalized_f1 += local_f1
        f1_list.append(local_f1)

        # 3. rougel
        local_rougel = _metric_max_over_ground_truths(
            _rougel_score, explanation, gold_candidate_answers
        )
        rougel += local_rougel
        rougel_list.append(local_rougel)

        if short_answer_dict is not None:
            short_answer_list = short_answer_dict.get(prompt_transform_f(full_prompt), [])
            if len(short_answer_list) > 0:
                have_short_answer += 1
                flag = int(any([short_answer in guess_answer.lower() for short_answer in short_answer_list]))
                short_answer_correct += flag
                short_answer_correct_list.append(flag)
            else:
                short_answer_correct_list.append(-1)


    if total_count > 0:
        # accuracy /= total_count
        normalized_em /= total_count
        normalized_f1 /= total_count
        rougel /= total_count

    short_answer_acc = short_answer_correct/have_short_answer if have_short_answer > 0 else -1
    if have_short_answer > 0:
        print(f'{have_short_answer/total_count:.2f} of examples have short answer')

    total_bleu = corpus_bleu(explanations, refs, lowercase=True).score / 100
    zipped = [
        {'prompt': p, 'pred': o, 'refs': r, 'rougel': rou}
        for p, o, r, rou in zip(full_prompts, outputs, refs, rougel_list)
    ]
    res = {
            # "accuracy": round(accuracy, 3),
            "em": round(normalized_em, 3),
            "f1": round(normalized_f1, 3),
            "rougel": round(rougel, 3),
            "bleu": round(total_bleu, 3),
            "em_list": em_list,
            "f1_list": f1_list,
            "rougel_list": rougel_list,
            "short_answer_acc": short_answer_acc,
            "zipped": zipped
    }
    return res

def quip_eval(quip_reports: List[Dict]) -> Dict:
    # macro average: avg per-instance quip scores directly
    macro_quip_list = [
        quip_report['quip_25_beta'] if quip_report['quip_25_beta'] is not None else 0 
        for quip_report in quip_reports]
    macro_quip = np.mean(macro_quip_list)

    # micro average: calculate based on total numerator and denominator
    micro_quip_n = np.sum([quip_report['numerator'] for quip_report in quip_reports])
    micro_quip_d = np.sum([quip_report['denominator'] for quip_report in quip_reports])
    micro_quip = micro_quip_n / micro_quip_d
    
    return macro_quip, micro_quip, macro_quip_list

def parse_args():
    parser = argparse.ArgumentParser()
    # basics
    parser.add_argument('gen_files', type=str, nargs='+', help='generation files')
    parser.add_argument('--tokenizer', '-t', type=str, required=True, help='tokenizer used for generation')
    parser.add_argument('--ppl', type=str, default=None, help='model for ppl calculation. default is None, which means no ppl calculation')
    parser.add_argument('--nq_short_answer_acc', '-sa', action='store_true', help='calculate nq short answer acc, 1 if the predicted long answer contains the short answer')
    parser.add_argument('--mauve', action='store_true', help='calculate mauve score')

    # quip
    parser.add_argument('--single_quip', action='store_true', help='only calculate quip based on the first pred of each prompt')
    parser.add_argument('--rerank_by_quip', action='store_true', help='rerank multiple generation by highest quip')
    parser.add_argument('--num_rerank_candidates', type=int, default=None, help='number of candidates to rerank by quip')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.nq_short_answer_acc:
        ds = load_dataset('nq_open', split='validation')
        nq_dev_sa_dict = {ex['question']: ex['answer'] for ex in ds}
        # nq_prompt_f: Callable[[str], str] = lambda prompt: prompt.replace('?\nAnswer:', '').lower().strip() # reverse the full prompt
        nq_prompt_f: Callable[[str], str] = lambda prompt: (prompt[:-1] if prompt.endswith("?") else prompt.replace('?\nAnswer:', '')).lower().strip() # reverse the full prompt
    else:
        nq_dev_sa_dict = None
        nq_prompt_f = None
    eval_ppl = args.ppl is not None
    eval_mauve = args.mauve

    results_json = load_json(args.gen_files[0])
    eval_quip = 'quip_reports' in results_json
    eval_bartscore = 'bartscore' in results_json
    results = [['gen_file', 'avglen', 'rougel', 'bleu']]
    if eval_quip: results[0].extend(['quip_macro', 'quip_micro'])
    if eval_bartscore: results[0].extend(['bartscore'])
    if args.nq_short_answer_acc: results[0].append('nq_short_answer_acc')
    if eval_ppl: results[0].append(f'ppl-{args.ppl}')
    if eval_mauve: results[0].append('mauve')

    for gen_file in tqdm(args.gen_files, desc='Files'):
        results_json = load_json(gen_file)
        if args.rerank_by_quip:
            assert eval_quip, 'quip_reports not found in generation file'
            outputs = []
            selected_quip_reports = []
            for preds, quip_reports in zip(results_json['pred'], results_json['quip_reports']):
                quip_scores = [quip_report['quip_25_beta'] if quip_report['quip_25_beta'] is not None else 0 for quip_report in quip_reports]
                if args.num_rerank_candidates is not None:
                    quip_scores = quip_scores[:args.num_rerank_candidates]
                argmax_idx = np.argmax(quip_scores)
                outputs.append(preds[argmax_idx])
                selected_quip_reports.append(quip_reports[argmax_idx])
        else:
            outputs = [l[0] for l in results_json['pred']] # currently, only support num_return_sequences=1! more sequences will be ignored
        # breakpoint()
        refs = results_json['refs']
        len_preds = [len(tokenizer.encode(output)) for output in outputs]
        avglen = np.mean(len_preds)
        eval_res = do_kilt_eval(results_json['full_prompts'], outputs, refs, short_answer_dict=nq_dev_sa_dict, prompt_transform_f=nq_prompt_f)
        zipped = eval_res['zipped']
        for i, l in enumerate(len_preds):
            zipped[i]['len'] = l
        line_res = [gen_file, avglen, eval_res['rougel'], eval_res['bleu']]
        if eval_quip:
            if args.rerank_by_quip:
                flattened_quip_reports = selected_quip_reports
            elif args.single_quip:
                flattened_quip_reports = [quip_reports[0] for quip_reports in results_json['quip_reports']]
            else:
                flattened_quip_reports = [quip_report for quip_reports in results_json['quip_reports'] for quip_report in quip_reports]
            macro_quip, micro_quip, macro_quip_list = quip_eval(flattened_quip_reports)
            line_res.extend([macro_quip, micro_quip])
            for i in range(len(zipped)):
                zipped[i]['quip_macro'] = macro_quip_list[i]
                if 'quoted_segments' in flattened_quip_reports[i]:
                    zipped[i]['quoted_segments'] = flattened_quip_reports[i]['quoted_segments']
                if 'longest_segment_string_length' in flattened_quip_reports[i]:
                    zipped[i]['longest_segment_string_length'] = flattened_quip_reports[i]['longest_segment_string_length']
        
        if eval_bartscore:
            line_res.append(np.mean(results_json['bartscore']))

        if args.nq_short_answer_acc:
            line_res.append(eval_res['short_answer_acc'])
        
        if eval_ppl:
            cache_path = f'{proj_dir()}/cache/ppl-{args.ppl}' + gen_file + '.ppl_cache.json'
            if os.path.exists(cache_path):
                ppl_results = load_json(cache_path)
            else:
                # perplexity = load("perplexity",  module_type= "measurement")
                # ppl_results = perplexity.compute(data=outputs, model_id=args.ppl)
                BATCH_SIZE = 1 if args.ppl == 'mistralai/Mixtral-8x7B-v0.1' else 16
                perplexity = load("./custom_perplexity.py",  module_type= "measurement")
                # breakpoint()
                ppl_results = perplexity.compute(data=outputs, model_id=args.ppl, batch_size=BATCH_SIZE, torch_dtype=torch.bfloat16)
                save_json(ppl_results, cache_path)
            line_res.append(ppl_results['mean_perplexity'])
            for i in range(len(zipped)):
                zipped[i]['ppl'] = ppl_results['perplexities'][i]
        
        if eval_mauve:
            mauve = load('mauve')
            mauve_results = mauve.compute(predictions=outputs, references=[ref[0] for ref in refs], device_id=0) # since mauve only supports single reference
            line_res.append(mauve_results.mauve)

        results.append(line_res)
        # outpath = add_suffix_before_ext(gen_file, '_eval')
        # results_json['eval'] = zipped
        # save_json(results_json, outpath)
        outjson = {
            'eval_args': vars(args),
            'gen_path': gen_file, 
            'args': results_json['args'],
            'gen_config': results_json.get('gen_config'),
            'eval': zipped}
        outpath = f'{proj_dir()}/eval_examples/{path_no_ext(os.path.basename(gen_file))}_{get_timestamp_now()}.json'
        save_json(outjson, outpath)
        print(f'Saved annotated results to {outpath}.')
    
    print(list_of_list_to_csv(results))
