import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import json
from jack_utils import proj_dir

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)['train']
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data

def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data

def get_paired_gen(data_dir: str, silent: bool = False) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    '''
    Construct dataset from paired_gens/* data, i.e., (prompt, chosen, rejected) pairs

    BE CAREFUL: this does not have the "split" argument. must manually make sure train or dev
    '''
    from jack_utils import load_json
    json_data = load_json(data_dir)
    data = defaultdict(lambda: defaultdict(list))
    for line in tqdm.tqdm(json_data['data'], desc='Processing', disable=silent):
        prompt = line['full_prompt']
        chosen = line['chosen']
        rejected = line['rejected']
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data


def get_sft(data_path: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    '''
    Note: this is only used for SFT, there's no real preference pairs
    '''
    from jack_utils import load_json
    # if split != 'train':
    #     split = 'dev'
    # data_json = load_json(f'{proj_dir()}/data/nq/nq-{split}.json')
    data_json = load_json(data_path)
    data = defaultdict(lambda: defaultdict(list))
    for i in tqdm.tqdm(range(len(data_json['full_prompts'])), desc='Processing', disable=silent):
        prompt = data_json['full_prompts'][i]
        chosen = data_json['refs'][i][0] # fake
        rejected = data_json['refs'][i][0] # fake
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = data_json['refs'][i][0]
    
    return data

def get_eli5(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    '''
    Note: this is only used for SFT, there's no real preference pairs
    '''
    print(f'Loading eli5 dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('eli5', split='train_eli5' if split=='train' else 'validation_eli5', cache_dir=cache_dir)
    print('done')
    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing eli5', disable=silent):
        question = row['title'].strip()
        prompt = f"{question}\nAnswer:" # follow prompt format of according to paper
        chosen = row['answers']['text'][0] # the highest scoring answer
        rejected = row['answers']['text'][1] if len(row['answers']['text']) > 1 else 'No answer.' # the 2nd best one if exists
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
        data[prompt]['refs'] = row['answers']['text']
    
    return data

def get_eli5_quip_pair(split: str, separate_eval=False, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    '''
    Prompts is from eli5 train, responses are highest / lowest quip answer of 16 model generated responses, where the model is finetuned on entire eli5 train
    '''
    from jack_utils import load_json
    PYTHIA_28_ELI5_SFT_TRAIN_10000_Bo16_QUIP = f'{proj_dir()}/gens/eli5_train_subsample_10000_policy_sft_n16_quip.json'
    PYTHIA_28_ELI5_SFT_EVAL_2000_Bo16_QUIP = f'{proj_dir()}/gens/eli5_eval_subsample_2000_policy_2023-10-31_16-35-05_136351_quip.json'
    
    data = defaultdict(lambda: defaultdict(list))
    if separate_eval:
        file_path = PYTHIA_28_ELI5_SFT_TRAIN_10000_Bo16_QUIP if split == 'train' else PYTHIA_28_ELI5_SFT_EVAL_2000_Bo16_QUIP
        data_json = load_json(file_path)
        data_range = range(len(data_json['full_prompts']))
    else:
        data_json = load_json(PYTHIA_28_ELI5_SFT_TRAIN_10000_Bo16_QUIP)
        l = len(data_json['full_prompts'])
        data_range = range(int(0.8*l)) if split == 'train' else range(int(0.8*l), l)
        if not silent: print(f'split {split}, total num data: {l}; data range: {data_range}')

    for i, (prompt, refs, preds, quip_reports) in enumerate(zip(data_json['full_prompts'], data_json['refs'], data_json['pred'], data_json['quip_reports'])):
        if not (i in data_range):
            continue
        quip_score = [q['quip_25_beta'] for q in quip_reports if q['quip_25_beta'] is not None]
        preds_no_null = [preds[i] for i in range(len(preds)) if quip_reports[i]['quip_25_beta'] is not None]
        # never select example whose quip score is null
        chosen = preds_no_null[np.argmax(quip_score)]
        rejected = preds_no_null[np.argmin(quip_score)]
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
        data[prompt]['refs'] = refs

    assert len(data_range) == len(data)
    return data


def get_eli5_according_to(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    '''
    get the subset of eli5 dataset used in acording to paper. this is only about 1500 examples from the validation set
    '''
    def postprocess_chatgpt_according_to(text: str):
        '''
        According to Wikipedia, \"the reason that dogs do not see small dogs as prey likely has to do with their long history of domestication and close relationship with humans. However, dogs may still have an instinctual drive to chase and catch small animals, such as cats, which are similar in their size and movements to prey animals that dogs would have encountered in the wild.\"
        ->
        The reason that dogs do not see small dogs as prey likely has to do with their long history of domestication and close relationship with humans. However, dogs may still have an instinctual drive to chase and catch small animals, such as cats, which are similar in their size and movements to prey animals that dogs would have encountered in the wild.
        '''
        text = text.replace('According to Wikipedia, ', '')
        if text[0] == '\"' and text[-1] == '\"': text = text[1:-1]
        text = text[0].upper() + text[1:]
        return text

    from jack_utils import load_json
    CHATGPT_ELI5_NULL_PROMPT_PATH = '/home/jzhan237/data/according-to/results_for_jack/chatgpt_all_results/chatgpt_all_eli5/results_2023-04-25-18-25-59.json'
    CHATGPT_ELI5_ACCORDING_TO_PROMPT_PATH = '/home/jzhan237/data/according-to/results_for_jack/chatgpt_all_results/chatgpt_all_eli5/results_2023-04-25-18-48-38.json'
    json_null = load_json(CHATGPT_ELI5_NULL_PROMPT_PATH)
    json_accord = load_json(CHATGPT_ELI5_ACCORDING_TO_PROMPT_PATH)
    data = defaultdict(lambda: defaultdict(list))
    l = len(json_null['results']['datasketch']['percent_overlap_list'])

    data_range = range(int(0.8*l)) if split == 'train' else range(int(0.8*l), l)
    if not silent: print(f'total num data: {l}; data range: {data_range}')

    for i in data_range:
        question = json_null['questions'][i]
        prompt = f"{question}\nAnswer:" # follow prompt format of according to paper
        null_response = json_null['pred'][i]
        null_quip = json_null['results']['datasketch']['percent_overlap_list'][i]
        accord_response = postprocess_chatgpt_according_to(json_accord['pred'][i])
        accord_quip = json_accord['results']['datasketch']['percent_overlap_list'][i]
        if accord_quip > null_quip:
            chosen, rejected = accord_response, null_response
        else:
            chosen, rejected = null_response, accord_response
        
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
        data[prompt]['refs'] = json_null['refs'][i]
    
    return data

def get_datapath(name: str, split: str):
    if name == 'eli5_bo16_paired_dq0.1_dl50_pythia28_sft': # NOTE this is problematic because length is not tokenized...
        data_path = f'{proj_dir()}/paired_gens/eli5_train_subsample_10000_policy_sft_n16_quip_dq0.10_dl50.json' if split == 'train' else f'{proj_dir()}/paired_gens/eli5_eval_subsample_2000_policy_sft_n16_quip_dq0.10_dl50.json'
    elif name == 'eli5_bo16_paired_dq0.1_dl20_pythia28_sft':
        data_path = f'{proj_dir()}/paired_gens/eli5_train_subsample_10000_policy_sft_n16_quip_dq0.10_dl20_tokpythia-2.8b.json' if split == 'train' else f'{proj_dir()}/paired_gens/eli5_eval_subsample_2000_policy_sft_n16_quip_dq0.10_dl20_tokpythia-2.8b.json'
    elif name == 'nq_bo16_paired_dq0.1_dl20_pythia28_sft':
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_10000_policy_2023-11-06_20-28-03_056114_quip_dq0.10_dl20_tokpythia-2.8b.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_subsample_2000_policy_2023-11-06_20-42-57_714756_quip_dq0.10_dl20_tokpythia-2.8b.json'
    elif name == 'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsrank_pythia28_sft':
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_policy_2023-11-11_23-35-57_401147_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsrank_tokpythia-2.8b.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_policy_n32_2023-11-12_11-57-11_553868_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsrank_tokpythia-2.8b.json'
    elif name == 'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_pythia28_sft':
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_policy_2023-11-11_23-35-57_401147_quip_bartscore_dq0.10_dl0.10_bsp0.25_tokpythia-2.8b.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_policy_n32_2023-11-12_11-57-11_553868_quip_bartscore_dq0.10_dl0.10_bsp0.25_tokpythia-2.8b.json'
    elif name == 'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_pythia28_sft':
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_policy_2023-11-11_23-35-57_401147_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_tokpythia-2.8b.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_policy_n32_2023-11-12_11-57-11_553868_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_tokpythia-2.8b.json'
    elif name == 'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_shp10_pythia28_sft':
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_policy_2023-11-11_23-35-57_401147_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_shq10_tokpythia-2.8b.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_policy_n32_2023-11-12_11-57-11_553868_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_shq10_tokpythia-2.8b.json'
    elif name == 'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_llama2-7b-chat':
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_policy_2023-11-11_23-35-57_401147_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_Llama-2-7b-chat-hf_n32_2023-11-23_00-13-36_451092_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json'
    elif name == 'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_shp10_llama2-7b-chat':
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_policy_2023-11-11_23-35-57_401147_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_shq10_tokLlama-2-7b-chat-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_Llama-2-7b-chat-hf_n32_2023-11-23_00-13-36_451092_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_shq10_tokLlama-2-7b-chat-hf.json'
    elif name == 'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_llama2-7b_sft':
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_policy_n32_2023-12-02_18-39-38_688332_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_tokllama2-7b-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_policy_n32_2023-12-02_19-13-42_391577_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_tokllama2-7b-hf.json'
    elif name == 'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_shp10_llama2-7b_sft':
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_policy_n32_2023-12-02_18-39-38_688332_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_shq10_tokllama2-7b-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_policy_n32_2023-12-02_19-13-42_391577_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_shq10_tokllama2-7b-hf.json'
    elif name == 'paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp0.25_bsval0.1': # new version, with system prompt
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_Llama-2-7b-chat-hf_n32_2024-01-09_06-54-22_338474_bartscore_quip_dq0.10_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_Llama-2-7b-chat-hf_n32_2024-01-11_23-37-28_899994_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json'
    elif name == 'paired_nq_llama2-7b-chat_bo32+acc32_dq0.1_dl0.1_bsp0.25_bsval0.1': # new version, with system prompt
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_Llama-2-7b-chat-hf_n32_2024-01-09_06/[54-20]COMBINED_bartscore_quip_dq0.10_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_Llama-2-7b-chat-hf_n32_2024-01-11_23-37-28_899994_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json'
    elif name == 'paired_nq_llama2-7b-chat_bo32+acc32_dq0.2_dl0.1_bsp0.25_bsval0.1': # new version, with system prompt
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_Llama-2-7b-chat-hf_n32_2024-01-09_06/[54-20]COMBINED_bartscore_quip_dq0.20_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_Llama-2-7b-chat-hf_n32_2024-01-11_23-37-28_899994_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json'
    elif name == 'paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp0.25_bsval0.1_37k': # new version, with system prompt
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000+l30000_start30000_Llama-2-7b-chat-hf_n32_bartscore_quip_dq0.10_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_Llama-2-7b-chat-hf_n32_2024-01-11_23-37-28_899994_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json'
    elif name == 'paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp0.25_bsval0.1_100k': # new version, with system prompt
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l100000_Llama-2-7b-chat-hf_n32_bartscore_quip_dq0.10_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_Llama-2-7b-chat-hf_n32_2024-01-11_23-37-28_899994_quip_bartscore_dq0.10_dl0.10_bsp0.25_bsval0.10_tokLlama-2-7b-chat-hf.json'
    elif name == 'paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0': # new version, with system prompt
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_Llama-2-7b-chat-hf_n32_2024-01-09_06-54-22_338474_bartscore_quip_dq0.10_dl0.10_bsp1.00_tokLlama-2-7b-chat-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_Llama-2-7b-chat-hf_n32_2024-01-11_23-37-28_899994_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokLlama-2-7b-chat-hf.json'
    elif name == 'paired_wiki_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0':
        data_path = f'{proj_dir()}/paired_gens/00_pfx2_maxlen512_num20000_Llama-2-7b-chat-hf_n32_2024-02-06_14-55-02_899995_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokLlama-2-7b-chat-hf.json' if split == 'train'else f'{proj_dir()}/paired_gens/01_pfx2_maxlen512_num20000_trim2000_Llama-2-7b-chat-hf_n32_2024-02-07_18-39-07_918537_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokLlama-2-7b-chat-hf.json'
    elif name == 'nq_sft':
        data_path = f'{proj_dir()}/data/nq/nq-{"train" if split=="train" else "dev"}.json'
    elif name == 'eli5_sft':
        data_path = f'{proj_dir()}/data/eli5/eli5_train.json' if split == 'train' else f'{proj_dir()}/data/eli5/eli5_eval.json'
    elif name == 'wiki_pile_sft':
        data_path = f'{proj_dir()}/data/wiki_pile_dedup/TRAIN_00.jsonl.wiki_plen32_tlen160_num20000.json' if split == 'train' else f'{proj_dir()}/data/wiki_pile_dedup/EVAL_01.jsonl.wiki_plen32_tlen160_num2000_dedup_TRAIN_00.json'
    elif name == 'paired_wiki_20k_llama2-7b_bo32_dq0.1_dl0.1_bsp1.0':
        data_path = f'{proj_dir()}/paired_gens/TRAIN_00.jsonl.wiki_plen32_tlen160_num20000_Llama-2-7b-hf_n32_2024-02-27_22-09-15_571465_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokLlama-2-7b-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/EVAL_01.jsonl.wiki_plen32_tlen160_num2000_dedup_TRAIN_00_Llama-2-7b-hf_n32_2024-02-27_19-13-23_658875_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokLlama-2-7b-hf.json'
    elif name == 'paired_wiki_20k_llama30b_bo32_dq0.1_dl0.1_bsp1.0':
        data_path = f'{proj_dir()}/paired_gens/TRAIN_00.jsonl.wiki_plen32_tlen160_num20000_llama-30b_n32_2024-03-07_20-57-52_843607_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokllama-30b.json' if split == 'train' else f'{proj_dir()}/paired_gens/EVAL_01.jsonl.wiki_plen32_tlen160_num2000_dedup_TRAIN_00_llama-30b_n32_2024-03-07_17-15-33_518602_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokllama-30b.json'
    elif name == 'paired_wiki_20k_llama2-13b_bo32_dq0.1_dl0.1_bsp1.0':
        data_path = f'{proj_dir()}/paired_gens/TRAIN_00.jsonl.wiki_plen32_tlen160_num20000_Llama-2-13b-hf_n32_2024-03-20_02-13-23_970883_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokLlama-2-13b-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/EVAL_01.jsonl.wiki_plen32_tlen160_num2000_dedup_TRAIN_00_Llama-2-13b-hf_n32_2024-03-21_20-14-03_045165_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokLlama-2-13b-hf.json'
    elif name == 'paired_nq_llama2-7b-chat_bo32_dq0.1_dl999_bsp1.0': # ABLATION of length constraint
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start10000_Llama-2-7b-chat-hf_n32_2024-01-09_06-54-22_338474_bartscore_quip_dq0.10_dl999.00_bsp1.00_tokLlama-2-7b-chat-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_Llama-2-7b-chat-hf_n32_2024-01-11_23-37-28_899994_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokLlama-2-7b-chat-hf.json'
    elif name == 'paired_nq_llama2-7b-chat-quote_bo32_dq0.1_dl0.1_bsp1.0': # new version, with system prompt
        data_path = f'{proj_dir()}/paired_gens/nq-train_subsample_l20000_start30000_policy_n32_2024-04-14_19-00-39_120850_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokLlama-2-7b-chat-hf.json' if split == 'train' else f'{proj_dir()}/paired_gens/nq-dev_Llama-2-7b-chat-hf_n32_2024-01-11_23-37-28_899994_quip_bartscore_dq0.10_dl0.10_bsp1.00_tokLlama-2-7b-chat-hf.json'
    elif name == 'bio1024_factscore_mistral-inst_smallsubset':
        data_path = '/weka/scratch/jzhan237/repos/adversarial-factuality/outputs/paired_gens/bio1024/train_1to200_DUP5_mistral-inst_split1.json' if split == 'train' else '/weka/scratch/jzhan237/repos/adversarial-factuality/outputs/paired_gens/bio1024/dev_1to50_DUP2_mistral-inst.json'
    elif name == 'bio1024_factscore_mistral-inst_mediumsubset':
        data_path = '/weka/scratch/jzhan237/repos/adversarial-factuality/outputs/paired_gens/bio1024/train_1to200_DUP5_mistral-inst.json' if split == 'train' else '/weka/scratch/jzhan237/repos/adversarial-factuality/outputs/paired_gens/bio1024/dev_1to50_DUP2_mistral-inst.json'
    elif name == 'bio1024_factscore_mistral-inst_train1to400subset':
        data_path = '/weka/scratch/jzhan237/repos/adversarial-factuality/outputs/paired_gens/bio1024/train_1to400_DUP5_mistral-inst.json' if split == 'train' else '/weka/scratch/jzhan237/repos/adversarial-factuality/outputs/paired_gens/bio1024/dev_DUP3_mistral-inst.json'
    elif name == 'bio1024_factscore_DATAtrain_401to800_dup5_GENmistral-inst_train1to400subset_beta005':
        data_path = '/weka/scratch/jzhan237/repos/adversarial-factuality/outputs/paired_gens/bio1024/train_401to800_dup5_mistral-inst_train1to400subset_beta005.json' if split == 'train' else '/weka/scratch/jzhan237/repos/adversarial-factuality/outputs/paired_gens/bio1024/dev_DUP3_mistral-inst.json'
        
    else:
        raise ValueError(f"Unknown dataset '{name}'")
    return data_path

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir)
    elif name in [
            'nq_sft',
            'eli5_sft',
            'wiki_pile_sft',
        ]:
        data_path = get_datapath(name, split)
        data = get_sft(data_path, silent=silent, cache_dir=cache_dir)
    elif name == 'eli5':
        data = get_eli5(split, silent=silent, cache_dir=cache_dir)
    elif name == 'eli5_train10000_quip_pair_bo16_maxmin_pythia28_sft':
        data = get_eli5_quip_pair(split, separate_eval=False, silent=silent, cache_dir=cache_dir)
    elif name == 'eli5_train10000_quip_pair_bo16_maxmin_pythia28_sft_separate_eval':
        data = get_eli5_quip_pair(split, separate_eval=True, silent=silent, cache_dir=cache_dir)
    elif name == 'eli5_according_to':
        data = get_eli5_according_to(split, silent=silent, cache_dir=cache_dir)
    elif name in [
        'eli5_bo16_paired_dq0.1_dl50_pythia28_sft', # NOTE this is problematic because length is not tokenized...
        'eli5_bo16_paired_dq0.1_dl20_pythia28_sft',
        'nq_bo16_paired_dq0.1_dl20_pythia28_sft',
        'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsrank_pythia28_sft',
        'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_pythia28_sft',
        'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_pythia28_sft',
        'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_shp10_pythia28_sft',
        'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_llama2-7b-chat',
        'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_shp10_llama2-7b-chat',
        'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_llama2-7b_sft',
        'nq_bo32_paired_dq0.1_dl0.1_bsp0.25_bsval0.1_shp10_llama2-7b_sft',
        'paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp0.25_bsval0.1',
        'paired_nq_llama2-7b-chat_bo32+acc32_dq0.1_dl0.1_bsp0.25_bsval0.1',
        'paired_nq_llama2-7b-chat_bo32+acc32_dq0.2_dl0.1_bsp0.25_bsval0.1',
        'paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp0.25_bsval0.1_37k',
        'paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp0.25_bsval0.1_100k',
        'paired_nq_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0',
        'paired_wiki_llama2-7b-chat_bo32_dq0.1_dl0.1_bsp1.0',
        'paired_wiki_20k_llama2-7b_bo32_dq0.1_dl0.1_bsp1.0',
        'paired_wiki_20k_llama30b_bo32_dq0.1_dl0.1_bsp1.0',
        'paired_wiki_20k_llama2-13b_bo32_dq0.1_dl0.1_bsp1.0',
        'paired_nq_llama2-7b-chat_bo32_dq0.1_dl999_bsp1.0',
        'paired_nq_llama2-7b-chat-quote_bo32_dq0.1_dl0.1_bsp1.0',
        'bio1024_factscore_mistral-inst_smallsubset',
        'bio1024_factscore_mistral-inst_mediumsubset',
        'bio1024_factscore_mistral-inst_train1to400subset',
        'bio1024_factscore_DATAtrain_401to800_dup5_GENmistral-inst_train1to400subset_beta005'
    ]:
        data_path = get_datapath(name, split)
        data = get_paired_gen(data_path, silent=silent)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    # assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        # f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"
    assert all(elt in set(list(data.values())[0].keys()) for elt in ['responses', 'pairs', 'sft_target'])

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       prompt_before: str = '',
                       prompt_after: str = '') -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir).items():
                prompt = prompt_before + prompt + prompt_after # add prompt before and after (JACK 2/7/24)
                flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            # breakpoint()
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True