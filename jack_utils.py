import json
import os
from datetime import datetime
from typing import List, Any
import torch
from vllm import SamplingParams, RequestOutput
import jsonlines

def proj_dir() -> str:
    return os.environ.get("PROJ_DIR", "")

# def send_outpath_to_env(outpath: str):
#     os.environ['PREV_SCRIPT_OUTPATH'] = outpath

def load_json(path: str):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(data, save_path: str, indent=2, overwrite=False):
    if os.path.exists(save_path) and not overwrite:
        print(f'File {save_path} already exists, not overwriting')
    else:
        if len(os.path.dirname(save_path)) > 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
                json.dump(data, f, indent=indent)

def load_jsonl(path: str):
    with jsonlines.open(path) as reader:
        data = [line for line in reader]
    return data

def filename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def path_no_ext(path: str) -> str:
    return os.path.splitext(path)[0]

def add_suffix_before_ext(path: str, suffix: str) -> str:
    a, b = os.path.splitext(path)
    return a + suffix + b

def get_timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

def flatten_lol(lol: List[List[Any]]) -> List[Any]:
    return [x for l in lol for x in l]

def unflatten_to_lol(ls: List[Any], inner_length: int) -> List[List[Any]]:
    return [ls[i:i+inner_length] for i in range(0,len(ls),inner_length)]

def list_of_list_to_csv(lol: List[List[Any]], delimiter: str=',', float_fmt='%.3f'):
    res = ''
    for line in lol:
        res += delimiter.join([float_fmt%x if isinstance(x,float) else str(x) for x in line]) + '\n'
    return res

def load_state_dict(model, state_dict_path):
    state_dict = torch.load(state_dict_path, map_location='cpu')
    step, metrics = state_dict['step_idx'], state_dict['metrics']
    print(f'loading pre-trained weights at step {step} from {state_dict_path} with metrics {json.dumps(metrics, indent=2)}')
    model.load_state_dict(state_dict['state'])
    print('loaded pre-trained weights')
    return model


# f"SamplingParams(n={self.n}, "
#                 f"best_of={self.best_of}, "
#                 f"presence_penalty={self.presence_penalty}, "
#                 f"frequency_penalty={self.frequency_penalty}, "
#                 f"temperature={self.temperature}, "
#                 f"top_p={self.top_p}, "
#                 f"top_k={self.top_k}, "
#                 f"use_beam_search={self.use_beam_search}, "
#                 f"length_penalty={self.length_penalty}, "
#                 f"early_stopping={self.early_stopping}, "
#                 f"stop={self.stop}, "
#                 f"ignore_eos={self.ignore_eos}, "
#                 f"max_tokens={self.max_tokens}, "
#                 f"logprobs={self.logprobs}, "
#                 f"prompt_logprobs={self.prompt_logprobs}, "
#                 f"skip_special_tokens={self.skip_special_tokens})"
def sampling_params_to_dict(sampling_params: SamplingParams):
    return {
        'n': sampling_params.n,
        'best_of': sampling_params.best_of,
        'presence_penalty': sampling_params.presence_penalty,
        'frequency_penalty': sampling_params.frequency_penalty,
        'temperature': sampling_params.temperature,
        'top_p': sampling_params.top_p,
        'top_k': sampling_params.top_k,
        'use_beam_search': sampling_params.use_beam_search,
        'length_penalty': sampling_params.length_penalty,
        'early_stopping': sampling_params.early_stopping,
        'stop': sampling_params.stop,
        'ignore_eos': sampling_params.ignore_eos,
        'max_tokens': sampling_params.max_tokens,
        'logprobs': sampling_params.logprobs,
        'prompt_logprobs': sampling_params.prompt_logprobs,
        'skip_special_tokens': sampling_params.skip_special_tokens,
    }

def request_output_to_list_of_str(request_output: RequestOutput) -> List[str]:
    return [competion_out.text for competion_out in request_output.outputs]