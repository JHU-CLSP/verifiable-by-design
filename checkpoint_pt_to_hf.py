from jack_utils import load_state_dict, path_no_ext
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='base model huggingface name or path, e.g., meta-llama/Llama-3.1-8B or a local path to model')
    parser.add_argument('checkpoint_path', type=str, help='path to the checkpoint .pt file')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name) # , device_map="auto"
    model = load_state_dict(model, args.checkpoint_path)
    model.save_pretrained(path_no_ext(args.checkpoint_path))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(path_no_ext(args.checkpoint_path))