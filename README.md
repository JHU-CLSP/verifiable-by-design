# Verifiable by Design: Aligning Language Models to Quote from Pre-Training Data
Code for NAACL 2025 paper "[Verifiable by Design: Aligning Language Models to Quote from Pre-Training Data](https://arxiv.org/abs/2404.03862)"


## Setup
- Remember to use the newest version of vLLM
- Might need to install azure cli if things don't work: `pip install azure-cli azure-functions azure-identity`
- Dependency: https://github.com/microsoft/controllable-safety-alignment

Make sure to correctly setup environment variables before running all scripts:
- `PROJ_DIR` and `PYTHONPATH` should be the project directory of `controllable-safety-alignment` repo (instead of this one)!
- `MODEL_DIR`, `DATA_DIR`: dedicated directory to store *downloaded* models and data
- `OUTPUT_DIR`: dedicated directory to store *trained* checkpoints

## Steps for running the Quote-Tuning pipeline
1. Format data into huggingface dataset

Example: `data/nq/dev`, `data/nq/train`
```
Dataset({
    features: ['prompt', 'reference'],
    num_rows: 110865
})
```

2. Start vLLM server by using `start_vllm.sh` (make sure port 8000 is not in use or change to a different port! Chaning port requires modifying the last few lines of `model_name_to_endpoints` function in `$PROJ_DIR/src/oai_inference.py`) and run `$PROJ_DIR/src/oai_inference.py` to generate responses on training and dev data.

Example: `run_gen_bo32.sh`

3. Setup quip score server (or use the existing one accesible on internet, 'https://acc2-private-wiki.dataportraits.org/quip'). Make sure the url in `quip_api.py` is correct.

4. Use `run_metric_on_gen.py` to score responses with quip score. See command line arguments there for details.

5. Use `best_of_n_to_paired_gen.py` to produce paired data for DPO. See examples in `best_of_n_to_paired_gen.sh`. Next, convert the produced .json into huggingface dataset via `data_processing/convert_paired_gens_json_to_dataset.py`. Example available in `data_processing/convert_paired_gens_json_to_dataset.sh`

6. Add path of the paired data (converted to HF dataset) as a dataset in the `PAIRED_DATA_DICT` of `$PROJ_DIR/dpo/preference_datasets.py`. Search for 'qt_gemma2-9b-it-inst_bo32_dq0.10_dl0.10-concise_sysp' in the file for example.

7. Conduct DPO training! Use `$PROJ_DIR/dpo/train_qt.sh`. This part of code is based on `https://github.com/eric-mitchell/direct-preference-optimization`. WANDB integration is supported. After training, Convert the trained model .pt file back to huggingface format using `checkpoint_pt_to_hf.py`.

8. Run evaluation using `run_eval_combined.sh`.

## Reference
If you find our work useful, we kindly invite you to cite it:
```
@misc{zhang2025verifiabledesignaligninglanguage,
      title={Verifiable by Design: Aligning Language Models to Quote from Pre-Training Data}, 
      author={Jingyu Zhang and Marc Marone and Tianjian Li and Benjamin Van Durme and Daniel Khashabi},
      year={2025},
      eprint={2404.03862},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.03862}, 
}
```
