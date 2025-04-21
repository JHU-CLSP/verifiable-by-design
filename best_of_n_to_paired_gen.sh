# gemma2-9b-it 20k
python best_of_n_to_paired_gen.py data/nq/dev_model-gemma2-9b-it_concise_gr_bestof32_quip.json data/nq/train-l20000_start10000_model-gemma2-9b-it_concise_gr_bestof32_quip.json --tokenizer $weka/models/gemma2-9b-it --output_dir paired_gens/gemma2-9b-it

# starling-7b-beta 20k
python best_of_n_to_paired_gen.py data/nq/dev_model-starling-7b-beta_concise_gr_bestof32_quip.json data/nq/train-l20000_start10000_model-starling-7b-beta_concise_gr_bestof32_quip.json --tokenizer $weka/models/starling-7b-beta --output_dir paired_gens/starling-7b-beta

# llama3.1-8b all training data NEGATED
python best_of_n_to_paired_gen.py data/nq/dev_model-llama3.1-8b-instruct_concise_gr_bestof32_quip.json data/nq/train_model-llama3.1-8b-instruct_concise_gr_bestof32_quip.json --negate_score --tokenizer $weka/models/llama3.1-8b-instruct --output_dir paired_gens/llama3.1-8b-instruct

# llama3.1-8b newsqa1k NEGATED
python best_of_n_to_paired_gen.py data/newsqa1k/dev_model-llama3.1-8b-instruct_concise_gr_bestof32_quip.json data/newsqa1k/train_model-llama3.1-8b-instruct_concise_gr_bestof32_quip.json --negate_score --tokenizer $weka/models/llama3.1-8b-instruct --output_dir paired_gens/newsqa1k

# llama3.1-8b newsspan-fullft newsspan NEGATED, 10 pairs
python best_of_n_to_paired_gen.py data/newsspan1k/dev_model-llama3.1-8b-newsspan-fullft-ep2_complete_gr_bestof32_quip.json data/newsspan1k/train_model-llama3.1-8b-newsspan-fullft-ep2_complete_gr_bestof32_quip.json --negate_score --num_pairs 10 --tokenizer $weka/models/llama3.1-8b-instruct --output_dir paired_gens/newsspan1k

# llama3.1-8b newsspan-fullft newsspan NEGATED, 20 pairs, delta_quip 0.2
python best_of_n_to_paired_gen.py data/newsspan1k/dev_model-llama3.1-8b-newsspan-fullft-ep2_complete_gr_bestof32_quip.json data/newsspan1k/train_model-llama3.1-8b-newsspan-fullft-ep2_complete_gr_bestof32_quip.json --negate_score --num_pairs 20 --tokenizer $weka/models/llama3.1-8b-instruct --output_dir paired_gens/newsspan1k --delta_quip 0.2

# llama3.1-8b newsspan_blockdev NEGATED
python best_of_n_to_paired_gen.py data/newsspan_blockdev/dev_model-llama3.1-8b-newsspan-fullft-ep2_complete_gr_bestof32_l200_quip.json data/newsspan_blockdev/train_model-llama3.1-8b-newsspan-fullft-ep2_complete_gr_bestof32_l200_combined_quip.json --negate_score --tokenizer $weka/models/llama3.1-8b-instruct --output_dir paired_gens/newsspan_blockdev

