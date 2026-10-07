cd ~/deliverance

export DELIVERANCE_MODEL_ROOT=/depot/deliverance/checkpoints
export DELIVERANCE_DATASET_ROOT=/depot/deliverance/datasets/moe-inference-datasets

bash deliverance/scripts/run_moe_inference.sh \
  --model mixtral-8x7b-tp4-pp1-ep1 \
  --dataset gsm8k \
  --parallelism dp \
  --samples 2 \
  --max-new-tokens 8 \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /depot/deliverance/models/mistralai/Mixtral-8x7B-v0.1/tokenizer.model
