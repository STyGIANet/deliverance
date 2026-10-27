cd ~/deliverance

export DELIVERANCE_CKPT_ROOT=/depot/shared/megatron-checkpoints
export DELIVERANCE_DATASET_ROOT=/depot/shared/datasets/moe-inference-datasets

# TP=1, PP=1, EP=8; downloads and converts the checkpoint on first use.
bash deliverance/scripts/run_moe_inference.sh \
  --model mistralai/Mixtral-8x7B-v0.1 \
  --dataset gsm8k \
  --samples 2 \
  --max-new-tokens 8
