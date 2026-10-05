# Eight-GPU MoE inference

First-time container setup: [container-setup.md](container-setup.md).

Run these commands **inside the container**, from the activated Python environment. Mount shared datasets at `/datasets` and converted Megatron checkpoints at `/models`; the checkout should be at `/workspace/deliverance`.

```bash
cd /workspace/deliverance
bash deliverance/scripts/run_moe_inference.sh   --model my-megatron-checkpoint --dataset gsm8k   --parallelism tp   --tokenizer-type Llama2Tokenizer   --tokenizer-model /models/mistralai/.Mixtral-8x7B-v0.1.incoming/tokenizer.model   --samples 2
```

The Bash launcher accepts a model directory name under `/models` or an absolute checkpoint path. It accepts a dataset name under `/datasets/moe-inference-datasets` or a path to a JSONL file with a `text` field on each line. Available names are `gsm8k`, `ifeval`, `mtbench`, `qasper`, `mixed_short_long`, and `mixed_length_sorted`.

`--parallelism tp` shards one model across eight GPUs (TP=8, DP=1). `--parallelism dp` runs eight replicas (TP=1, DP=8); the full model and its KV cache must fit on **each** GPU. Both modes use PP=1 and EP=1. Coordinator mode routes prompts across DP replicas, so results are written once. These modes do not measure pipeline-parallel bubbles.

For a short test or a tokenizer path that differs from the checkpoint's saved path:

```bash
bash deliverance/scripts/run_moe_inference.sh --model my-megatron-checkpoint --dataset qasper --parallelism dp --samples 8 --tokenizer-type Llama2Tokenizer   --tokenizer-model /models/mistralai/.Mixtral-8x7B-v0.1.incoming/tokenizer.model --dry-run
```

Remove `--dry-run` to launch. The default output is a timestamped JSON file under `local/moe-inference-results/`. Use `--output PATH` to choose another location; an existing file is never overwritten. `--max-new-tokens`, `--max-seq-length`, and `--kv-cache-gb` tune generation and memory use. Run `bash deliverance/scripts/run_moe_inference.sh --help` for all options.

Hugging Face safetensors are **not** Megatron checkpoints. The shared `/srv/deliverance/models/mistralai/.Mixtral-8x7B-v0.1.incoming` directory contains HF-format weights and cannot be passed directly to this script; convert a model to a Megatron checkpoint first. A checkpoint saved with another TP size must also support repartitioning for the selected mode. The launcher validates paths and prints the command, but it has not been exercised against a converted checkpoint in this checkout.
