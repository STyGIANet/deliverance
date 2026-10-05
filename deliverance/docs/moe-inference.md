# Eight-GPU MoE inference

First-time container setup: [container-setup.md](container-setup.md).

Run these commands **inside the container**, from the activated Python environment. Mount shared datasets at `/datasets` and converted Megatron checkpoints at `/models` (or keep a checkpoint in `models/` under the checkout).

```bash
cd /workspace/deliverance
source .venv/bin/activate
bash deliverance/scripts/run_moe_inference.sh \
  --model models/mixtral-8x7b-tp4-pp1-ep1 \
  --dataset gsm8k \
  --parallelism dp \
  --samples 2 \
  --max-new-tokens 8 \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /models/mistralai/.Mixtral-8x7B-v0.1.incoming/tokenizer.model \
  --dry-run
```

Remove `--dry-run` to launch. The launcher accepts a model directory name under `/models` or a checkpoint path. It accepts a dataset name under `/datasets/moe-inference-datasets` or a path to a JSONL file with a `text` field on each line. Available names are `gsm8k`, `ifeval`, `mtbench`, `qasper`, `mixed_short_long`, and `mixed_length_sorted`.

`--parallelism tp` shards one model across eight GPUs (TP=8, DP=1). `--parallelism dp` reads the TP size from a **legacy** checkpoint's shard directories and uses the remaining GPUs for data-parallel replicas: TP=4 becomes DP=2, TP=2 becomes DP=4, and TP=1 becomes DP=8. A TP=8 checkpoint cannot provide DP replicas. For distributed checkpoints, specify `--tp-size 1`, `2`, or `4` with `--parallelism dp`; the directory layout does not reliably expose the saved TP size. Both modes use PP=1 and EP=1. Coordinator mode routes prompts across DP replicas, so results are written once. These modes do not measure pipeline-parallel bubbles.

The script checks that legacy checkpoint shards match the requested topology. It also blocks a multi-replica legacy run when the estimated checkpoint load exceeds 80% of available host RAM. This is a conservative CPU-memory check, not a GPU-memory guarantee; use `--skip-memory-check` only if you accept the OOM risk. The TP=1 Mixtral checkpoint with DP=8 previously OOM-killed a worker on this machine.

The default output is a timestamped JSON file under `local/moe-inference-results/`. Use `--output PATH` to choose another location; an existing file is never overwritten. `--max-new-tokens`, `--max-seq-length`, and `--kv-cache-gb` tune generation and GPU memory use. Run `bash deliverance/scripts/run_moe_inference.sh --help` for all options.

Hugging Face safetensors are **not** Megatron checkpoints. The shared `/srv/deliverance/models/mistralai/.Mixtral-8x7B-v0.1.incoming` directory contains HF-format weights and cannot be passed directly to this script; convert a model to a Megatron checkpoint first.
