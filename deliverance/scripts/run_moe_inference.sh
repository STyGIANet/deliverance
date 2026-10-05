#!/usr/bin/env bash
set -euo pipefail

# Run the fork's offline inference example on all eight local GPUs.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_root="${DELIVERANCE_MODEL_ROOT:-/models}"
dataset_root="${DELIVERANCE_DATASET_ROOT:-/datasets/moe-inference-datasets}"

usage() {
  cat <<'EOF'
Usage: run_moe_inference.sh --model NAME_OR_PATH --dataset NAME_OR_PATH [options]

Required:
  --model PATH|NAME       Megatron checkpoint directory; names resolve under /models
  --dataset PATH|NAME     JSONL prompt file; names resolve under /datasets/moe-inference-datasets

Options:
  --parallelism tp|dp     TP=8, DP=1 (default), or TP=1, DP=8
  --output PATH           Results JSON (default: local/moe-inference-results/<timestamp>.json)
  --tokenizer-model PATH  Override the tokenizer path saved in the checkpoint
  --tokenizer-type TYPE   Tokenizer type for the override (default: HuggingFaceTokenizer)
  --precision bf16|fp16   Compute precision (default: bf16)
  --samples N             Use only the first N prompts
  --max-new-tokens N      Tokens generated per prompt (default: 64)
  --max-seq-length N      Prompt + generation token limit (default: 4096)
  --kv-cache-gb N         GPU KV cache per rank in GiB (default: 4)
  --dry-run               Print the command without launching GPUs
  -h, --help              Show this help

Dataset names: gsm8k, ifeval, mtbench, qasper, mixed_short_long,
               mixed_length_sorted. A custom JSONL path also works.
Override roots with DELIVERANCE_MODEL_ROOT and DELIVERANCE_DATASET_ROOT.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 needs a value"; }
positive_integer() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"; }

model_arg=""
dataset_arg=""
parallelism="tp"
output=""
tokenizer_model=""
tokenizer_type=""
precision=bf16
samples=""
max_new_tokens=64
max_seq_length=4096
kv_cache_gb=4
dry_run=false

while (($#)); do
  case "$1" in
    --model|--dataset|--parallelism|--output|--tokenizer-model|--tokenizer-type|--precision|--samples|--max-new-tokens|--max-seq-length|--kv-cache-gb)
      need_value "$@"
      case "$1" in
        --model) model_arg="$2" ;;
        --dataset) dataset_arg="$2" ;;
        --parallelism) parallelism="$2" ;;
        --output) output="$2" ;;
        --tokenizer-model) tokenizer_model="$2" ;;
        --tokenizer-type) tokenizer_type="$2" ;;
        --precision) precision="$2" ;;
        --samples) samples="$2" ;;
        --max-new-tokens) max_new_tokens="$2" ;;
        --max-seq-length) max_seq_length="$2" ;;
        --kv-cache-gb) kv_cache_gb="$2" ;;
      esac
      shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

[[ -n "$model_arg" ]] || die "--model is required"
[[ -n "$dataset_arg" ]] || die "--dataset is required"
[[ "$parallelism" == tp || "$parallelism" == dp ]] || die "--parallelism must be tp or dp"
[[ "$precision" == bf16 || "$precision" == fp16 ]] || die "--precision must be bf16 or fp16"
[[ -z "$samples" ]] || positive_integer --samples "$samples"
positive_integer --max-new-tokens "$max_new_tokens"
positive_integer --max-seq-length "$max_seq_length"
[[ "$kv_cache_gb" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--kv-cache-gb must be a nonnegative number"
[[ -z "$tokenizer_type" || -n "$tokenizer_model" ]] || die "--tokenizer-type requires --tokenizer-model"

case "$dataset_arg" in
  mtbench) dataset_arg=mtbench_first_turn ;;
  qasper) dataset_arg=longbench_qasper ;;
esac
if [[ -f "$dataset_arg" ]]; then
  dataset="$dataset_arg"
elif [[ -f "$dataset_root/$dataset_arg" ]]; then
  dataset="$dataset_root/$dataset_arg"
elif [[ -f "$dataset_root/$dataset_arg.jsonl" ]]; then
  dataset="$dataset_root/$dataset_arg.jsonl"
else
  die "dataset not found: $dataset_arg (looked under $dataset_root)"
fi
[[ -s "$dataset" ]] || die "dataset is empty: $dataset"
dataset="$(realpath "$dataset")"

if [[ -d "$model_arg" ]]; then
  model="$model_arg"
elif [[ -d "$model_root/$model_arg" ]]; then
  model="$model_root/$model_arg"
else
  die "model directory not found: $model_arg (looked under $model_root)"
fi
if [[ -f "$model/config.json" ]] && compgen -G "$model/*.safetensors" >/dev/null; then
  die "$model contains Hugging Face weights, not a Megatron checkpoint. Convert it before using --model."
fi
model="$(realpath "$model")"
if [[ -n "$tokenizer_model" && ! -e "$tokenizer_model" ]]; then
  die "tokenizer model not found: $tokenizer_model"
fi
if [[ -n "$tokenizer_model" ]]; then tokenizer_model="$(realpath "$tokenizer_model")"; fi

if [[ "$parallelism" == tp ]]; then tp_size=8; else tp_size=1; fi
if [[ -z "$output" ]]; then
  output="$repo_dir/local/moe-inference-results/$(date +%Y%m%d-%H%M%S)-$$.json"
elif [[ "$output" != /* ]]; then
  output="$(pwd)/$output"
fi
[[ ! -e "$output" ]] || die "output already exists: $output"

cmd=(
  python -m torch.distributed.run --standalone --nproc-per-node=8 -m examples.inference.offline_inference
  --load "$model"
  --use-checkpoint-args --auto-detect-ckpt-format
  --tensor-model-parallel-size "$tp_size"
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 1
  --micro-batch-size 1
  --moe-token-dispatcher-type alltoall
  --dist-ckpt-strictness log_unexpected
  "--$precision"
  --use-coordinator
  --prompt-file "$dataset"
  --output-path "$output"
  --num-tokens-to-generate "$max_new_tokens"
  --inference-max-seq-length "$max_seq_length"
  --inference-dynamic-batching-buffer-size-gb "$kv_cache_gb"
  --incoming-requests-per-sec -1
)
if [[ -n "$samples" ]]; then cmd+=(--prompt-file-num-truncate "$samples"); fi
if [[ -n "$tokenizer_model" ]]; then
  cmd+=(--no-use-tokenizer-model-from-checkpoint-args --tokenizer-type "${tokenizer_type:-HuggingFaceTokenizer}" --tokenizer-model "$tokenizer_model")
fi

printf 'Model: %s\nDataset: %s\nParallelism: TP=%s DP=%s\nOutput: %s\n' \
  "$model" "$dataset" "$tp_size" "$((8 / tp_size))" "$output"
if "$dry_run"; then
  printf 'Command: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

command -v torchrun >/dev/null || die "torchrun not found; activate the container's Python environment"
cd "$repo_dir"
mkdir -p "$(dirname "$output")"
exec "${cmd[@]}"
