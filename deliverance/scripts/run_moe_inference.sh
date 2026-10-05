#!/usr/bin/env bash
set -euo pipefail

# Run the fork's offline inference example on all eight local GPUs.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_root="${DELIVERANCE_MODEL_ROOT:-/models}"
dataset_root="${DELIVERANCE_DATASET_ROOT:-/datasets/moe-inference-datasets}"
gpu_count=8

usage() {
  cat <<'EOF'
Usage: run_moe_inference.sh --model NAME_OR_PATH --dataset NAME_OR_PATH [options]

Required:
  --model PATH|NAME       Megatron checkpoint directory; names resolve under /models
  --dataset PATH|NAME     JSONL prompt file; names resolve under /datasets/moe-inference-datasets

Options:
  --parallelism tp|dp     TP=8, DP=1 (default), or use checkpoint TP with DP
  --tp-size N             DP mode: specify TP for a distributed checkpoint
  --output PATH           Results JSON (default: local/moe-inference-results/<timestamp>.json)
  --tokenizer-model PATH  Override the tokenizer path saved in the checkpoint
  --tokenizer-type TYPE   Tokenizer type for the override (default: HuggingFaceTokenizer)
  --precision bf16|fp16   Compute precision (default: bf16)
  --samples N             Use only the first N prompts
  --max-new-tokens N      Tokens generated per prompt (default: 64)
  --max-seq-length N      Prompt + generation token limit (default: 4096)
  --kv-cache-gb N         GPU KV cache per rank in GiB (default: 4)
  --skip-memory-check     Bypass the legacy-checkpoint host RAM safety check
  --dry-run               Print the command without launching GPUs
  -h, --help              Show this help

Dataset names: gsm8k, ifeval, mtbench, qasper, mixed_short_long,
               mixed_length_sorted. A custom JSONL path also works.
Override roots with DELIVERANCE_MODEL_ROOT and DELIVERANCE_DATASET_ROOT.
For legacy checkpoints, DP mode reads the TP size from checkpoint shards:
TP=4 gives DP=2; TP=2 gives DP=4; TP=1 gives DP=8.
Distributed checkpoints need --tp-size because their TP layout is not
reliably inferable from directory names.
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

need_value() {
  if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
    die "$1 needs a value"
  fi
}

positive_integer() {
  if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
    die "$1 must be a positive integer"
  fi
}

model_arg=""
dataset_arg=""
parallelism="tp"
requested_tp_size=""
output=""
tokenizer_model=""
tokenizer_type=""
precision=bf16
samples=""
max_new_tokens=64
max_seq_length=4096
kv_cache_gb=4
dry_run=false
skip_memory_check=false

while (($#)); do
  case "$1" in
    --model)
      need_value "$@"
      model_arg="$2"
      shift 2
      ;;
    --dataset)
      need_value "$@"
      dataset_arg="$2"
      shift 2
      ;;
    --parallelism)
      need_value "$@"
      parallelism="$2"
      shift 2
      ;;
    --tp-size)
      need_value "$@"
      requested_tp_size="$2"
      shift 2
      ;;
    --output)
      need_value "$@"
      output="$2"
      shift 2
      ;;
    --tokenizer-model)
      need_value "$@"
      tokenizer_model="$2"
      shift 2
      ;;
    --tokenizer-type)
      need_value "$@"
      tokenizer_type="$2"
      shift 2
      ;;
    --precision)
      need_value "$@"
      precision="$2"
      shift 2
      ;;
    --samples)
      need_value "$@"
      samples="$2"
      shift 2
      ;;
    --max-new-tokens)
      need_value "$@"
      max_new_tokens="$2"
      shift 2
      ;;
    --max-seq-length)
      need_value "$@"
      max_seq_length="$2"
      shift 2
      ;;
    --kv-cache-gb)
      need_value "$@"
      kv_cache_gb="$2"
      shift 2
      ;;
    --skip-memory-check)
      skip_memory_check=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (see --help)"
      ;;
  esac
done

[[ -n "$model_arg" ]] || die "--model is required"
[[ -n "$dataset_arg" ]] || die "--dataset is required"
[[ "$parallelism" == tp || "$parallelism" == dp ]] || die "--parallelism must be tp or dp"
if [[ -n "$requested_tp_size" ]]; then
  positive_integer --tp-size "$requested_tp_size"
  case "$requested_tp_size" in
    1|2|4) ;;
    *) die "--tp-size must be 1, 2, or 4 for two or more DP replicas on eight GPUs" ;;
  esac
fi
[[ "$precision" == bf16 || "$precision" == fp16 ]] || die "--precision must be bf16 or fp16"
[[ -z "$samples" ]] || positive_integer --samples "$samples"
positive_integer --max-new-tokens "$max_new_tokens"
positive_integer --max-seq-length "$max_seq_length"
(( max_new_tokens <= max_seq_length )) || die "--max-new-tokens cannot exceed --max-seq-length"
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
if [[ -n "$tokenizer_model" ]]; then
  tokenizer_model="$(realpath "$tokenizer_model")"
fi

# A legacy Megatron checkpoint has one mp_rank_XX directory per TP shard.
# Inspect directory names and file sizes only; do not load multi-GB weights here.
checkpoint_format=""
checkpoint_tp_size=""
checkpoint_bytes=0

inspect_checkpoint() {
  local tracker_file="$model/latest_checkpointed_iteration.txt"
  local checkpoint_step checkpoint_dir shard_dir shard_name shard_file
  local expected_name file_bytes index
  local -a shard_dirs

  [[ -f "$tracker_file" ]] || die "checkpoint tracker not found: $tracker_file"
  checkpoint_step="$(< "$tracker_file")"

  if [[ "$checkpoint_step" == release ]]; then
    checkpoint_dir="$model/release"
  elif [[ "$checkpoint_step" =~ ^[0-9]+$ ]]; then
    printf -v checkpoint_dir '%s/iter_%07d' "$model" "$((10#$checkpoint_step))"
  else
    die "invalid checkpoint iteration in $tracker_file: $checkpoint_step"
  fi
  [[ -d "$checkpoint_dir" ]] || die "checkpoint iteration directory not found: $checkpoint_dir"

  # Megatron's distributed checkpoints have metadata instead of mp_rank_XX files.
  if [[ -f "$checkpoint_dir/metadata.json" || -f "$checkpoint_dir/.metadata" ]]; then
    checkpoint_format=distributed
    return
  fi

  shard_dirs=("$checkpoint_dir"/mp_rank_*)
  [[ -e "${shard_dirs[0]}" ]] || die "checkpoint format not recognized in $checkpoint_dir"

  for shard_dir in "${shard_dirs[@]}"; do
    [[ -d "$shard_dir" ]] || die "expected a shard directory: $shard_dir"
    shard_name="${shard_dir##*/}"
    [[ "$shard_name" =~ ^mp_rank_[0-9][0-9]$ ]] || \
      die "unsupported shard layout $shard_name (this launcher requires PP=1 and EP=1)"
  done

  checkpoint_tp_size="${#shard_dirs[@]}"
  for ((index = 0; index < checkpoint_tp_size; index++)); do
    printf -v expected_name 'mp_rank_%02d' "$index"
    shard_file="$checkpoint_dir/$expected_name/model_optim_rng.pt"
    [[ -s "$shard_file" ]] || die "missing or empty checkpoint shard: $shard_file"
    file_bytes="$(stat -c %s "$shard_file")"
    checkpoint_bytes=$((checkpoint_bytes + file_bytes))
  done
  checkpoint_format=legacy
}

inspect_checkpoint

if [[ "$parallelism" == tp ]]; then
  [[ -z "$requested_tp_size" ]] || die "--tp-size is only used with --parallelism dp"
  tp_size="$gpu_count"
elif [[ -n "$requested_tp_size" ]]; then
  tp_size="$requested_tp_size"
elif [[ "$checkpoint_format" == legacy ]]; then
  tp_size="$checkpoint_tp_size"
else
  die "cannot infer TP from a distributed checkpoint; pass --tp-size 1, 2, or 4"
fi

(( gpu_count % tp_size == 0 )) || die "TP=$tp_size does not divide $gpu_count GPUs"
dp_size=$((gpu_count / tp_size))

if [[ "$checkpoint_format" == legacy && "$tp_size" != "$checkpoint_tp_size" ]]; then
  die "checkpoint is TP=$checkpoint_tp_size but launch requested TP=$tp_size; legacy checkpoints cannot be resharded while loading"
fi
if [[ "$parallelism" == dp && "$dp_size" -lt 2 ]]; then
  die "DP mode needs at least two replicas; this checkpoint uses TP=$tp_size on all $gpu_count GPUs"
fi

# Each DP replica loads its own full copy of a legacy checkpoint into CPU RAM.
# Reject runs near the host's available RAM; the TP=1 Mixtral run OOM-killed a
# worker on this machine. This is a conservative preflight, not a guarantee.
check_memory_headroom() {
  local available_kib available_bytes estimated_bytes
  local cgroup_limit cgroup_current cgroup_available

  [[ "$checkpoint_format" == legacy && "$dp_size" -gt 1 ]] || return 0
  if "$skip_memory_check"; then
    printf 'Warning: host RAM safety check skipped.\n' >&2
    return
  fi

  available_kib="$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)"
  [[ "$available_kib" =~ ^[0-9]+$ ]] || die "could not read available host RAM"
  available_bytes=$((available_kib * 1024))

  if [[ -r /sys/fs/cgroup/memory.max && -r /sys/fs/cgroup/memory.current ]]; then
    cgroup_limit="$(< /sys/fs/cgroup/memory.max)"
    cgroup_current="$(< /sys/fs/cgroup/memory.current)"
    if [[ "$cgroup_limit" =~ ^[0-9]+$ && "$cgroup_current" =~ ^[0-9]+$ ]]; then
      cgroup_available=$((cgroup_limit - cgroup_current))
      if (( cgroup_available < available_bytes )); then
        available_bytes="$cgroup_available"
      fi
    fi
  fi

  estimated_bytes=$((checkpoint_bytes * dp_size))
  if (( estimated_bytes * 10 > available_bytes * 8 )); then
    if "$dry_run"; then
      printf 'Warning: estimated checkpoint load is %s GiB for DP=%s; available RAM is %s GiB.\n' \
        "$((estimated_bytes / 1073741824))" "$dp_size" "$((available_bytes / 1073741824))" >&2
    else
      die "estimated checkpoint load is $((estimated_bytes / 1073741824)) GiB for DP=$dp_size, over 80% of $((available_bytes / 1073741824)) GiB available RAM. Use a higher-TP checkpoint or --skip-memory-check if you accept the OOM risk"
    fi
  fi
}

check_memory_headroom

if [[ -z "$output" ]]; then
  output="$repo_dir/local/moe-inference-results/$(date +%Y%m%d-%H%M%S)-$$.json"
elif [[ "$output" != /* ]]; then
  output="$(pwd)/$output"
fi
[[ ! -e "$output" && ! -L "$output" ]] || die "output already exists: $output"

cmd=(
  python -m torch.distributed.run --standalone "--nproc-per-node=$gpu_count" -m examples.inference.offline_inference
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
if [[ -n "$samples" ]]; then
  cmd+=(--prompt-file-num-truncate "$samples")
fi
if [[ -n "$tokenizer_model" ]]; then
  cmd+=(
    --no-use-tokenizer-model-from-checkpoint-args
    --tokenizer-type "${tokenizer_type:-HuggingFaceTokenizer}"
    --tokenizer-model "$tokenizer_model"
  )
fi

printf 'Model: %s\nDataset: %s\nCheckpoint: %s\nParallelism: TP=%s DP=%s\nOutput: %s\n' \
  "$model" "$dataset" "$checkpoint_format" "$tp_size" "$dp_size" "$output"
if "$dry_run"; then
  printf 'Command: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

command -v python >/dev/null || die "python not found; activate the container's Python environment"
cd "$repo_dir"
mkdir -p "$(dirname "$output")"
exec "${cmd[@]}"
