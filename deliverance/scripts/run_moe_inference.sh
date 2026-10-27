#!/usr/bin/env bash
set -euo pipefail

# Run the fork's offline inference example on the local GPUs, preparing the
# Megatron checkpoint from Hugging Face first when needed.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# DELIVERANCE_MODEL_ROOT is the older name for the checkpoint root.
ckpt_root="${DELIVERANCE_CKPT_ROOT:-${DELIVERANCE_MODEL_ROOT:-/depot/shared/megatron-checkpoints}}"
dataset_root="${DELIVERANCE_DATASET_ROOT:-/depot/shared/datasets/moe-inference-datasets}"

usage() {
  cat <<'EOF'
Usage: run_moe_inference.sh --model MODEL --dataset NAME_OR_PATH [options] [-- MEGATRON_ARGS...]

Required:
  --model MODEL           One of, in lookup order:
                            - a Megatron checkpoint directory path
                            - a checkpoint name under $DELIVERANCE_CKPT_ROOT
                            - a Hugging Face repo ID (org/name): uses
                              <ckpt_root>/<name>-tp<T>-pp<P>-ep<E>, downloading
                              and converting it with prepare_checkpoint.sh first
                              if it does not exist
  --dataset PATH|NAME     JSONL prompt file; names resolve under $DELIVERANCE_DATASET_ROOT

Parallelism:
  --tp N                  Tensor parallel size (default: 1)
  --pp N                  Pipeline parallel size (default: 1)
  --ep N                  Expert parallel size (default: 8)
  --gpus N                GPUs / processes on this node (default: 8)

Options:
  --dispatcher TYPE       MoE token dispatcher (default: alltoall)
  --timing-dir DIR        Enable MoE phase timing, writing rank*.jsonl to DIR
  --output PATH           Results JSON (default: local/moe-inference-results/<timestamp>.json)
  --tokenizer-model PATH  Tokenizer file; defaults to the one recorded by
                          prepare_checkpoint.sh, otherwise the checkpoint's own
  --tokenizer-type TYPE   Tokenizer type for --tokenizer-model (default: Llama2Tokenizer)
  --precision bf16|fp16   Compute precision (default: bf16)
  --samples N             Use only the first N prompts
  --max-new-tokens N      Tokens generated per prompt (default: 64)
  --max-seq-length N      Prompt + generation token limit (default: 4096)
  --kv-cache-gb N         GPU KV cache per rank in GiB (default: 4)
  --dry-run               Print what would run (including any download or
                          conversion) without doing it
  -h, --help              Show this help

Arguments after `--` are passed to Megatron unchanged.

Each run writes <output>.manifest.json next to the results (command, git
commit, dirty flag, environment) and <output>.diff when the tree is dirty.

Dataset names: gsm8k, ifeval, mtbench, qasper, mixed_short_long,
               mixed_length_sorted. A custom JSONL path also works.
Legacy (torch format) checkpoints cannot be resharded while loading, so their
shard layout must match --tp/--pp/--ep.
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
tp_size=1
pp_size=1
ep_size=8
gpu_count=8
dispatcher=alltoall
timing_dir=""
output=""
tokenizer_model=""
tokenizer_type=""
precision=bf16
samples=""
max_new_tokens=64
max_seq_length=4096
kv_cache_gb=4
dry_run=false
extra_args=()

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
    --tp)
      need_value "$@"
      tp_size="$2"
      shift 2
      ;;
    --pp)
      need_value "$@"
      pp_size="$2"
      shift 2
      ;;
    --ep)
      need_value "$@"
      ep_size="$2"
      shift 2
      ;;
    --gpus)
      need_value "$@"
      gpu_count="$2"
      shift 2
      ;;
    --dispatcher)
      need_value "$@"
      dispatcher="$2"
      shift 2
      ;;
    --timing-dir)
      need_value "$@"
      timing_dir="$2"
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
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    *)
      die "unknown option: $1 (see --help)"
      ;;
  esac
done

[[ -n "$model_arg" ]] || die "--model is required"
[[ -n "$dataset_arg" ]] || die "--dataset is required"
positive_integer --tp "$tp_size"
positive_integer --pp "$pp_size"
positive_integer --ep "$ep_size"
positive_integer --gpus "$gpu_count"
[[ "$precision" == bf16 || "$precision" == fp16 ]] || die "--precision must be bf16 or fp16"
[[ -z "$samples" ]] || positive_integer --samples "$samples"
positive_integer --max-new-tokens "$max_new_tokens"
positive_integer --max-seq-length "$max_seq_length"
(( max_new_tokens <= max_seq_length )) || die "--max-new-tokens cannot exceed --max-seq-length"
[[ "$kv_cache_gb" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--kv-cache-gb must be a nonnegative number"
[[ -z "$tokenizer_type" || -n "$tokenizer_model" ]] || die "--tokenizer-type requires --tokenizer-model"
command -v python >/dev/null || die "python not found; source .venv/bin/activate first"

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

# Resolve --model to a checkpoint directory, preparing it from HF if needed.
model_source=""
if [[ -d "$model_arg" ]]; then
  model="$model_arg"
  model_source=path
elif [[ -d "$ckpt_root/$model_arg" ]]; then
  model="$ckpt_root/$model_arg"
  model_source=name
elif [[ "$model_arg" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  model_name="${model_arg##*/}"
  model="$ckpt_root/${model_name,,}-tp$tp_size-pp$pp_size-ep$ep_size"
  model_source=hf
else
  die "model not found: $model_arg (not a directory, not under $ckpt_root, not an org/name HF repo ID)"
fi

prepare_cmd=()
if [[ "$model_source" == hf && ! -d "$model" ]]; then
  prepare_cmd=(
    "$repo_dir/deliverance/scripts/prepare_checkpoint.sh"
    --hf-repo "$model_arg" --tp "$tp_size" --pp "$pp_size" --ep "$ep_size"
  )
  printf 'Checkpoint %s does not exist; preparing it from %s\n' "$model" "$model_arg"
  if "$dry_run"; then
    DELIVERANCE_CKPT_ROOT="$ckpt_root" "${prepare_cmd[@]}" --dry-run
    printf '(dry run: skipping checks that need the converted checkpoint)\n'
  else
    DELIVERANCE_CKPT_ROOT="$ckpt_root" "${prepare_cmd[@]}"
    [[ -d "$model" ]] || die "prepare_checkpoint.sh finished but $model does not exist"
  fi
fi

if [[ -d "$model" ]]; then
  if [[ -f "$model/config.json" ]] && compgen -G "$model/*.safetensors" >/dev/null; then
    die "$model contains Hugging Face weights, not a Megatron checkpoint. Pass the HF repo ID instead to convert it."
  fi
  model="$(realpath "$model")"
fi

# Tokenizer: explicit flag, else the sidecar written by prepare_checkpoint.sh,
# else whatever the checkpoint args recorded.
source_json="$model/deliverance_source.json"
if [[ -z "$tokenizer_model" && -s "$source_json" ]]; then
  read -r tokenizer_type tokenizer_model < <(python - "$source_json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(s["tokenizer_type"], s["tokenizer_model"])
EOF
  )
elif [[ -z "$tokenizer_model" && "$model_source" == hf ]] && "$dry_run"; then
  tokenizer_type=Llama2Tokenizer
  tokenizer_model="<hub snapshot of $model_arg>/tokenizer.model"
fi
if [[ -n "$tokenizer_model" && "$tokenizer_model" != \<* ]]; then
  [[ -e "$tokenizer_model" ]] || die "tokenizer model not found: $tokenizer_model"
  # -s: keep the readable hub snapshot path rather than its blobs/<hash> target.
  tokenizer_model="$(realpath -s "$tokenizer_model")"
fi

# Inspect the checkpoint's directory names only; do not load multi-GB weights
# here. A legacy checkpoint has one mp_rank_TT[_PPP][_EEE] directory per
# model-parallel shard.
checkpoint_format=""

inspect_checkpoint() {
  local tracker_file="$model/latest_checkpointed_iteration.txt"
  local checkpoint_step checkpoint_dir shard tp pp ep found expected=0

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

  # Megatron's distributed checkpoints have metadata instead of mp_rank_* files
  # and are resharded at load time.
  if [[ -f "$checkpoint_dir/metadata.json" || -f "$checkpoint_dir/.metadata" ]]; then
    checkpoint_format=distributed
    return
  fi

  found="$(find "$checkpoint_dir" -mindepth 1 -maxdepth 1 -type d -name 'mp_rank_*' | wc -l)"
  (( found > 0 )) || die "checkpoint format not recognized in $checkpoint_dir"

  for ((tp = 0; tp < tp_size; tp++)); do
    for ((pp = 0; pp < pp_size; pp++)); do
      for ((ep = 0; ep < ep_size; ep++)); do
        printf -v shard 'mp_rank_%02d' "$tp"
        (( pp_size == 1 )) || printf -v shard '%s_%03d' "$shard" "$pp"
        (( ep_size == 1 )) || printf -v shard '%s_%03d' "$shard" "$ep"
        [[ -s "$checkpoint_dir/$shard/model_optim_rng.pt" ]] ||
          die "checkpoint layout does not match TP=$tp_size PP=$pp_size EP=$ep_size: missing $shard ($found shard directories present); legacy checkpoints cannot be resharded while loading"
        expected=$((expected + 1))
      done
    done
  done
  (( found == expected )) ||
    die "checkpoint has $found shard directories but TP=$tp_size PP=$pp_size EP=$ep_size expects $expected"
  checkpoint_format=legacy
}

[[ ! -d "$model" ]] || inspect_checkpoint

if [[ -z "$output" ]]; then
  output="$repo_dir/local/moe-inference-results/$(date +%Y%m%d-%H%M%S)-$$.json"
elif [[ "$output" != /* ]]; then
  output="$(pwd)/$output"
fi
[[ ! -e "$output" && ! -L "$output" ]] || die "output already exists: $output"
manifest="${output%.json}.manifest.json"
diff_file="${output%.json}.diff"

if [[ -n "$timing_dir" ]]; then
  [[ "$timing_dir" == /* ]] || timing_dir="$(pwd)/$timing_dir"
  ! compgen -G "$timing_dir/rank*.jsonl" >/dev/null || die "timing directory already has rank files: $timing_dir"
fi

cmd=(
  python -m torch.distributed.run --standalone "--nproc-per-node=$gpu_count" -m examples.inference.offline_inference
  --load "$model"
  --use-checkpoint-args --auto-detect-ckpt-format
  --tensor-model-parallel-size "$tp_size"
  --pipeline-model-parallel-size "$pp_size"
  --expert-model-parallel-size "$ep_size"
  --micro-batch-size 1
  --moe-token-dispatcher-type "$dispatcher"
  # Pinned so runs compare the same layer implementation; inference_optimized
  # would swap in a different MoE dispatch path.
  --transformer-impl transformer_engine
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
  # Without this flag the checkpoint's tokenizer args override ours.
  cmd+=(
    --no-use-tokenizer-model-from-checkpoint-args
    --tokenizer-type "${tokenizer_type:-Llama2Tokenizer}"
    --tokenizer-model "$tokenizer_model"
  )
fi
cmd+=("${extra_args[@]}")

printf 'Model: %s (%s)\nCheckpoint: %s\nDataset: %s\n' \
  "$model" "$model_source" "${checkpoint_format:-not yet prepared}" "$dataset"
printf 'Parallelism: TP=%s PP=%s EP=%s on %s GPUs\n' \
  "$tp_size" "$pp_size" "$ep_size" "$gpu_count"
printf 'Dispatcher: %s\nTokenizer: %s\nTiming: %s\nOutput: %s\n' \
  "$dispatcher" "${tokenizer_model:-from checkpoint args}" "${timing_dir:-off}" "$output"
if "$dry_run"; then
  printf 'Command: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

cd "$repo_dir"
mkdir -p "$(dirname "$output")"
if [[ -n "$timing_dir" ]]; then
  mkdir -p "$timing_dir"
  export MOE_PHASE_TIMING_DIR="$timing_dir"
fi

# Record what produced the results, for reproducibility.
git_dirty=false
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  git_dirty=true
  git diff HEAD > "$diff_file"
fi
python - "$manifest" "$git_dirty" "$diff_file" "$model_source" "$model_arg" "$model" \
  "$tp_size" "$pp_size" "$ep_size" "$gpu_count" "${cmd[@]}" <<'EOF'
import datetime, importlib.metadata as md, json, os, socket, subprocess, sys
(out, dirty, diff_file, source, model_arg, model,
 tp, pp, ep, gpus), cmd = sys.argv[1:11], sys.argv[11:]
def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
def ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None
env_keys = ("CUDA_", "NCCL_", "MOE_", "DELIVERANCE_", "HF_HUB_", "TORCH_", "PYTORCH_", "NVTE_")
json.dump({
    "created": datetime.datetime.now().astimezone().isoformat(),
    "host": socket.gethostname(),
    "git": {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": dirty == "true", "diff": diff_file if dirty == "true" else None},
    "model": {"arg": model_arg, "source": source, "checkpoint": model},
    "parallelism": {"tp": int(tp), "pp": int(pp), "ep": int(ep), "gpus": int(gpus)},
    "versions": {p: ver(p) for p in ("torch", "transformer_engine", "transformers", "megatron-core")},
    "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(env_keys)},
    "command": cmd,
}, open(out, "w"), indent=2)
EOF
printf 'Manifest: %s\n' "$manifest"

exec "${cmd[@]}"
