#!/usr/bin/env bash
set -euo pipefail

# Download a Hugging Face MoE model and convert it to a legacy Megatron checkpoint.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ckpt_root="${DELIVERANCE_CKPT_ROOT:-/depot/shared/megatron-checkpoints}"
hf_include=('*.json' '*.safetensors' 'tokenizer.model')

usage() {
  cat <<'EOF'
Usage: prepare_checkpoint.sh [options]

Options:
  --hf-repo ID            Hugging Face repo ID (default: mistralai/Mixtral-8x7B-v0.1)
  --hf-dir PATH           Use an existing local HF model directory instead of
                          the hub cache; nothing is downloaded
  --name NAME             Checkpoint name prefix (default: lowercased repo basename)
  --tp N                  Target tensor parallel size (default: 1)
  --pp N                  Target pipeline parallel size (default: 1)
  --ep N                  Target expert parallel size (default: 8)
  --skip-download         Use the cached snapshot as-is; fail if it is incomplete
  --dry-run               Print the commands without running them
  -h, --help              Show this help

Steps:
  1. Download the repo into the Hugging Face hub cache (HF_HUB_CACHE, shared
     at /depot/shared/.cache/huggingface/hub on this host) with
     `hf download`, unless the cached snapshot already has every shard in
     model.safetensors.index.json.
  2. Convert with tools/checkpoint/convert.py into
     <ckpt_root>/<name>-tp<T>-pp<P>-ep<E>, via a .incoming directory that is
     renamed only after the shard layout is verified.

Only Mixtral is supported: it is the only HF MoE loader in tools/checkpoint.
The Mixtral loader needs transformers 4.x (>= 4.36; 4.57.6 is known good).
This script checks the version but never changes the Python environment.
Gated repos need HF_TOKEN in the environment (or a prior `hf auth login`).
Override the checkpoint root with DELIVERANCE_CKPT_ROOT
(default /depot/shared/megatron-checkpoints).
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

hf_repo="mistralai/Mixtral-8x7B-v0.1"
hf_dir=""
name=""
tp_size=1
pp_size=1
ep_size=8
skip_download=false
dry_run=false

while (($#)); do
  case "$1" in
    --hf-repo)
      need_value "$@"
      hf_repo="$2"
      shift 2
      ;;
    --hf-dir)
      need_value "$@"
      hf_dir="$2"
      shift 2
      ;;
    --name)
      need_value "$@"
      name="$2"
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
    --skip-download)
      skip_download=true
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

[[ "$hf_repo" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || die "--hf-repo must look like org/name"
positive_integer --tp "$tp_size"
positive_integer --pp "$pp_size"
positive_integer --ep "$ep_size"
if [[ -z "$name" ]]; then
  name="${hf_repo##*/}"
  name="${name,,}"
fi
[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || die "--name may only contain letters, digits, '.', '_' and '-'"

command -v python >/dev/null || die "python not found; source .venv/bin/activate first"

ckpt_dir="$ckpt_root/$name-tp$tp_size-pp$pp_size-ep$ep_size"
incoming_dir="$ckpt_dir.incoming"

# The Mixtral loader indexes per-expert w1/w2/w3 modules, which transformers 5
# fused into gate_up_proj/down_proj tensors.
transformers_version="$(python -c 'import importlib.metadata as m; print(m.version("transformers"))' 2>/dev/null)" ||
  die "transformers is not installed in $(command -v python)"
[[ "$transformers_version" =~ ^4\.([0-9]+)\. ]] && (( BASH_REMATCH[1] >= 36 )) ||
  die "transformers $transformers_version found; the Mixtral loader needs 4.x >= 4.36 (4.57.6 is known good). Install it yourself, e.g. uv pip install --python .venv/bin/python 'transformers==4.57.6'"

[[ ! -e "$ckpt_dir" ]] || die "checkpoint already exists: $ckpt_dir"
[[ ! -e "$incoming_dir" ]] || die "partial conversion found: $incoming_dir (inspect and remove it first)"
[[ -d "$ckpt_root" ]] || die "checkpoint root not found: $ckpt_root"
[[ -w "$ckpt_root" ]] || die "checkpoint root is not writable: $ckpt_root (set DELIVERANCE_CKPT_ROOT or fix permissions)"

# Print the shards listed in the safetensors index that are missing locally.
missing_shards() {
  python - "$hf_dir" <<'EOF'
import json, os, sys
d = sys.argv[1]
with open(os.path.join(d, "model.safetensors.index.json")) as f:
    shards = sorted(set(json.load(f)["weight_map"].values()))
for s in shards:
    if not os.path.isfile(os.path.join(d, s)):
        print(s)
EOF
}

hf_complete() {
  local f
  for f in config.json tokenizer.model model.safetensors.index.json; do
    [[ -s "$hf_dir/$f" ]] || return 1
  done
  [[ -z "$(missing_shards)" ]]
}

# Print the cached snapshot directory for the repo, or nothing if it is not
# cached. Does not touch the network.
cached_snapshot() {
  python - "$hf_repo" "${hf_include[@]}" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
try:
    print(snapshot_download(sys.argv[1], allow_patterns=sys.argv[2:], local_files_only=True))
except LocalEntryNotFoundError:
    pass
EOF
}

download_cmd=(hf download "$hf_repo" --include "${hf_include[@]}")
need_download=false

if [[ -n "$hf_dir" ]]; then
  [[ -d "$hf_dir" ]] || die "--hf-dir not found: $hf_dir"
  hf_dir="$(realpath "$hf_dir")"
  hf_complete || die "HF model is incomplete at $hf_dir"
  printf 'HF model: %s (local directory)\n' "$hf_dir"
else
  hub_cache="$(python -c 'from huggingface_hub import constants; print(constants.HF_HUB_CACHE)')"
  printf 'HF hub cache: %s\n' "$hub_cache"
  hf_dir="$(cached_snapshot)"
  if [[ -n "$hf_dir" ]] && hf_complete; then
    printf 'HF model: %s (cached, skipping download)\n' "$hf_dir"
  elif "$skip_download"; then
    die "$hf_repo is not fully cached under $hub_cache and --skip-download was given"
  else
    printf 'HF model: %s (will download into the hub cache)\n' "$hf_repo"
    need_download=true
  fi
fi

if "$need_download" && ! "$dry_run"; then
  command -v hf >/dev/null || die "hf CLI not found; source .venv/bin/activate first"
  # The hub cache is shared with the depotUser group; keep new files group-writable.
  (umask 002 && "${download_cmd[@]}")
  hf_dir="$(cached_snapshot)"
  [[ -n "$hf_dir" ]] || die "download finished but $hf_repo is not in the hub cache"
  hf_complete || die "download finished but files are still missing in $hf_dir: $(missing_shards | tr '\n' ' ')"
fi

# Read the model config (skipped on a dry run that still needs a download).
model_type=""
num_experts=""
if [[ -n "$hf_dir" && -s "$hf_dir/config.json" ]]; then
  read -r model_type num_experts < <(python - "$hf_dir/config.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
print(c.get("model_type", ""), c.get("num_local_experts", 0))
EOF
  )
  [[ "$model_type" == mixtral ]] ||
    die "model_type '$model_type' is not supported; only mixtral has an HF loader in tools/checkpoint"
  (( num_experts % ep_size == 0 )) || die "EP=$ep_size does not divide $num_experts experts"
fi

# A dry run that still needs a download has no snapshot path yet.
load_dir="${hf_dir:-<hub snapshot of $hf_repo>}"
convert_cmd=(
  python tools/checkpoint/convert.py
  --model-type GPT
  --loader mixtral_hf
  --saver core
  --load-dir "$load_dir"
  --save-dir "$incoming_dir"
  --tokenizer-model "$load_dir/tokenizer.model"
  --target-tensor-parallel-size "$tp_size"
  --target-pipeline-parallel-size "$pp_size"
  --target-expert-parallel-size "$ep_size"
)

printf 'Transformers: %s\nTarget: TP=%s PP=%s EP=%s\nCheckpoint: %s\n' \
  "$transformers_version" "$tp_size" "$pp_size" "$ep_size" "$ckpt_dir"
if "$dry_run"; then
  if "$need_download"; then
    printf 'Download: '
    printf '%q ' "${download_cmd[@]}"
    printf '\n'
  fi
  printf 'Convert: '
  printf '%q ' "${convert_cmd[@]}"
  printf '\n'
  exit 0
fi

# convert.py does not check the saver process's exit status, so verify the
# shard layout ourselves before publishing the checkpoint.
verify_layout() {
  local step iter_dir tp pp ep shard expected=0 found

  [[ -s "$incoming_dir/latest_checkpointed_iteration.txt" ]] || die "converter wrote no tracker file in $incoming_dir"
  step="$(< "$incoming_dir/latest_checkpointed_iteration.txt")"
  [[ "$step" =~ ^[0-9]+$ ]] || die "unexpected checkpoint iteration '$step' in $incoming_dir"
  printf -v iter_dir '%s/iter_%07d' "$incoming_dir" "$((10#$step))"

  for ((tp = 0; tp < tp_size; tp++)); do
    for ((pp = 0; pp < pp_size; pp++)); do
      for ((ep = 0; ep < ep_size; ep++)); do
        printf -v shard 'mp_rank_%02d' "$tp"
        (( pp_size == 1 )) || printf -v shard '%s_%03d' "$shard" "$pp"
        (( ep_size == 1 )) || printf -v shard '%s_%03d' "$shard" "$ep"
        [[ -s "$iter_dir/$shard/model_optim_rng.pt" ]] || die "missing shard after conversion: $iter_dir/$shard"
        expected=$((expected + 1))
      done
    done
  done
  found="$(find "$iter_dir" -mindepth 1 -maxdepth 1 -type d -name 'mp_rank_*' | wc -l)"
  (( found == expected )) || die "expected $expected shard directories in $iter_dir, found $found"
}

write_source_json() {
  python - "$1" "$hf_repo" "$hf_dir" "$tp_size" "$pp_size" "$ep_size" <<'EOF'
import json, os, sys
out, repo, hf_dir, tp, pp, ep = sys.argv[1:]
# Hub-cache snapshots live in snapshots/<commit>; a --hf-dir copy has no revision.
parent = os.path.basename(os.path.dirname(hf_dir))
json.dump({
    "hf_repo": repo,
    "hf_revision": os.path.basename(hf_dir) if parent == "snapshots" else None,
    "hf_dir": hf_dir,
    "tokenizer_type": "Llama2Tokenizer",
    "tokenizer_model": os.path.join(hf_dir, "tokenizer.model"),
    "tensor_model_parallel_size": int(tp),
    "pipeline_model_parallel_size": int(pp),
    "expert_model_parallel_size": int(ep),
    "format": "torch",
}, open(out, "w"), indent=2)
EOF
}

cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
# The checkpoint root is shared with the depotUser group.
umask 002
"${convert_cmd[@]}"
verify_layout
# The converter does not record the tokenizer path in the checkpoint args, so
# leave a sidecar that run_moe_inference.sh reads to pick the tokenizer.
write_source_json "$incoming_dir/deliverance_source.json"
mv "$incoming_dir" "$ckpt_dir"
printf 'Done: %s\n' "$ckpt_dir"
