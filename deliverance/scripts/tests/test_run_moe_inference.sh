#!/usr/bin/env bash
set -euo pipefail

# Exercise topology and model selection without loading a model or using a GPU.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launcher="$script_dir/../run_moe_inference.sh"
fixtures_dir="$(mktemp -d)"
trap 'rm -r -- "$fixtures_dir"' EXIT

# Keep the HF route and name lookups inside the fixtures.
export DELIVERANCE_CKPT_ROOT="$fixtures_dir/ckpts"
unset DELIVERANCE_MODEL_ROOT
mkdir -p "$DELIVERANCE_CKPT_ROOT"

dataset="$fixtures_dir/prompts.jsonl"
printf '{"text":"hello"}\n' > "$dataset"

# make_legacy_checkpoint DIR TP PP EP: shard names follow Megatron's
# mp_rank_TT[_PPP][_EEE] convention.
make_legacy_checkpoint() {
  local checkpoint_dir="$1" tp_size="$2" pp_size="$3" ep_size="$4"
  local shard tp pp ep

  mkdir -p "$checkpoint_dir/iter_0000001"
  printf '1\n' > "$checkpoint_dir/latest_checkpointed_iteration.txt"

  for ((tp = 0; tp < tp_size; tp++)); do
    for ((pp = 0; pp < pp_size; pp++)); do
      for ((ep = 0; ep < ep_size; ep++)); do
        printf -v shard 'mp_rank_%02d' "$tp"
        (( pp_size == 1 )) || printf -v shard '%s_%03d' "$shard" "$pp"
        (( ep_size == 1 )) || printf -v shard '%s_%03d' "$shard" "$ep"
        mkdir -p "$checkpoint_dir/iter_0000001/$shard"
        printf 'fixture\n' > "$checkpoint_dir/iter_0000001/$shard/model_optim_rng.pt"
      done
    done
  done
}

expect_output() {
  local expected="$1"
  shift

  local output
  output="$(bash "$launcher" --dataset "$dataset" --dry-run "$@" 2>&1)"
  if [[ "$output" != *"$expected"* ]]; then
    printf 'Expected output containing %q, got:\n%s\n' "$expected" "$output" >&2
    exit 1
  fi
}

# Checks the sizes passed to Megatron.
expect_topology() {
  local tp="$1" pp="$2" ep="$3"
  shift 3

  expect_output "Parallelism: TP=$tp PP=$pp EP=$ep on 8 GPUs" "$@"
  expect_output "--tensor-model-parallel-size $tp --pipeline-model-parallel-size $pp --expert-model-parallel-size $ep" "$@"
}

expect_error() {
  local expected="$1"
  shift

  local output
  if output="$(bash "$launcher" --dataset "$dataset" --dry-run "$@" 2>&1)"; then
    printf 'Expected an error containing %q, got success:\n%s\n' "$expected" "$output" >&2
    exit 1
  fi
  if [[ "$output" != *"$expected"* ]]; then
    printf 'Expected error containing %q, got:\n%s\n' "$expected" "$output" >&2
    exit 1
  fi
}

make_legacy_checkpoint "$fixtures_dir/tp1-ep1" 1 1 1
make_legacy_checkpoint "$fixtures_dir/tp4-ep1" 4 1 1
make_legacy_checkpoint "$fixtures_dir/tp8-ep1" 8 1 1
make_legacy_checkpoint "$fixtures_dir/tp1-ep8" 1 1 8
make_legacy_checkpoint "$fixtures_dir/tp2-ep4" 2 1 4
make_legacy_checkpoint "$fixtures_dir/tp1-pp2-ep4" 1 2 4

# Defaults are TP=1, PP=1, EP=8.
expect_topology 1 1 8 --model "$fixtures_dir/tp1-ep8"
expect_topology 1 1 1 --model "$fixtures_dir/tp1-ep1" --ep 1
expect_topology 4 1 1 --model "$fixtures_dir/tp4-ep1" --tp 4 --ep 1
expect_topology 8 1 1 --model "$fixtures_dir/tp8-ep1" --tp 8 --ep 1
expect_topology 2 1 4 --model "$fixtures_dir/tp2-ep4" --tp 2 --ep 4
expect_topology 1 2 4 --model "$fixtures_dir/tp1-pp2-ep4" --pp 2 --ep 4

# Legacy layouts must match the request exactly.
expect_error 'layout does not match TP=1 PP=1 EP=8' --model "$fixtures_dir/tp4-ep1"
expect_error 'checkpoint has 4 shard directories but TP=2 PP=1 EP=1 expects 2' \
  --model "$fixtures_dir/tp4-ep1" --tp 2 --ep 1
expect_error 'checkpoint has 8 shard directories but TP=1 PP=1 EP=4 expects 4' \
  --model "$fixtures_dir/tp1-ep8" --ep 4
expect_error 'layout does not match TP=1 PP=1 EP=4' --model "$fixtures_dir/tp1-pp2-ep4" --ep 4

make_legacy_checkpoint "$fixtures_dir/incomplete" 1 1 8
rm -- "$fixtures_dir/incomplete/iter_0000001/mp_rank_00_005/model_optim_rng.pt"
expect_error 'missing mp_rank_00_005' --model "$fixtures_dir/incomplete"

# Argument validation.
expect_error '--model needs a value' --model --ep 8
expect_error '--max-new-tokens cannot exceed --max-seq-length' \
  --model "$fixtures_dir/tp1-ep8" --max-new-tokens 33 --max-seq-length 32
expect_error 'output already exists' --model "$fixtures_dir/tp1-ep8" --output "$dataset"
ln -s "$fixtures_dir/nonexistent.json" "$fixtures_dir/broken-output-link.json"
expect_error 'output already exists' \
  --model "$fixtures_dir/tp1-ep8" --output "$fixtures_dir/broken-output-link.json"

# Distributed checkpoints are resharded at load time, so any layout is accepted.
mkdir -p "$fixtures_dir/distributed/iter_0000001"
printf '1\n' > "$fixtures_dir/distributed/latest_checkpointed_iteration.txt"
printf '{}\n' > "$fixtures_dir/distributed/iter_0000001/metadata.json"
expect_topology 1 1 8 --model "$fixtures_dir/distributed"
expect_topology 4 1 2 --model "$fixtures_dir/distributed" --tp 4 --ep 2

# Model resolution: name under the checkpoint root, then HF repo ID.
make_legacy_checkpoint "$DELIVERANCE_CKPT_ROOT/some-ckpt" 1 1 8
expect_output "(name)" --model some-ckpt
make_legacy_checkpoint "$DELIVERANCE_CKPT_ROOT/mixtral-8x7b-v0.1-tp1-pp1-ep8" 1 1 8
expect_output "$DELIVERANCE_CKPT_ROOT/mixtral-8x7b-v0.1-tp1-pp1-ep8 (hf)" --model mistralai/Mixtral-8x7B-v0.1
expect_error 'model not found: nope' --model nope

# The sidecar written by prepare_checkpoint.sh supplies the tokenizer.
printf 'tok\n' > "$fixtures_dir/tokenizer.model"
printf '{"tokenizer_type": "Llama2Tokenizer", "tokenizer_model": "%s"}\n' "$fixtures_dir/tokenizer.model" \
  > "$DELIVERANCE_CKPT_ROOT/some-ckpt/deliverance_source.json"
expect_output "--no-use-tokenizer-model-from-checkpoint-args --tokenizer-type Llama2Tokenizer --tokenizer-model $fixtures_dir/tokenizer.model" \
  --model some-ckpt

# Pinned settings, timing and pass-through.
expect_output "--moe-token-dispatcher-type allgather --transformer-impl transformer_engine" \
  --model "$fixtures_dir/tp1-ep8" --dispatcher allgather
expect_output "Timing: $fixtures_dir/timing" --model "$fixtures_dir/tp1-ep8" --timing-dir "$fixtures_dir/timing"
mkdir -p "$fixtures_dir/used-timing" && touch "$fixtures_dir/used-timing/rank00000.jsonl"
expect_error 'already has rank files' --model "$fixtures_dir/tp1-ep8" --timing-dir "$fixtures_dir/used-timing"
expect_output "--incoming-requests-per-sec -1 --num-layers 2" --model "$fixtures_dir/tp1-ep8" -- --num-layers 2

printf 'Launcher topology tests passed.\n'
