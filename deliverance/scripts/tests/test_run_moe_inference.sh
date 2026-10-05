#!/usr/bin/env bash
set -euo pipefail

# Exercise topology selection without loading a model or using a GPU.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launcher="$script_dir/../run_moe_inference.sh"
fixtures_dir="$(mktemp -d)"
trap 'rm -r -- "$fixtures_dir"' EXIT

dataset="$fixtures_dir/prompts.jsonl"
printf '{"text":"hello"}\n' > "$dataset"

make_legacy_checkpoint() {
  local name="$1"
  local tp_size="$2"
  local checkpoint_dir="$fixtures_dir/$name"
  local shard_name index

  mkdir -p "$checkpoint_dir/iter_0000001"
  printf '1\n' > "$checkpoint_dir/latest_checkpointed_iteration.txt"

  for ((index = 0; index < tp_size; index++)); do
    printf -v shard_name 'mp_rank_%02d' "$index"
    mkdir -p "$checkpoint_dir/iter_0000001/$shard_name"
    printf 'fixture\n' > "$checkpoint_dir/iter_0000001/$shard_name/model_optim_rng.pt"
  done
}

expect_topology() {
  local expected="$1"
  shift

  local output tp_size
  output="$(bash "$launcher" --dataset "$dataset" --dry-run "$@")"
  if [[ "$output" != *"Parallelism: $expected"* ]]; then
    printf 'Expected %s, got:\n%s\n' "$expected" "$output" >&2
    exit 1
  fi

  [[ "$expected" =~ TP=([0-9]+) ]] || exit 1
  tp_size="${BASH_REMATCH[1]}"
  if [[ "$output" != *"--tensor-model-parallel-size $tp_size"* ]]; then
    printf 'The command does not use TP=%s:\n%s\n' "$tp_size" "$output" >&2
    exit 1
  fi
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

make_legacy_checkpoint tp1 1
make_legacy_checkpoint tp2 2
make_legacy_checkpoint tp4 4
make_legacy_checkpoint tp8 8

expect_topology 'TP=8 DP=1' --model "$fixtures_dir/tp8" --parallelism tp
expect_topology 'TP=1 DP=8' --model "$fixtures_dir/tp1" --parallelism dp
expect_topology 'TP=2 DP=4' --model "$fixtures_dir/tp2" --parallelism dp
expect_topology 'TP=4 DP=2' --model "$fixtures_dir/tp4" --parallelism dp
expect_topology 'TP=4 DP=2' --model "$fixtures_dir/tp4" --parallelism dp --tp-size 4

expect_error 'needs at least two replicas' --model "$fixtures_dir/tp8" --parallelism dp
expect_error 'checkpoint is TP=4 but launch requested TP=2' \
  --model "$fixtures_dir/tp4" --parallelism dp --tp-size 2
expect_error 'checkpoint is TP=4 but launch requested TP=8' \
  --model "$fixtures_dir/tp4" --parallelism tp
expect_error '--tp-size must be 1, 2, or 4' \
  --model "$fixtures_dir/tp4" --parallelism dp --tp-size 3
expect_error '--model needs a value' --model --parallelism dp
expect_error '--max-new-tokens cannot exceed --max-seq-length' \
  --model "$fixtures_dir/tp4" --parallelism dp \
  --max-new-tokens 33 --max-seq-length 32
expect_error 'output already exists' \
  --model "$fixtures_dir/tp4" --parallelism dp --output "$dataset"

ln -s "$fixtures_dir/nonexistent.json" "$fixtures_dir/broken-output-link.json"
expect_error 'output already exists' \
  --model "$fixtures_dir/tp4" --parallelism dp \
  --output "$fixtures_dir/broken-output-link.json"

mkdir -p "$fixtures_dir/distributed/iter_0000001"
printf '1\n' > "$fixtures_dir/distributed/latest_checkpointed_iteration.txt"
printf '{}\n' > "$fixtures_dir/distributed/iter_0000001/metadata.json"
expect_error 'cannot infer TP from a distributed checkpoint' \
  --model "$fixtures_dir/distributed" --parallelism dp
expect_topology 'TP=4 DP=2' \
  --model "$fixtures_dir/distributed" --parallelism dp --tp-size 4

make_legacy_checkpoint incomplete 4
rm -- "$fixtures_dir/incomplete/iter_0000001/mp_rank_02/model_optim_rng.pt"
expect_error 'missing or empty checkpoint shard' \
  --model "$fixtures_dir/incomplete" --parallelism dp

make_legacy_checkpoint nonconsecutive 4
mv -- "$fixtures_dir/nonconsecutive/iter_0000001/mp_rank_02" \
  "$fixtures_dir/nonconsecutive/iter_0000001/mp_rank_05"
expect_error 'missing or empty checkpoint shard' \
  --model "$fixtures_dir/nonconsecutive" --parallelism dp

make_legacy_checkpoint pipeline_parallel 4
mv -- "$fixtures_dir/pipeline_parallel/iter_0000001/mp_rank_02" \
  "$fixtures_dir/pipeline_parallel/iter_0000001/mp_rank_02_000"
expect_error 'unsupported shard layout' \
  --model "$fixtures_dir/pipeline_parallel" --parallelism dp

printf 'Launcher topology tests passed.\n'
