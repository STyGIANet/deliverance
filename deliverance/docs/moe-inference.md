# Eight-GPU MoE inference

Run from `<MEGATRON_ROOT>` with the venv activated (`source .venv/bin/activate`). The venv's
`activate` sets the shared roots (see [install.md](install.md)).

## Quick start

```bash
# Default: TP=1, PP=1, EP=8 on 8 GPUs. Downloads and converts the model on first use.
bash deliverance/scripts/run_moe_inference.sh \
  --model mistralai/Mixtral-8x7B-v0.1 --dataset gsm8k --samples 4 --max-new-tokens 8

# Same, with MoE phase timing (see moe-timing.md)
bash deliverance/scripts/run_moe_inference.sh \
  --model mistralai/Mixtral-8x7B-v0.1 --dataset gsm8k --samples 4 --max-new-tokens 8 \
  --timing-dir local/moe-timing/$(date +%Y%m%d-%H%M%S)
```

Add `--dry-run` to print what would run, including any download or conversion, without doing it.
`--help` lists every option.

## Model resolution

`--model` is resolved in this order:

1. **A path** to a Megatron checkpoint directory.
2. **A checkpoint name** under `DELIVERANCE_CKPT_ROOT` (default
   `/depot/shared/megatron-checkpoints`), e.g. `mixtral-8x7b-tp4-pp1-ep1`.
3. **A Hugging Face repo ID** (`org/name`). The launcher uses
   `<ckpt_root>/<lowercased name>-tp<T>-pp<P>-ep<E>`. If that does not exist, it runs
   [`prepare_checkpoint.sh`](#preparing-checkpoints) first, then launches inference.

Routes 1 and 2 never convert. Hugging Face safetensors directories are rejected; pass the
repo ID instead.

## Parallelism

`--tp`, `--pp` and `--ep` default to 1, 1 and 8; `--gpus` defaults to 8.

Legacy (`torch` format) checkpoints cannot be resharded while loading, so their shard directories
(`mp_rank_TT[_PPP][_EEE]`) must match the requested TP/PP/EP exactly; the launcher checks this before
starting. To run another layout from HF, pass the repo ID with the new sizes and it converts a
matching checkpoint. The older EP=1 checkpoints (`mixtral-8x7b-tp{1,4,8}-pp1-ep1`) need `--ep 1` and
the matching `--tp`.

## Tokenizer

Checkpoints made by `prepare_checkpoint.sh` contain `deliverance_source.json` recording the HF repo,
revision and tokenizer. The launcher uses it automatically; the converter itself stores no
tokenizer path. For the older EP=1 checkpoints, whose stored tokenizer path points into a
container, pass it explicitly:

```bash
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /depot/shared/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0/tokenizer.model
```

## Pinned settings and pass-through

The launcher passes `--transformer-impl transformer_engine` and `--moe-token-dispatcher-type`
(`--dispatcher`, default `alltoall`) explicitly so runs compare the same layer implementation;
`--use-checkpoint-args` does not override either. Anything after `--` goes to Megatron unchanged.

## Outputs

- Results JSON: `local/moe-inference-results/<timestamp>.json` by default, or `--output PATH`. An
  existing file is never overwritten.
- `<output>.manifest.json`: command, git commit/branch/dirty flag, model resolution, parallelism,
  package versions and `CUDA_*`/`NCCL_*`/`MOE_*`/`DELIVERANCE_*`/`TORCH_*`/`NVTE_*` environment.
- `<output>.diff`: `git diff HEAD` when the tree has uncommitted tracked changes.

## Preparing checkpoints

`deliverance/scripts/prepare_checkpoint.sh` is what the launcher calls; it can also run alone:

```bash
bash deliverance/scripts/prepare_checkpoint.sh --hf-repo mistralai/Mixtral-8x7B-v0.1 --tp 1 --pp 1 --ep 8 --dry-run
```

1. Downloads the repo into the shared HF hub cache (`HF_HUB_CACHE`) unless the cached snapshot
   already has every shard. `--hf-dir PATH` uses an existing local copy instead.
2. Converts with `tools/checkpoint/convert.py` (`mixtral_hf` loader, `core` saver) into
   `<ckpt_root>/<name>-tp<T>-pp<P>-ep<E>.incoming`, verifies every shard (the converter ignores the
   saver's exit status), writes `deliverance_source.json`, then renames it into place.

Constraints:

- Only Mixtral: it is the only HF MoE loader in `tools/checkpoint`.
- The loader needs **transformers 4.x ≥ 4.36** (4.57.6 is known good; transformers 5 fused the
  expert weights and renamed the MoE block). The script checks but never changes the venv. Note
  `uv sync` restores the locked 5.x.
- With EP > 1 every shard holds a full copy of the attention and embedding weights (the legacy
  format writes each rank's complete state dict), so EP=8 Mixtral is 108 GB on disk versus 87 GB.

Verified on 2026-10-27: Mixtral EP=8 conversion, then a 4-prompt gsm8k run produced token IDs
identical to the TP=4/EP=1 checkpoint, with 608 timing rows per rank.
