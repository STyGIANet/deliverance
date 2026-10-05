# Deliverance NGC container setup (this machine)

This setup uses `nvcr.io/nvidia/pytorch:26.08-py3` and the checkout at `/home/nirmal/2026/deliverance`. Run the Docker command **on the host**. Run every install command **inside the container**; no `docker exec` is needed.

## Start the container

From the host:

```bash
cd /home/nirmal/2026/deliverance
docker run --rm -it --name deliverance-ngc --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  --workdir /workspace/deliverance \
  -v "$PWD:/workspace/deliverance" \
  -v /srv/deliverance/datasets:/datasets:ro \
  -v /srv/deliverance/models:/models \
  -e PIP_CONSTRAINT= \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e XDG_CACHE_HOME=/tmp/deliverance-cache \
  -e TRITON_CACHE_DIR=/tmp/deliverance-cache/triton \
  nvcr.io/nvidia/pytorch:26.08-py3 bash
```

Choose another `--name` if `deliverance-ngc` already exists. The NGC image already provides CUDA, PyTorch, and Transformer Engine. `/datasets` and `/models` are shared host mounts; writing a converted checkpoint to `/models` also requires host filesystem permission. Save large checkpoints there, not under the Git checkout.

## Set up Python inside the container

For a fresh checkout, run these commands in order. If `.venv` already exists from a previous container, skip steps 1, 2, and 4; just activate it and check it.

1. Create a venv that can use the NGC-provided GPU packages:

   ```bash
   python -m venv --system-site-packages .venv
   ```

2. Install uv into that persistent venv. The container runs without a usable home directory, so the default uv install path (`//.local/bin`) fails:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/workspace/deliverance/.venv/bin sh
   ```

3. Activate the venv:

   ```bash
   source .venv/bin/activate
   ```

4. Install this checkout and inference dependencies without replacing NGC's PyTorch:

   ```bash
   uv pip install --python .venv/bin/python --no-build-isolation -e '.[training]' simpy msgpack accelerate sentencepiece
   ```

5. Verify the environment and all eight GPUs:

   ```bash
   python -c 'import torch, megatron.core, transformer_engine, sentencepiece, simpy; assert torch.cuda.device_count() == 8; print("Megatron import OK; 8 GPUs visible")'
   ```

For a fresh Mixtral conversion, use the [inference guide](moe-inference.md) after conversion and pass the tokenizer explicitly (`--tokenizer-type Llama2Tokenizer --tokenizer-model /models/mistralai/.Mixtral-8x7B-v0.1.incoming/tokenizer.model`). The legacy Mixtral converter's version check rejects Transformers 5.x; use `uv pip install --python .venv/bin/python 'transformers==4.57.6'` before running that converter.

Model Conversion steps (For tp8)
```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mixtral_hf \
  --saver core \
  --load-dir /models/mistralai/.Mixtral-8x7B-v0.1.incoming \
  --save-dir /models/mixtral-8x7b-tp8-pp1-ep1 \
  --tokenizer-model /models/mistralai/.Mixtral-8x7B-v0.1.incoming/tokenizer.model \
  --target-tensor-parallel-size 8 \
  --target-pipeline-parallel-size 1 \
  --target-expert-parallel-size 1
  ```

## Reusing the environment

The checkout is bind-mounted, so `.venv` survives stopping and recreating the container. On each new start, run `cd /workspace/deliverance` and `source .venv/bin/activate`. Create a new venv only if the checkout/venv was removed, the base image or Python version changes incompatibly, or the environment is broken and cannot be repaired. Do **not** recreate it merely because the container was restarted.

The caches above live in the container's `/tmp`: they remain available while that container stays running, but disappear when it is removed. A new container can rebuild JIT kernels; it does not need to reinstall the checkout's venv. The shared datasets and models survive because they are host mounts.

If a module is installed in `.venv` but a distributed run says `ModuleNotFoundError`, check the launcher: the NGC image's `/usr/local/bin/torchrun` uses `/usr/bin/python3`, not the activated venv. Launch distributed Python with `.venv/bin/python -m torch.distributed.run ...` so worker processes see venv packages. The current `deliverance/scripts/run_moe_inference.sh` still calls `torchrun` and needs that one-line change before using venv-only packages.
