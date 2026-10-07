#!/usr/bin/bash
set -euo pipefail

# This script lives in <Megatron-Root>/deliverance/scripts/.
# It can be launched from any directory; all work happens in the Megatron root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ADVENT=${ADVENT:-0}

if [[ ! -f "${REPO_DIR}/pretrain_gpt.py" ]]; then
	echo "Error: ${REPO_DIR} does not look like the Megatron-LM root (no pretrain_gpt.py)." >&2
	echo "Expected this script at <Megatron-Root>/deliverance/scripts/install.sh" >&2
	exit 1
fi

# Ask everything up front so the long steps below run unattended.
# Non-interactive runs can set INSTALL_FLASH_ATTN=yes or =no instead.
INSTALL_FA="${INSTALL_FLASH_ATTN:-}"
if [[ -z "${INSTALL_FA}" ]]; then
	if [[ -t 0 ]]; then
		read -r -p "Install flash-attn? Builds from source, ~30-60 min. [y/N] " reply
		INSTALL_FA="${reply:-n}"
	else
		INSTALL_FA="n"
	fi
fi
case "${INSTALL_FA,,}" in
	y|yes) INSTALL_FA=yes ;;
	*)     INSTALL_FA=no ;;
esac

cd "${REPO_DIR}"
# Everything is relative to the Megatron root from here

# Example scripts live in <root>/deliverance/examples/ (not in Megatron's own examples/).
EXAMPLES_DIR="${REPO_DIR}/deliverance/examples"
for f in train_test.py flash_attn_check.py train_llama3_8b_b600_fp8.sh; do
	if [[ ! -f "${EXAMPLES_DIR}/${f}" ]]; then
		echo "Error: missing ${EXAMPLES_DIR}/${f}" >&2
		exit 1
	fi
done
chmod +x "${EXAMPLES_DIR}/train_llama3_8b_b600_fp8.sh"


if [[ ! -d .venv ]]; then
	uv venv
else
	echo "venv already exists. Using existing virtual environment"
fi

source .venv/bin/activate
uv pip install pip
pip install torch --index-url https://download.pytorch.org/whl/cu130

uv pip install --group build
uv pip install --no-build-isolation -e ".[training,dev]"

# extra packages for inference
uv pip install pyzmq
uv pip install msgpack


# Set up ninja
if [[ ! -d ninja ]]; then
	git clone https://github.com/ninja-build/ninja.git
	cd ninja
	git checkout release

	./configure.py --bootstrap
	./ninja all
	cd "${REPO_DIR}" # Go back
else
	echo "ninja directory already exists. skipping clone"
fi


# Set up apex
if [[ ! -d apex ]]; then
	git clone https://github.com/NVIDIA/apex
	cd apex
	NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
	cd "${REPO_DIR}" # Go back
else
	echo "apex already cloned"
fi



# Install the Transformer Engine (TE) related to this release
# To be more robust, must search the nvidia matrix:
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
# For 08.26, that's TE v2.18. Can be reflected in the requirements.txt but foregoing that

uv pip install --no-build-isolation "transformer_engine[pytorch]==2.18"


####### NOTE: This is advent specific and may not generalise #######
# Enforcing libcudnn as multiple versions will crash the program.
# Symlink to libcudnn.so.9
_CUDNN_DIR="${REPO_DIR}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"

# Relative target: resolved against the link's own directory.
# -f makes this re-runnable and replaces any dangling link from earlier runs.
ln -sf libcudnn.so.9 "${_CUDNN_DIR}/libcudnn.so"

export LD_LIBRARY_PATH="${_CUDNN_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Persist for future shells. Single quotes: expanded at activation, not now.
if ! grep -q "nvidia/cudnn/lib" .venv/bin/activate; then
	echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> .venv/bin/activate
fi

if [[ "${ADVENT}" == "1" ]]; then
	echo "Adding advent specific config to venv"
	# Export now: this script already sourced activate, so it will not re-read
	# the lines appended below, and the inference test at the end needs them.
	export DEPOT="/depot/deliverance/"
	export DELIVERANCE_MODEL_ROOT="/depot/deliverance/models"
	export DELIVERANCE_DATASET_ROOT="/depot/deliverance/datasets/moe-inference-datasets"

	# Persist for future shells. Guarded so re-runs do not append duplicates.
	if ! grep -q "DELIVERANCE_MODEL_ROOT" .venv/bin/activate; then
		cat >> .venv/bin/activate <<'EOT'
export ADVENT=1
export DEPOT="/depot/deliverance/"
export DELIVERANCE_MODEL_ROOT="/depot/deliverance/models"
export DELIVERANCE_DATASET_ROOT="/depot/deliverance/datasets/moe-inference-datasets"
EOT
	fi
fi

####### End advent tweaks ########


# Fast check (seconds): TE fused attention works. Fails here if the env is wrong.
python "${EXAMPLES_DIR}/train_test.py"

# Optional: flash-attn (answered at the top)
if [[ "${INSTALL_FA}" == "yes" ]]; then
	bash "${SCRIPT_DIR}/install_flash_attn.sh"
fi

# Now run the sample
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py

# Inference test
if [[ "${ADVENT}" == "1" ]]; then
	INFERENCE_TEST="${REPO_DIR}/deliverance/scripts/advent_inference_test.sh"
	if [[ ! -x "${INFERENCE_TEST}" ]]; then
		echo "Error: ${INFERENCE_TEST} not found or not executable" >&2
		exit 1
	fi
	cd "${REPO_DIR}"
	"${INFERENCE_TEST}"
fi
