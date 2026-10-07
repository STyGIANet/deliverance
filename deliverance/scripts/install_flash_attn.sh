#!/usr/bin/bash
# install_flash_attn.sh - optional. Builds flash-attn for torch 2.14 / CUDA 13 / sm_120.
# Lives in <Megatron-Root>/deliverance/scripts/. Can be run on its own after install.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -d "${REPO_DIR}/.venv" ]]; then
	echo "Error: no .venv in ${REPO_DIR}. Run install.sh first." >&2
	exit 1
fi
cd "${REPO_DIR}"
source .venv/bin/activate

# Tag from the flash-attention GitHub releases. 2.7.4.post1 matches NGC 26.08
# (vLLM/SGLang rows). The alternative is v2.8.3.post1 (NGC 26.09 SGLang).
FA_TAG="v2.7.4.post1"
# Kept inside deliverance/ so the Megatron tree stays clean (~370 MB per wheel).
WHEEL_DIR="${REPO_DIR}/deliverance/wheels"
mkdir -p "${WHEEL_DIR}"

if ! ls "${WHEEL_DIR}"/flash_attn-*.whl >/dev/null 2>&1; then
	BUILD_DIR="$(mktemp -d)"
	git clone --depth 1 --branch "${FA_TAG}" \
		https://github.com/Dao-AILab/flash-attention "${BUILD_DIR}/fa"
	cd "${BUILD_DIR}/fa"
	git submodule update --init --depth 1 csrc/cutlass

	# Workaround: torch 2.14 headers need C++20 (std::strong_ordering), but this
	# flash-attn version builds with C++17. Remove if upstream stops needing it.
	sed -i 's/-std=c++17/-std=c++20/g' setup.py
	grep -q "std=c++20" setup.py   # abort loudly if the patch did not apply

	MAX_JOBS=8 FLASH_ATTENTION_FORCE_BUILD=TRUE \
		pip wheel . --no-build-isolation --no-deps -w "${WHEEL_DIR}"
	cd "${REPO_DIR}"
	rm -rf "${BUILD_DIR}"
fi

pip install "${WHEEL_DIR}"/flash_attn-*.whl
python "${REPO_DIR}/deliverance/examples/flash_attn_check.py"
