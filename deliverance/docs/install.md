# Installation

Our work is based on the 26.08 tag of Megatron with cuda13. 

If using stygianet advent machine, prepend `ADVENT=1` for additional ease of access such as the defined variables:
```
DEPOT = /depot/deliverance/
DELIVERANCE_MODEL_ROOT = /depot/shared/megatron-checkpoints   # read by run_moe_inference.sh
DELIVERANCE_CKPT_ROOT = /depot/shared/megatron-checkpoints    # read by prepare_checkpoint.sh
DELIVERANCE_DATASET_ROOT = /depot/shared/datasets/moe-inference-datasets
HF_HOME = /depot/shared/.cache/huggingface                    # HF_HUB_CACHE, HF_DATASETS_CACHE below it
```

HF models live in the shared hub cache, not as plain folders. The `HF_*` variables are also set
in `/etc/profile.d/huggingface.sh`, but only login shells read that file, so the venv sets them
too (keeping any value already set).

Existing venvs keep their old exports; re-running the installer will not update them. Fix them by
hand to match the list above.

## Instructions
Note, for inference, we build flash-attn from source with c++20. This takes a while. If you select to build from source, go make yourself some tea after confirming. Without flash-attn, installation takes about 10 minutes. Installation jumps to about 1.1 hours.

The file can be sourced from anywhere. This instruction assumes you are sourcing from Megatron's root directory. 

For unattended install, prepend `INSTALL_FLASH_ATTN=yes/no` to automatically approve/decline flash-attn build from source respectively.
```
./deliverance/scripts/install.sh
```
