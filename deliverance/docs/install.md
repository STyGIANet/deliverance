# Installation

Our work is based on the 26.08 tag of Megatron with cuda13. 

If using stygianet advent machine, prepend `ADVENT=1` for additional ease of access such as the defined variables:
```
DEPOT = /depot/deliverance/
DELIVERANCE_MODEL_ROOT = /depot/deliverance/models
DELIVERANCE_DATASET_ROOT = /depot/deliverance/datasets/moe-inference-datasets
```

## Instructions
Note, for inference, we build flash-attn from source with c++20. This takes a while. If you select to build from source, go make yourself some tea after confirming. Without flash-attn, installation takes about 10 minutes. Installation jumps to about 1.1 hours.

The file can be sourced from anywhere. This instruction assumes you are sourcing from Megatron's root directory. 

For unattended install, prepend `INSTALL_FLASH_ATTN=yes/no` to automatically approve/decline flash-attn build from source respectively.
```
./deliverance/scripts/install.sh
```
