# train_test.py - isolates TE fused attention with the script's shapes.
import os
os.environ["NVTE_FUSED_ATTN"] = "1"
os.environ["NVTE_FLASH_ATTN"] = "0"
os.environ["NVTE_UNFUSED_ATTN"] = "0"

import torch
import transformer_engine.pytorch as te

S, B = 8192, 1          # seq length, micro batch (from the script)
H, G, D = 16, 4, 64     # heads, GQA groups, head dim (script's kv-channels)

attn = te.DotProductAttention(
    num_attention_heads=H,
    kv_channels=D,
    num_gqa_groups=G,
    attn_mask_type="causal",
    qkv_format="sbhd",
).cuda()

dt = torch.bfloat16
q = torch.randn(S, B, H, D, device="cuda", dtype=dt, requires_grad=True)
k = torch.randn(S, B, G, D, device="cuda", dtype=dt, requires_grad=True)
v = torch.randn(S, B, G, D, device="cuda", dtype=dt, requires_grad=True)

out = attn(q, k, v)
print("forward ok:", tuple(out.shape))
out.float().sum().backward()
print("backward ok")
