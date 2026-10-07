# flash_attn_check.py - runs flash-attn forward and backward on the GPU.
import torch
from flash_attn import flash_attn_func

S, H, G, D = 8192, 16, 4, 64   # seq, heads, kv groups, head dim (from the train script)
dt = torch.bfloat16
q = torch.randn(1, S, H, D, device="cuda", dtype=dt, requires_grad=True)
k = torch.randn(1, S, G, D, device="cuda", dtype=dt, requires_grad=True)
v = torch.randn(1, S, G, D, device="cuda", dtype=dt, requires_grad=True)

out = flash_attn_func(q, k, v, causal=True)   # layout is (batch, seq, heads, dim)
out.float().sum().backward()
print("flash-attn forward and backward ok:", tuple(out.shape))
