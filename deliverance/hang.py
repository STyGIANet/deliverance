"""How NCCL P2P matches a receive to a send when two messages have the same peer and size.

Rank 0 sends X (all 1s) then Y (all 2s) to rank 1, same size. Rank 1 posts the receive meant
for Y BEFORE the one meant for X. With X and Y on different communicators, each receive gets
its own message. On one communicator there are no tags: matching is by order only.
"""

import os

import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
rank = dist.get_rank()
comm_a = dist.new_group([0, 1])
comm_b = dist.new_group([0, 1])
for g in (comm_a, comm_b):
    dist.all_reduce(torch.zeros(1, device="cuda"), group=g)

N = 1 << 20

if os.environ.get("WARMUP_P2P", "1") == "1":
    # Establish the P2P connections on both communicators, in the SAME order on both ranks.
    for g in (comm_a, comm_b):
        t = torch.zeros(N, device="cuda")
        op = dist.isend if rank == 0 else dist.irecv
        for w in dist.batch_isend_irecv([dist.P2POp(op, t, 1 - rank, group=g)]):
            w.wait()
    torch.cuda.synchronize()
    if rank == 1:
        print("P2P warm-up done", flush=True)


def trial(comm_for_x, comm_for_y, label):
    if rank == 0:
        x = torch.full((N,), 1.0, device="cuda")
        y = torch.full((N,), 2.0, device="cuda")
        works = dist.batch_isend_irecv([dist.P2POp(dist.isend, x, 1, group=comm_for_x)])
        works += dist.batch_isend_irecv([dist.P2POp(dist.isend, y, 1, group=comm_for_y)])
    else:
        buf_x = torch.zeros(N, device="cuda")
        buf_y = torch.zeros(N, device="cuda")
        # Opposite order: the receive meant for Y is posted first.
        works = dist.batch_isend_irecv([dist.P2POp(dist.irecv, buf_y, 0, group=comm_for_y)])
        works += dist.batch_isend_irecv([dist.P2POp(dist.irecv, buf_x, 0, group=comm_for_x)])
    for w in works:
        w.wait()
    torch.cuda.synchronize()
    if rank == 1:
        got_x, got_y = buf_x.unique().tolist(), buf_y.unique().tolist()
        ok = got_x == [1.0] and got_y == [2.0]
        print(f"{label}: buffer for X got {got_x}, buffer for Y got {got_y} -> "
              f"{'correct' if ok else 'SWAPPED'}", flush=True)


trial(comm_a, comm_b, "two communicators")
trial(comm_a, comm_a, "one communicator ")
dist.barrier()
dist.destroy_process_group()
