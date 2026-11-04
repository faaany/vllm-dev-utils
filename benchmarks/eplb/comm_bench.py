#!/usr/bin/env python3
"""Time the collectives one MoE layer runs in DEP, to see how much of a step is
communication.

Same calls as vLLM's XPU default all2all backend (allgather_reducescatter:
AgRsAll2AllManager + xpu_communicator.py), Qwen3-30B-A3B shapes. Per MoE layer,
with t tokens on each of W ranks:
  dispatch  all_gather of hidden [t, 2048] bf16, topk_weights [t, 8] fp32 and
            topk_ids [t, 8] int32: three collectives
  combine   reduce_scatter of [W*t, 2048] bf16 -> [t, 2048]
A step runs this in all 48 layers. t = tokens per rank in the step: decode
~CONC, prefill ~MNBT. Equal t on every rank; uneven t takes a slower
list all_gather + cat in vLLM, not measured here.

Usage (in the container, free cards):
  torchrun --nproc-per-node 8 comm_bench.py [--tokens 64,2048]
  ZE_AFFINITY_MASK=0,1,2,3 torchrun --nproc-per-node 4 comm_bench.py
Times are the slowest rank's median over --iters; ranks are aligned with a
barrier before every timed call.
"""
import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist

HIDDEN, TOPK, LAYERS = 2048, 8, 48


def timed(fn, iters):
    times = []
    for _ in range(iters):
        dist.barrier()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    # the slowest rank decides when the step can go on
    ms = torch.tensor([statistics.median(times)], device="xpu")
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return ms.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="64,256,2048,8192", help="tokens per rank")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--ccl-local", action="store_true",
                    help="tell oneCCL the local rank / size from torchrun's variables; without it, "
                         "oneCCL warns it cannot find them and asks its OFI transport instead")
    args = ap.parse_args()

    if args.ccl_local:
        os.environ["CCL_PROCESS_LAUNCHER"] = "none"
        os.environ["CCL_LOCAL_RANK"] = os.environ["LOCAL_RANK"]
        os.environ["CCL_LOCAL_SIZE"] = os.environ["LOCAL_WORLD_SIZE"]
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.xpu.set_device(local_rank)
    dist.init_process_group("xccl")
    w, rank = dist.get_world_size(), dist.get_rank()
    if rank == 0:
        print(f"world {w}, {torch.xpu.get_device_name()}, torch {torch.__version__}")
        print(f"{'t/rank':>7} {'op':>16} {'out MiB':>8} {'ms':>8} {'GB/s':>7}")

    for t in (int(x) for x in args.tokens.split(",")):
        hidden = torch.randn(t, HIDDEN, dtype=torch.bfloat16, device="xpu")
        weights = torch.rand(t, TOPK, dtype=torch.float32, device="xpu")
        ids = torch.randint(0, 128, (t, TOPK), dtype=torch.int32, device="xpu")
        expert_out = torch.randn(w * t, HIDDEN, dtype=torch.bfloat16, device="xpu")
        outs = {name: torch.empty(w * x.shape[0], *x.shape[1:], dtype=x.dtype, device="xpu")
                for name, x in (("hidden", hidden), ("weights", weights), ("ids", ids))}
        combined = torch.empty_like(hidden)

        ops = {
            "ag hidden": (lambda: dist.all_gather_into_tensor(outs["hidden"], hidden), outs["hidden"]),
            "ag topk_weights": (lambda: dist.all_gather_into_tensor(outs["weights"], weights), outs["weights"]),
            "ag topk_ids": (lambda: dist.all_gather_into_tensor(outs["ids"], ids), outs["ids"]),
            "rs combine": (lambda: dist.reduce_scatter_tensor(combined, expert_out), expert_out),
        }

        def layer():
            for fn, _ in ops.values():
                fn()

        for fn, _ in ops.values():
            for _ in range(args.warmup):
                fn()
        for name, (fn, big) in ops.items():
            ms = timed(fn, args.iters)
            mib = big.numel() * big.element_size() / 2**20
            if rank == 0:
                print(f"{t:>7} {name:>16} {mib:>8.2f} {ms:>8.3f} {mib * 2**20 / ms / 1e6:>7.1f}")
        ms = timed(layer, args.iters)
        if rank == 0:
            print(f"{t:>7} {'one MoE layer':>16} {'':>8} {ms:>8.3f}   -> x{LAYERS} layers = {ms * LAYERS:.0f} ms/step")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
