#!/usr/bin/env python3
"""Time one rank's MoE expert kernel against the number of tokens routed to it.

Question it answers: in DEP decode, does the busiest rank's MoE layer take longer 
than a balanced one's? EPLB only shortens a step if it does.

What runs is the kernel vLLM uses for unquantized MoE on XPU,
vllm_xpu_kernels.fused_moe_interface.XpuFusedMoe (via XPUExperts), with
Qwen3-30B-A3B shapes: hidden 2048, moe_intermediate 768, top-8, SiLU, bf16,
weights in whichever layout the installed kernel expects from vLLM (see
KERNEL_TAKES_CHECKPOINT_LAYOUT).
Not included: router GEMM, top-k, the all-gather / reduce-scatter around it.

Shape of one call, as in DEP<EP> (--ep, default 8) with allgather_reducescatter:
  T  tokens on every rank's MoE = all DP ranks' tokens gathered
     (decode: T = EP x requests per rank)
  L  local experts per rank: 128 / EP (16 at EP8, 32 at EP4), one more with
     one redundant expert per rank
  A  (token, expert) pairs routed to this rank's L experts. Balanced:
     A = T x top-8 / EP. lam = A / balanced A, so the busiest rank at
     balancedness b has lam ~= 1 / b (b = mean / max rank load).
Each token gets its local slots on consecutive local experts (round robin),
so per-expert counts differ by at most 1 and as many experts as possible are
active; its other slots go to distinct random experts of other ranks, which
the kernel skips through expert_map.

Per point, the call is captured in an XPU graph (as in graph mode serving) and
replayed; time = median over trials of (event time / replays).

Derived, per T (an estimate: b is a per-step sum over layers, used here as
one layer's load):
  t_off  L=128/EP at lam = 1/b_off      t_on  L=128/EP+1 at lam = 1/b_on
The default b_off / b_on were measured at DEP8 on B60 (GSM8K); at another EP
pass the ones measured there (fake_off / on runs of bench_eplb_balance.sh).
  moe_speedup = t_off / t_on = the most EPLB can cut this layer's time at T

Usage (in the container, one free card):
  ZE_AFFINITY_MASK=2 python3 moe_microbench.py [--ep 8] [--tokens 8,16,...]
      [--lams 0,1,...] [--b-off 0.60 --b-on 0.83] [--out moe_microbench_results/xpu.csv]

"""
import argparse
import csv
import importlib.metadata
import os
import statistics
import time

import torch
import vllm_xpu_kernels.fused_moe_interface as fmi
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

HIDDEN, INTER, TOPK, NUM_EXPERTS = 2048, 768, 8, 128
EP = 8   # --ep
EXPERT_BYTES = (HIDDEN * 2 * INTER + INTER * HIDDEN) * 2   # w13 + w2, bf16

# Newer vllm_xpu_kernels (e.g. the CRI build) take the weights in checkpoint
# layout [E, N, K] and rewrite them into the device's kernel layout (xe_2, or
# xe_3 with B prepack) themselves; older ones (0.1.16.x) expect vLLM to have
# transposed them to [E, K, N] already.
KERNEL_TAKES_CHECKPOINT_LAYOUT = hasattr(fmi, "_to_xe2_layout")


def arch_flags():
    ops = torch.ops._xpu_C
    dev = torch.xpu.current_device()
    flags = {"xe2": ops.is_xe2_arch(), "xe3": ops.is_xe3_arch()}
    for name in ("is_cri", "is_nvl_p"):
        if hasattr(ops, name):
            flags[name[3:]] = getattr(ops, name)(dev)
    return flags


def make_routes(T, A, L, rank, gen):
    """topk_ids [T, TOPK] (global physical ids) putting A slots on rank's experts."""
    base, extra = divmod(A, T)
    assert base + (extra > 0) <= min(TOPK, L), f"A={A} too large for T={T}"
    k = torch.full((T,), base, dtype=torch.int64)
    k[torch.randperm(T, generator=gen)[:extra]] += 1

    others = torch.tensor([e for e in range(EP * L) if e // L != rank])
    pick = torch.rand(T, len(others), generator=gen).argsort(dim=1)[:, :TOPK]
    ids = others[pick]
    # Local slots: experts rank*L + (start .. start+k-1) mod L, start running on.
    start = torch.cumsum(k, 0) - k
    col = torch.arange(TOPK)
    local = rank * L + (start[:, None] + col[None, :]) % L
    ids = torch.where(col[None, :] < k[:, None], local, ids)
    return ids.to(torch.int32)


def time_point(L, T, A, rank, args, gen):
    dev = "xpu"
    if KERNEL_TAKES_CHECKPOINT_LAYOUT:
        w13_shape, w2_shape = (L, 2 * INTER, HIDDEN), (L, HIDDEN, INTER)
    else:
        w13_shape, w2_shape = (L, HIDDEN, 2 * INTER), (L, INTER, HIDDEN)
    w13 = (torch.randn(w13_shape, generator=gen) * 0.02).to(torch.bfloat16).to(dev)
    w2 = (torch.randn(w2_shape, generator=gen) * 0.02).to(torch.bfloat16).to(dev)
    expert_map = torch.full((EP * L,), -1, dtype=torch.int32)
    expert_map[rank * L:(rank + 1) * L] = torch.arange(L, dtype=torch.int32)
    expert_map = expert_map.to(dev)
    moe = XpuFusedMoe(w13=w13, w13_scales=None, w13_bias=None, w2=w2, w2_scales=None,
                      w2_bias=None, n_experts_per_token=TOPK, activation="silu",
                      num_experts=L, ep_rank=rank, ep_size=EP, expert_map=expert_map)

    x = (torch.randn(T, HIDDEN, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
    ids = make_routes(T, A, L, rank, gen).to(dev)
    w = torch.softmax(torch.randn(T, TOPK, generator=gen), dim=1).to(dev)
    out = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=dev)

    def call():
        moe.apply(output=out, hidden_states=x, topk_weights=w, topk_ids=ids)

    for _ in range(3):
        call()
    torch.xpu.synchronize()
    run = call
    if not args.eager:
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            call()
        run = g.replay
    for _ in range(args.iters):
        run()
    torch.xpu.synchronize()

    times = []
    for _ in range(args.trials):
        t0, t1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        t0.record()
        for _ in range(args.iters):
            run()
        t1.record()
        torch.xpu.synchronize()
        times.append(t0.elapsed_time(t1) * 1e3 / args.iters)   # us per call
    assert torch.isfinite(out).all(), "non-finite MoE output"
    del moe, w13, w2
    torch.xpu.empty_cache()
    return statistics.median(times), min(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens",
                    default="8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768",
                    help="T: tokens on each rank's MoE (decode: EP x requests per rank)")
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--lams", default="0,0.5,1,2",
                    help="lam = A / balanced A (T x top-8 / EP); 1/b = busiest rank at balancedness b; 1/b-off and 1/b-on are always added")
    ap.add_argument("--local-experts", help="default 128/EP and 128/EP + 1")
    ap.add_argument("--b-off", type=float, default=0.60)
    ap.add_argument("--b-on", type=float, default=0.83)
    ap.add_argument("--rank", type=int, help="which EP rank's experts are local (default EP/2)")
    ap.add_argument("--iters", type=int, default=100, help="replays per trial")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--eager", action="store_true", help="no XPU graph")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "moe_microbench_results", "xpu.csv"))
    args = ap.parse_args()
    global EP
    EP = args.ep
    if args.rank is None:
        args.rank = EP // 2
    L0 = NUM_EXPERTS // EP
    if args.local_experts is None:
        args.local_experts = f"{L0},{L0 + 1}"

    tokens = [int(t) for t in args.tokens.split(",")]
    lams = sorted({float(v) for v in args.lams.split(",")} | {1.0, 1 / args.b_off, 1 / args.b_on})
    Ls = [int(v) for v in args.local_experts.split(",")]
    gen = torch.Generator().manual_seed(args.seed)
    kernels = importlib.metadata.version("vllm_xpu_kernels")
    flags = arch_flags()
    layout = "checkpoint [E,N,K]" if KERNEL_TAKES_CHECKPOINT_LAYOUT else "pre-transposed [E,K,N]"
    print(f"device: {torch.xpu.get_device_name()}  torch {torch.__version__}  "
          f"vllm_xpu_kernels {kernels}  graph={not args.eager}")
    print(f"arch: {flags}  weights passed as {layout}  EP={EP} local experts {Ls}")
    # Upstream vLLM's XPUExperts only accepts Xe2/Xe3, so on anything else
    # serving may not run this kernel at all; check which MoE backend the
    # server log reports there.
    if not any(flags.values()):
        print("WARNING: no known Xe arch -- check vLLM actually picks XPUExperts on this card")
    t_start = time.time()

    rows = {}
    for L in Ls:
        for T in tokens:
            for lam in lams:
                A = round(lam * T * TOPK / EP)
                if A > min(TOPK, L) * T:
                    continue
                med, best = time_point(L, T, A, args.rank, args, gen)
                active = min(L, A)
                rows[L, T, lam] = {
                    "local_experts": L, "tokens": T, "lam": round(lam, 3), "routed": A,
                    "active_experts": active, "us_median": round(med, 2), "us_min": round(best, 2),
                    # weight bytes of the active experts over time: near the
                    # card's DRAM bandwidth = weight reads dominate
                    "weight_GBps": round(active * EXPERT_BYTES / med / 1e3, 1),
                }
                print(f"L={L:>2} T={T:>5} lam={lam:5.2f} A={A:>6}  {med:8.1f} us  "
                      f"{rows[L, T, lam]['weight_GBps']:6.1f} GB/s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        f.write(f"# {torch.xpu.get_device_name()} torch {torch.__version__} "
                f"vllm_xpu_kernels {kernels} graph={not args.eager} "
                f"iters={args.iters} trials={args.trials} arch={flags} weights={layout} ep={EP}\n")
        w = csv.DictWriter(f, fieldnames=next(iter(rows.values())).keys())
        w.writeheader()
        w.writerows(rows.values())
    print(f"\nwrote {args.out} ({time.time() - t_start:.0f} s)")

    lam_off, lam_on = 1 / args.b_off, 1 / args.b_on
    if L0 in Ls and L0 + 1 in Ls:
        print(f"\nper-layer MoE kernel, b_off={args.b_off} (L={L0}, lam={lam_off:.2f}) "
              f"vs b_on={args.b_on} (L={L0 + 1}, lam={lam_on:.2f}):")
        print(f"{'T':>6} {'balanced':>9} {'t_off':>8} {'t_on':>8} {'speedup':>8} "
              f"{'x48 off':>9}  (us; x48 = all MoE layers, ms)")
        for T in tokens:
            if (L0, T, lam_off) not in rows or (L0 + 1, T, lam_on) not in rows:
                continue
            bal = rows[L0, T, 1.0]["us_median"]
            t_off = rows[L0, T, lam_off]["us_median"]
            t_on = rows[L0 + 1, T, lam_on]["us_median"]
            print(f"{T:>6} {bal:>9.1f} {t_off:>8.1f} {t_on:>8.1f} {t_off / t_on:>8.3f} "
                  f"{48 * t_off / 1e3:>9.2f}")


if __name__ == "__main__":
    main()
