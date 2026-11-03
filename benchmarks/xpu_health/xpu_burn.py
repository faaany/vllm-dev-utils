#!/usr/bin/env python3
"""Load one XPU with bf16 matmuls and report its throughput.

Usage: ZE_AFFINITY_MASK=<xpu> python3 xpu_burn.py [seconds]
Prints TFLOPS every 5 s, then one machine-readable line:
  RESULT <pci bus> <mean TFLOPS over the last 10 s>
The PCI bus comes from the device uuid (e.g. 868011e2-0000-0000-4200-... -> 42).
"""
import sys
import time

import torch

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 120
bus = str(torch.xpu.get_device_properties(0).uuid).split("-")[3][:2]
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="xpu")
b = torch.randn(8192, 8192, dtype=torch.bfloat16, device="xpu")
flop = 2 * 8192**3

windows = []   # (seconds since start, TFLOPS)
t0 = last = time.time()
n = 0
while time.time() - t0 < secs:
    for _ in range(20):
        a @ b
    torch.xpu.synchronize()
    n += 20
    if (now := time.time()) - last >= 5:
        windows.append((now - t0, n * flop / (now - last) / 1e12))
        print(f"bus {bus} {windows[-1][0]:5.0f}s {windows[-1][1]:6.1f} TFLOPS", flush=True)
        n, last = 0, now

tail = [tf for t, tf in windows if t >= secs - 10] or [tf for _, tf in windows]
print(f"RESULT {bus} {sum(tail) / len(tail):.1f}", flush=True)
