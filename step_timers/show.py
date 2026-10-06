#!/usr/bin/env python3
"""Tabulate step_timers reports from one or more vLLM server logs.

usage: python3 show.py <server_log> [<server_log> ...]

Each column is one report window (VLLM_STEP_TIMERS_EVERY steps), in us per step.
The first window includes warmup and one-time compilation: ignore it.
The last column is the median of the remaining windows.
"""
import collections
import re
import statistics
import sys

PAT = re.compile(r"\[step_timers\]\s+([\d.]+) us\s+x\s*([\d.]+)\s+(\S+)")

for path in sys.argv[1:]:
    rows = collections.defaultdict(list)
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if m:
            rows[m.group(3)].append(float(m.group(1)))
    n = max((len(v) for v in rows.values()), default=0)
    print(f"== {path}: {n} report windows (us per step)")
    for name, vals in rows.items():
        med = statistics.median(vals[1:]) if len(vals) > 1 else vals[0]
        print(f"  {name:62s} " + " ".join(f"{x:7.0f}" for x in vals) + f"   | median {med:7.0f}")
