#!/usr/bin/env python3
"""Summarize bench_prefix_caching.sh results: PC gain per workload point, the
cache hit rate against what the workload allows, and whether hits changed outputs.

A model folder holds one run per (on|off, point, repeat) -- bench_<mode>_<point>_r<k>.json
plus the prefix-cache counters scraped around it -- and the engine's cache config,
log and run_meta per mode. Folders from the older one-point script
(bench_<mode>.json) are read as that script's only point, LEGACY_POINT.

Per point, over repeats (median; +- is half the run-to-run range of TTFT p50):
  TTFT p50/p99 - the direct effect of PC and the headline; gain = off / on
  hits only    - concurrency 1: TTFT gain of the cache-hit requests alone, as a
                 fraction of the ideal gain (see hit_only); the cross-platform
                 comparison for PC performance
  TPOT p50     - decode speed; the median, since under load a tail of requests
                 decodes alongside other requests' prefill chunks
  throughput   - secondary: how much of the saving survives decode, which
                 depends on output length and load as much as on PC
  hit / exp    - measured hit rate vs. what the workload allows: per prefix the
                 first request misses and the rest reuse whole blocks of it,
                 (N - prefixes) * floor(prefix / block) * block hit tokens.
                 Falling short means eviction, concurrent misses or a retention
                 gap, not a slower PC; and a different block size per platform
                 alone moves exp, which is why both are shown.
Flags: hit below expected, cache pressure (shared prefixes + in-flight requests
vs. KV size), noisy points (spread > 10%), failed requests.
Correctness comes from accuracy_<mode>.json, written by pc_accuracy.py.

Usage: python3 analyze_pc.py [result_root_or_model_dir ...] [-v N]
       python3 analyze_pc.py --compare ROOT_A ROOT_B    # e.g. .../cuda .../xpu
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

from run_meta import cache_config, server as parse_server_log

POINT = re.compile(r"p(\d+)_s(\d+)_n(\d+)_c(\d+)_N(\d+)_o(\d+)$")
RUN = re.compile(r"^bench_(on|off)(?:_(.+)_r(\d+))?\.json$")
# The one point the old script ran; its files carry no point in their names.
# Kept apart from the new runs of the same workload: those ran under another
# build, batch budget and NUMA binding, so they must never pool as repeats.
LEGACY_POINT = "legacy_p4096_s512_n5_c16_N100_o128"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_counters(path):
    """prefix_cache_{hits,queries}_total summed over engines; None if missing."""
    try:
        text = open(path).read()
    except OSError:
        return None
    out = {}
    for name in ("hits", "queries"):
        vals = re.findall(rf"^vllm:prefix_cache_{name}_total\{{.*\}} (\S+)$", text, re.M)
        if not vals:
            return None
        out[name] = sum(float(v) for v in vals)
    return out


def hit_rate(model_dir, tag):
    before = read_counters(os.path.join(model_dir, f"metrics_before_{tag}.txt"))
    after = read_counters(os.path.join(model_dir, f"metrics_after_{tag}.txt"))
    if after is None:
        return None
    before = before or {"hits": 0.0, "queries": 0.0}
    queries = after["queries"] - before["queries"]
    return (after["hits"] - before["hits"]) / queries if queries > 0 else 0.0


def point_params(pid):
    m = POINT.search(pid)
    keys = ("prefix", "suffix", "prefixes", "conc", "prompts", "output")
    return dict(zip(keys, map(int, m.groups()))) if m else None


def engine(model_dir, mode):
    """Block size and KV capacity (tokens) the engine settled on, None if unknown."""
    cache = cache_config(os.path.join(model_dir, f"cache_config_{mode}.txt")) or {}
    meta = load_json(os.path.join(model_dir, f"run_meta_{mode}.json")) or {}
    srv = meta.get("server") or parse_server_log(os.path.join(model_dir, f"server_{mode}.log")) or {}
    block = cache.get("block_size") or srv.get("attention_block_size_set")
    kv = srv.get("kv_cache_tokens")
    if kv is None and cache.get("num_gpu_blocks") and block:
        kv = cache["num_gpu_blocks"] * block
    exec_mode = (meta.get("run") or {}).get("exec_mode")
    if exec_mode is None and (srv.get("engine_config") or {}).get("enforce_eager") is not None:
        exec_mode = "eager" if srv["engine_config"]["enforce_eager"] else "graph"
    return {"block": block, "kv": kv, "exec": exec_mode}


def expected_hit(p, block):
    if not p or not block:
        return None
    reuse = (p["prompts"] - p["prefixes"]) * (p["prefix"] // block) * block
    return reuse / (p["prompts"] * (p["prefix"] + p["suffix"]))


def working_set(p):
    """Tokens that must stay resident for every expected hit to land: each
    shared prefix once (in-flight requests share its blocks), plus every
    in-flight request's own tokens."""
    if p["prefixes"] < p["prompts"]:
        return p["prefixes"] * p["prefix"] + p["conc"] * (p["suffix"] + p["output"])
    return p["conc"] * (p["prefix"] + p["suffix"] + p["output"])


def collect(model_dir):
    """{(point, mode): [bench result + its hit rate]} for one model folder."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "bench_*.json"))):
        m = RUN.match(os.path.basename(path))
        r = load_json(path) if m else None
        if r is None:
            continue
        mode, pid, rep = m.groups()
        r["hit"] = hit_rate(model_dir, f"{mode}_{pid}_r{rep}" if pid else mode)
        runs.setdefault((pid or LEGACY_POINT, mode), []).append(r)
    return runs


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def aggregate(rs):
    ttft = [r["median_ttft_ms"] for r in rs]
    mid = median(ttft)
    return {"runs": len(rs), "ttft50": mid,
            "ttft99": median(r["p99_ttft_ms"] for r in rs),
            "tpot": median(r["median_tpot_ms"] for r in rs),
            "thpt": median(r["total_token_throughput"] for r in rs),
            "hit": median(r["hit"] for r in rs),
            "spread": (max(ttft) - min(ttft)) / 2 / mid if len(rs) > 1 and mid else None,
            "failed": sum(r.get("failed") or 0 for r in rs)}


def ratio(a, b):
    return a / b if a and b else None


def gains(agg):
    on, off = agg.get("on"), agg.get("off")
    if not on or not off:
        return None
    return {"ttft50": ratio(off["ttft50"], on["ttft50"]),
            "ttft99": ratio(off["ttft99"], on["ttft99"]),
            "thpt": ratio(on["thpt"], off["thpt"])}


def flags(p, agg, eng):
    out = []
    on, off = agg.get("on"), agg.get("off")
    exp = expected_hit(p, eng["block"])
    if on and on["hit"] is not None and exp and on["hit"] < 0.9 * exp:
        out.append(f"hit {on['hit']:.0%} < expected {exp:.0%}")
    if on and on["hit"] == 0 and p and p["prefixes"] < p["prompts"]:
        out.append("caching on but no hits")
    if off and (off["hit"] or 0) > 0:
        out.append("hits while caching was off")
    # Pressure only matters where hits are expected.
    if p and p["prefixes"] < p["prompts"] and eng["kv"] and working_set(p) > 0.8 * eng["kv"]:
        out.append(f"cache pressure: working set {working_set(p):,} vs KV {eng['kv']:,} tok")
    for mode, a in agg.items():
        if a["spread"] and a["spread"] > 0.10:
            out.append(f"{mode} noisy: TTFT p50 +-{a['spread']:.0%}")
        if a["failed"]:
            out.append(f"{mode}: {a['failed']} failed requests")
    return out


def fmt(v, spec, scale=1):
    return "-" if v is None else format(v * scale, spec)


def point_order(pid):
    p = point_params(pid) or {}
    # Per concurrency: the no-reuse (overhead) point first, then by prefix length.
    return (p.get("conc", 0), p.get("prefixes", 0) != p.get("prompts"), p.get("prefix", 0), pid)


def accuracy(model_dir, mode, verbose):
    acc = load_json(os.path.join(model_dir, f"accuracy_{mode}.json"))
    if acc is None:
        return "n/a (no accuracy probe; rerun with the current bench script)"
    reqs = [r for r in acc["requests"] if r["verdict"] != "same"]
    reqs.sort(key=lambda r: (r["verdict"] != "MISMATCH", -(r["first_dlp"] or 0)))
    for r in reqs[:verbose]:
        print(f"    [{mode}] {r['verdict']:<8} {r.get('phase', 'seq'):<7} len {r.get('length', '-')} "
              f"prefix {r['prefix']} q{r['question']} @tok {r['diverge_at']}  "
              f"cached {r['cached_tokens']}/{r['prompt_len']}  first_dlp {r['first_dlp']:.3f}\n"
              f"      hit: {r['hit_text'][:90]!r}\n      ref: {r['ref_text'][:90]!r}")
    return acc["summary"]


def model_dirs(paths):
    for p in paths:
        if os.path.exists(os.path.join(p, "model.txt")):
            yield p
        elif os.path.isdir(p):
            for d in sorted(os.listdir(p)):
                if os.path.exists(os.path.join(p, d, "model.txt")):
                    yield os.path.join(p, d)


def hit_only(p, block, on_runs, off_runs):
    """Concurrency-1 TTFT gain of the cache-hit requests alone, against the ideal.

    Results keep per-request end-to-end latency, not TTFT; at concurrency 1 a
    request's TTFT is its latency minus (output - 1) decode steps. The hit-rate
    counters say how many requests hit (each hit reuses `cached` tokens); those
    are the fastest ones. Ideal gain = prompt / uncached tokens, i.e. prefill
    time proportional to the tokens computed; efficiency = measured / ideal,
    comparable across platforms since each is divided by its own ideal."""
    if not p or p["conc"] != 1 or not block or p["prefixes"] >= p["prompts"]:
        return None
    prompt, cached = p["prefix"] + p["suffix"], p["prefix"] // block * block
    if not cached or not on_runs or not off_runs:
        return None
    ttfts = []
    for r in on_runs:
        if not r.get("latencies") or not r.get("hit"):
            continue
        hits = round(r["hit"] * p["prompts"] * prompt / cached)
        est = sorted(x * 1000 - (p["output"] - 1) * r["median_tpot_ms"] for x in r["latencies"])
        if hits > 0:
            ttfts.append(statistics.median(est[:hits]))
    off = median(r["median_ttft_ms"] for r in off_runs)
    if not ttfts or not off:
        return None
    ttft = median(ttfts)
    gain, ideal = off / ttft, prompt / (prompt - cached)
    return {"ttft": ttft, "gain": gain, "ideal": ideal, "eff": gain / ideal}


def load_model(d):
    """Everything one model folder holds, aggregated per point and mode."""
    runs = collect(d)
    eng = engine(d, "on")
    pts = {}
    for (pid, mode), rs in runs.items():
        pts.setdefault(pid, {})[mode] = aggregate(rs)
    hit = {pid: h for pid in pts
           if (h := hit_only(point_params(pid), eng["block"],
                             runs.get((pid, "on")), runs.get((pid, "off"))))}
    return {"model": open(os.path.join(d, "model.txt")).read().strip(), "dir": d,
            "engine": eng, "points": pts, "hit_only": hit}


def report(d, verbose):
    m = load_model(d)
    eng = m["engine"]
    print(f"\n{m['model']}   exec {eng['exec'] or '?'}  block {eng['block'] or '?'}  "
          f"KV {fmt(eng['kv'], ',')} tok")
    if not m["points"]:
        print("  no bench results")
    else:
        print(f"  {'point':<36} mode runs  ttft_p50     +-   ttft_p99 tpot_p50     tok/s   hit%   exp%")
    for pid in sorted(m["points"], key=point_order):
        p, agg = point_params(pid), m["points"][pid]
        exp = expected_hit(p, eng["block"])
        for mode in ("on", "off"):
            a = agg.get(mode)
            if a is None:
                continue
            print(f"  {pid:<36} {mode:<4} {a['runs']:>4}  {fmt(a['ttft50'], '8.0f')} "
                  f"{fmt(a['spread'], '4.0%'):>5}  {fmt(a['ttft99'], '9.0f')}  {fmt(a['tpot'], '7.1f')}  "
                  f"{fmt(a['thpt'], '8.0f')}  {fmt(a['hit'], '5.1f', 100)}  "
                  f"{fmt(exp if mode == 'on' else None, '5.1f', 100)}")
        g = gains(agg)
        if g:
            print(f"  {'':<36} gain: TTFT p50 x{fmt(g['ttft50'], '.2f')}  "
                  f"p99 x{fmt(g['ttft99'], '.2f')}  throughput x{fmt(g['thpt'], '.2f')}")
        if (h := m["hit_only"].get(pid)):
            print(f"  {'':<36} hits only: TTFT p50 {h['ttft']:.0f} ms, gain x{h['gain']:.2f} "
                  f"of ideal x{h['ideal']:.2f} = {h['eff']:.0%} efficiency")
        for msg in flags(p, agg, eng):
            print(f"  {'':<36} WARN: {msg}")
    for mode in ("on", "off"):
        if os.path.exists(os.path.join(d, f"accuracy_{mode}.json")):
            print(f"  acc {mode:<3}: {accuracy(d, mode, verbose)}")


def compare(root_a, root_b):
    """PC gains of the same models and points on two result roots, side by side."""
    label_a, label_b = (os.path.basename(os.path.normpath(r)) for r in (root_a, root_b))
    a = {m["model"]: m for m in map(load_model, model_dirs([root_a]))}
    b = {m["model"]: m for m in map(load_model, model_dirs([root_b]))}
    print(f"PC gain (off / on for TTFT, on / off for throughput): {label_a} | {label_b}")
    for model in sorted(set(a) & set(b)):
        ma, mb = a[model], b[model]
        ea, eb = ma["engine"], mb["engine"]
        print(f"\n{model}   block {ea['block'] or '?'} | {eb['block'] or '?'}   "
              f"KV {fmt(ea['kv'], ',')} | {fmt(eb['kv'], ',')} tok   "
              f"exec {ea['exec'] or '?'} | {eb['exec'] or '?'}")
        print(f"  {'point':<36}  {'hit% (exp%)':^25}  {'TTFT p50 gain':^15}  "
              f"{'TTFT p99 gain':^15}  {'throughput gain':^15}")
        for pid in sorted(set(ma["points"]) & set(mb["points"]), key=point_order):
            p = point_params(pid)
            ga, gb = gains(ma["points"][pid]), gains(mb["points"][pid])
            ha = (ma["points"][pid].get("on") or {}).get("hit")
            hb = (mb["points"][pid].get("on") or {}).get("hit")
            hits = (f"{fmt(ha, '.1f', 100)} ({fmt(expected_hit(p, ea['block']), '.0f', 100)}) | "
                    f"{fmt(hb, '.1f', 100)} ({fmt(expected_hit(p, eb['block']), '.0f', 100)})")
            cols = [f"x{fmt((ga or {}).get(k), '.2f')} | x{fmt((gb or {}).get(k), '.2f')}"
                    for k in ("ttft50", "ttft99", "thpt")]
            print(f"  {pid:<36}  {hits:^25}  {cols[0]:^15}  {cols[1]:^15}  {cols[2]:^15}")
            ha, hb = ma["hit_only"].get(pid), mb["hit_only"].get(pid)
            if ha or hb:
                print(f"  {'':<36}  hits only, efficiency (gain / ideal): "
                      f"{fmt((ha or {}).get('eff'), '.0%')} | {fmt((hb or {}).get('eff'), '.0%')}  "
                      f"(x{fmt((ha or {}).get('gain'), '.2f')} / x{fmt((ha or {}).get('ideal'), '.2f')} | "
                      f"x{fmt((hb or {}).get('gain'), '.2f')} / x{fmt((hb or {}).get('ideal'), '.2f')})")
            for label, m in ((label_a, ma), (label_b, mb)):
                for msg in flags(p, m["points"][pid], m["engine"]):
                    print(f"  {'':<36}  WARN {label}: {msg}")
    for label, only in ((label_a, set(a) - set(b)), (label_b, set(b) - set(a))):
        if only:
            print(f"\nonly in {label}: {', '.join(sorted(only))}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*",
                    default=[os.path.join(here, "bench_perf_results_graph", "xpu")])
    ap.add_argument("-v", "--verbose", type=int, default=0, metavar="N",
                    help="print the N worst probe divergences per model and mode")
    ap.add_argument("--compare", nargs=2, metavar=("ROOT_A", "ROOT_B"),
                    help="side-by-side PC gains of two result roots (e.g. cuda vs xpu)")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    dirs = list(model_dirs(args.paths))
    if not dirs:
        sys.exit(f"no model folders (with model.txt) under {args.paths}")
    print("(latencies in ms; hit% = prefix-cache hits / queries during the run; "
          "exp% = what the workload allows at this block size)")
    for d in dirs:
        report(d, args.verbose)


if __name__ == "__main__":
    main()
