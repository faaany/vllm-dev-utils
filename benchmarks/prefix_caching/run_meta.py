#!/usr/bin/env python3
"""Record what one bench_prefix_caching.sh run actually ran on: run_meta_<mode>.json.

Results are only comparable across platforms when the conditions behind them
match, and most of those conditions are invisible in bench_*.json: which vLLM
build served the run, eager vs. graph, the block size and KV capacity the engine
settled on, the device. This gathers them in one file per run:

  run       - the script's parameters (--field k=v), incl. the server command
  software  - installed versions; vLLM's git commit when it is a checkout
  hardware  - host, CPU, devices visible to the run (name, memory, driver)
  env       - VLLM_* / device-runtime env vars (anything secret-looking dropped)
  cache     - vllm:cache_config_info labels scraped from /metrics: the block
              size, block count and cache modes the engine really used
  server    - facts parsed from the server log: version, backends, KV size,
              compile/graph mode, warnings, first error

Run it after the server is stopped (it opens the devices to query them). Every
section is best-effort: a missing log or package leaves a gap, never a failure.

Usage: python3 run_meta.py --out run_meta_on.json --log server_on.log
         --metrics metrics_before_on.txt --field model=Qwen/Qwen3-8B ...
"""
import argparse
import datetime
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import socket
import subprocess

PACKAGES = ["vllm", "torch", "triton", "pytorch-triton-xpu", "vllm-xpu-kernels",
            "intel-extension-for-pytorch", "flashinfer-python", "transformers"]
ENV_PREFIXES = ("VLLM_", "ZE_", "SYCL_", "ONEAPI_", "CCL_", "UR_", "CUDA_", "NCCL_",
                "TORCH_", "PYTORCH_")
SECRET = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.I)


def value(s):
    """'16' -> 16, '0.9' -> 0.9, 'True' -> True; anything else stays a string."""
    if s in ("True", "False"):
        return s == "True"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def git(path, *args):
    try:
        return subprocess.run(["git", "-C", path, *args], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def software():
    out = {"python": platform.python_version(), "packages": {}}
    for name in PACKAGES:
        try:
            out["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    spec = importlib.util.find_spec("vllm")
    if spec and spec.origin:
        path = os.path.dirname(spec.origin)
        out["vllm_path"] = path
        # Only set for an editable/source checkout; a wheel keeps the commit in
        # its version string (e.g. 0.31.0.dev12+gabc1234) instead.
        commit = git(path, "rev-parse", "HEAD")
        if commit:
            out["vllm_commit"] = commit
            out["vllm_dirty"] = bool(git(path, "status", "--porcelain", "--untracked-files=no"))
    return out


def hardware(plat):
    out = {"host": socket.gethostname(), "kernel": platform.release()}
    try:
        cpu = re.search(r"^model name\s*:\s*(.*)$", open("/proc/cpuinfo").read(), re.M)
        out["cpu"] = cpu.group(1) if cpu else None
    except OSError:
        pass
    try:
        import torch
        dev = getattr(torch, plat)
        props = [dev.get_device_properties(i) for i in range(dev.device_count())]
        out["devices"] = [{"name": p.name,
                           "total_memory_gib": round(p.total_memory / 2**30, 2),
                           "driver_version": getattr(p, "driver_version", None)}
                          for p in props]
    except Exception as e:  # no torch / no device runtime: keep the rest
        out["devices_error"] = repr(e)
    if plat == "cuda":
        try:
            out["driver_version"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.split("\n")[0] or None
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def environment():
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(ENV_PREFIXES) and not SECRET.search(k)}


def cache_config(metrics_path):
    """Labels of vllm:cache_config_info, engine 0. None if never scraped."""
    try:
        text = open(metrics_path).read()
    except OSError:
        return None
    m = re.search(r"^vllm:cache_config_info\{(.*)\}", text, re.M)
    if not m:
        return None
    labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    labels.pop("engine", None)
    return {k: value(v) for k, v in sorted(labels.items())}


def first(pattern, text, cast=str):
    m = re.search(pattern, text, re.M)
    return cast(m.group(1)) if m else None


def server(log_path):
    try:
        text = open(log_path, errors="ignore").read()
    except OSError:
        return None
    # The engine-init line dumps the resolved config as k=v pairs.
    init = first(r"Initializing a V1 LLM engine .* with config: (.*)$", text) or ""
    config = {k: value(v) for k, v in re.findall(
        r"\b(enforce_eager|dtype|kv_cache_dtype|quantization|tensor_parallel_size|"
        r"pipeline_parallel_size|enable_prefix_caching)=([^,\s]+)", init)}
    for key, enum in (("compilation_mode", "CompilationMode"), ("cudagraph_mode", "CUDAGraphMode")):
        config[key] = first(rf"{enum}\.(\w+)", init)
    avail = [float(v) for v in re.findall(r"Available KV cache memory: ([\d.]+) GiB", text)]
    warnings = []
    for msg in re.findall(r"WARNING [\d-]+ [\d:]+ \[[^\]]+\] (.*)$", text, re.M):
        msg = msg[:200]
        if msg not in warnings:
            warnings.append(msg)
    return {
        "version": first(r"\bversion (\d[\w.+-]*)", text),
        "non_default_args": first(r"non-default args: (\{.*\})$", text),
        "engine_config": config,
        "backends": sorted(set(re.findall(r"Using (.{1,40}?) backend", text))),
        "attention_block_size_set": first(r"Setting attention block size to (\d+)", text, int),
        "kv_cache_tokens": first(r"KV cache size: ([\d,]+) tokens", text,
                                 lambda s: int(s.replace(",", ""))),
        # One line per worker; the smallest one bounds the whole engine.
        "available_kv_cache_gib_min": min(avail) if avail else None,
        "model_weights_gib": first(r"Model loading took ([\d.]+) GiB", text, float),
        "warnings": warnings[:30],
        "first_error": first(r"(\w+(?:Error|Exception): .*)$", text),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", help="server log of this run")
    ap.add_argument("--metrics", help="a /metrics scrape holding vllm:cache_config_info")
    ap.add_argument("--field", action="append", default=[], metavar="K=V",
                    help="run parameter to record; repeatable")
    args = ap.parse_args()

    run = {"timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    for kv in args.field:
        k, _, v = kv.partition("=")
        run[k] = value(v)

    meta = {"run": run, "software": software(),
            "hardware": hardware(run.get("platform", "xpu")), "env": environment(),
            "cache": cache_config(args.metrics) if args.metrics else None,
            "server": server(args.log) if args.log else None}
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=1)

    srv, cache = meta["server"] or {}, meta["cache"] or {}
    print(f"run_meta: vllm {srv.get('version') or meta['software']['packages'].get('vllm')}"
          f"  {run.get('exec_mode')}  block {cache.get('block_size')}"
          f"  kv {srv.get('kv_cache_tokens')} tok  -> {args.out}")


if __name__ == "__main__":
    main()
