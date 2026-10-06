"""Wall-clock timers for functions in a running vLLM engine.

Wraps the functions listed in a targets file with a stopwatch
(time.perf_counter). Every N engine steps it prints, for each function, the
average time per step and the number of calls per step, then resets.

Configuration (environment variables):
  VLLM_STEP_TIMERS=1                 enable (read by sitecustomize.py)
  VLLM_STEP_TIMERS_TARGETS=<file>    functions to time, one per line:
                                       module:function
                                       module:Class.method
                                     default: targets/default.txt next to this file
  VLLM_STEP_TIMERS_ANCHOR=<target>   function called exactly once per engine step;
                                     defines "per step" and when to report.
                                     default: vllm.v1.worker.gpu_worker:Worker.sample_tokens
  VLLM_STEP_TIMERS_EVERY=<n>         report every n steps (default 1000)

Notes:
  * Times are inclusive (a function's time includes the functions it calls).
  * No GPU synchronization is added: a function that waits for the GPU
    (.item(), .tolist(), Event.synchronize, ...) includes that waiting time.
  * "module:function" wraps the name as seen from that module. To time a
    function imported with "from x import f", list the module that calls it.
  * Targets that don't exist are reported as "skipped" and otherwise ignored.
"""
import collections
import functools
import importlib
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

EVERY = int(os.environ.get("VLLM_STEP_TIMERS_EVERY", "1000"))
ANCHOR = os.environ.get("VLLM_STEP_TIMERS_ANCHOR",
                        "vllm.v1.worker.gpu_worker:Worker.sample_tokens")
TARGETS_FILE = os.environ.get("VLLM_STEP_TIMERS_TARGETS",
                              os.path.join(_HERE, "targets", "default.txt"))

_time = collections.defaultdict(float)
_calls = collections.Counter()
_order = []    # targets in file order, for reports
_labels = {}   # target -> short label used in reports


def _log(msg):
    print(f"[step_timers] {msg}", file=sys.stderr, flush=True)


def read_targets(path):
    targets = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line and line not in targets:
                targets.append(line)
    return targets


def _make_labels(targets):
    """Short labels (last module component + attribute); full name if ambiguous."""
    short = {t: f"{t.split(':')[0].rsplit('.', 1)[-1]}:{t.split(':', 1)[1]}" for t in targets}
    counts = collections.Counter(short.values())
    return {t: (s if counts[s] == 1 else t) for t, s in short.items()}


def _report():
    steps = _calls[ANCHOR]
    lines = [f"pid {os.getpid()}: average over last {steps} steps (us per step, calls per step)"]
    for t in _order:
        if _calls[t]:
            lines.append(f"  {_time[t] / steps * 1e6:9.1f} us  x{_calls[t] / steps:4.1f}  {_labels[t]}")
    _log("\n[step_timers] ".join(lines))
    _time.clear()
    _calls.clear()


def _wrap(target, fn):
    @functools.wraps(fn)
    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _time[target] += time.perf_counter() - t0
            _calls[target] += 1
            if target == ANCHOR and _calls[target] >= EVERY:
                _report()
    return timed


def _resolve(target):
    module, attr = target.split(":", 1)
    owner = importlib.import_module(module)
    *path, name = attr.split(".")
    for part in path:
        owner = getattr(owner, part)
    return owner, name


def install():
    targets = read_targets(TARGETS_FILE)
    if ANCHOR not in targets:
        targets.insert(0, ANCHOR)
    _labels.update(_make_labels(targets))
    for target in targets:
        try:
            owner, name = _resolve(target)
            setattr(owner, name, _wrap(target, getattr(owner, name)))
            _order.append(target)
        except Exception as e:  # missing in this vLLM version: skip, don't break vLLM
            _log(f"skipped {target}: {e!r}")
    if ANCHOR not in _order:
        _log(f"anchor {ANCHOR} could not be wrapped: no reports will be printed")
    _log(f"installed in pid {os.getpid()}: {len(_order)} targets from {TARGETS_FILE}, "
         f"anchor {_labels.get(ANCHOR, ANCHOR)}, reporting every {EVERY} steps")
