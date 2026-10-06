"""Wall-clock timers for vLLM's per-step sampling path (model runner V1 and V2).

Wraps the functions listed in TARGETS. Every VLLM_STEP_TIMERS_EVERY steps
(default 1000), prints for each function the average time per step and the
number of calls per step, over the steps since the previous report.

Times are CPU wall-clock (time.perf_counter), with no GPU synchronization
added. A function that waits for the GPU (e.g. .tolist(), .item(),
Event.synchronize) includes that waiting time.
Timings are inclusive: a function's time includes the functions it calls.
"""
import collections
import functools
import importlib
import os
import sys
import time

EVERY = int(os.environ.get("VLLM_STEP_TIMERS_EVERY", "1000"))

# (module, attribute). "Class.method" wraps a method; "name" wraps a module-level
# function as seen from that module (where it is called from).
TARGETS = [
    # Worker level: both runners go through these.
    ("vllm.v1.worker.gpu_worker", "Worker.execute_model"),
    ("vllm.v1.worker.gpu_worker", "Worker.sample_tokens"),
    # Model (same for both runners; V2 calls it inside sample_tokens,
    # V1 calls it inside execute_model).
    ("vllm.model_executor.models.deepseek_ocr", "DeepseekOCRForCausalLM.compute_logits"),
    # --- V2 model runner (vllm/v1/worker/gpu/) ---
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner.sample_tokens"),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner.sample"),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner.postprocess_sampled"),
    ("vllm.v1.worker.gpu.model_runner", "post_update"),
    ("vllm.v1.worker.gpu.sample.sampler", "Sampler.__call__"),
    ("vllm.v1.worker.gpu.sample.sampler", "Sampler.sample"),
    ("vllm.v1.worker.gpu.sample.sampler", "Sampler.apply_sampling_params"),
    ("vllm.v1.worker.gpu.sample.sampler", "apply_top_k_top_p"),
    ("vllm.v1.worker.gpu.sample.sampler", "gumbel_sample"),
    ("vllm.v1.worker.gpu.sample.sampler", "get_num_sampled_and_rejected"),
    ("vllm.v1.worker.gpu.async_utils", "AsyncOutput.__init__"),
    ("vllm.v1.worker.gpu.async_utils", "AsyncOutput.get_output"),
    # --- V1 model runner (vllm/v1/worker/gpu_model_runner.py) ---
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner.sample_tokens"),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner._sample"),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner._update_states_after_model_execute"),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner._bookkeeping_sync"),
    ("vllm.v1.worker.gpu_model_runner", "AsyncGPUModelRunnerOutput.__init__"),
    ("vllm.v1.worker.gpu_model_runner", "AsyncGPUModelRunnerOutput.get_output"),
]

ANCHOR = "gpu_worker:Worker.sample_tokens"  # one call per engine step

_time = collections.defaultdict(float)
_calls = collections.Counter()


def _label(module, attr):
    return f"{module.rsplit('.', 1)[-1]}:{attr}"


def _report():
    steps = _calls[ANCHOR]
    lines = [f"[step_timers] pid {os.getpid()}: average over last {steps} steps"
             " (us per step, calls per step)"]
    for module, attr in TARGETS:
        lab = _label(module, attr)
        if _calls[lab]:
            lines.append(f"[step_timers]   {_time[lab] / steps * 1e6:9.1f} us"
                         f"  x{_calls[lab] / steps:4.1f}  {lab}")
    print("\n".join(lines), file=sys.stderr, flush=True)
    _time.clear()
    _calls.clear()


def _wrap(label, fn):
    @functools.wraps(fn)
    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _time[label] += time.perf_counter() - t0
            _calls[label] += 1
            if label == ANCHOR and _calls[label] >= EVERY:
                _report()
    return timed


def install():
    for module, attr in TARGETS:
        try:
            mod = importlib.import_module(module)
            owner, name = mod, attr
            if "." in attr:
                cls, name = attr.split(".", 1)
                owner = getattr(mod, cls)
            setattr(owner, name, _wrap(_label(module, attr), getattr(owner, name)))
        except Exception as e:  # missing in this vLLM version: skip, don't break vLLM
            print(f"[step_timers] skipped {module}:{attr}: {e!r}", file=sys.stderr, flush=True)
    print(f"[step_timers] installed in pid {os.getpid()}, reporting every {EVERY} steps",
          file=sys.stderr, flush=True)
