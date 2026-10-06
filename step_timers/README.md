# step_timers

Measure the real CPU (wall-clock) time of individual functions inside a running
vLLM engine, per engine step, without a profiler.

It wraps the functions you list with a stopwatch (`time.perf_counter`) and,
every N engine steps, prints for each one the average time per step and the
number of calls per step to the server log. vLLM itself is not modified.

## When to use it
- You know a step is slower (e.g. TPOT regressed) and want to know **which
  function** costs the extra time, in real microseconds.
- The torch profiler is too heavy for that: with Python stacks it adds roughly
  2 µs per recorded call, more time than the step itself, and that overhead
  varies enough to hide or fake differences of a few percent.
- The cost here is ~1 µs per *wrapped* call. With a few dozen targets that's
  tens of µs per step, so it barely changes the step time.

## Files
| File | Purpose |
|---|---|
| `sitecustomize.py` | Python loads it at startup when this directory is on `PYTHONPATH`. If `VLLM_STEP_TIMERS=1`, it installs the timers right after vLLM imports its worker module (so only in engine processes). |
| `step_timers.py` | The timers: reads the targets file, wraps each function, prints reports. |
| `show.py` | Tabulates the reports from one or more server logs. |
| `targets/default.txt` | Minimal targets: `Worker.execute_model` and `Worker.sample_tokens`. |
| `targets/mrv2_vs_mrv1.txt` | Example: model runner V2 vs V1 breakdown (vLLM 0.29, DeepSeek-OCR). |

## Configuration
| Variable | Default | Meaning |
|---|---|---|
| `VLLM_STEP_TIMERS` | (off) | `1` enables the timers |
| `VLLM_STEP_TIMERS_TARGETS` | `targets/default.txt` | File with functions to time |
| `VLLM_STEP_TIMERS_ANCHOR` | `vllm.v1.worker.gpu_worker:Worker.sample_tokens` | A function called exactly once per engine step; defines "per step" and when to report |
| `VLLM_STEP_TIMERS_EVERY` | `1000` | Report every N steps |
| `VLLM_STEP_TIMERS_TRIGGER` | `vllm.v1.worker.gpu_worker` | Module whose import triggers installation |

### Targets file
One target per line, `#` starts a comment:
```
module:function
module:Class.method
```
- A target that doesn't exist (renamed, other vLLM version) is reported as
  `skipped` and ignored; vLLM keeps running.
- `module:function` replaces the name **as seen from that module**. To time a
  function that another module imports with `from x import f`, list the
  *calling* module, e.g. `vllm.v1.worker.gpu.sample.sampler:gumbel_sample`.
- To find names: `grep -n "    def \|^def \|^class " <file>` on the vLLM source,
  or look at function names in a torch-profiler trace.

## Usage (vLLM server in a container)
Put this directory where the container can see it (below: `/fanli/step_timers`).

1. Start the server with the timers enabled. Pin GPU and CPU socket for stable numbers:
   ```bash
   docker exec -e ZE_AFFINITY_MASK=3 -e PYTHONPATH=/fanli/step_timers -e VLLM_STEP_TIMERS=1 \
     -e VLLM_STEP_TIMERS_TARGETS=/fanli/step_timers/targets/mrv2_vs_mrv1.txt <container> \
     taskset -c 0-63 vllm serve <model> <args...> > a_server.log 2>&1 &
   ```
   If `PYTHONPATH` is already used, append: `-e PYTHONPATH=/fanli/step_timers:$EXISTING`.
2. Check it's active:
   ```bash
   grep -E "step_timers\] (installed|skipped|anchor)" a_server.log
   ```
   Expect `installed in pid ...: N targets ...` from the engine process.
3. Run your workload (e.g. `vllm bench serve ...`). Reports appear in the server
   log every `VLLM_STEP_TIMERS_EVERY` steps.
4. Stop the server, repeat for the other configuration (e.g. with
   `-e VLLM_USE_V2_MODEL_RUNNER=0`) into `b_server.log`.
5. Compare:
   ```bash
   python3 show.py a_server.log b_server.log
   ```
   One column per report window (µs per step), plus the median of all windows
   except the first.

Offline (`vllm bench latency`, `LLM(...)`) works the same: set the
environment variables for that process.

## Reading the numbers
- **Ignore the first window**: it includes warmup and one-time compilation.
- **Times are inclusive**: a function's time includes everything it calls.
  Subtract children to get a function's own time.
- **Check the totals**: per-step `Worker.execute_model` + `Worker.sample_tokens`
  should be close to the benchmark TPOT (for decode-only steps). If not,
  something between steps isn't covered.
- **GPU waits are included**: no synchronization is added, so a function that
  waits for the GPU (`.item()`, `.tolist()`, `Event.synchronize`, a blocking
  copy) includes the wait. When the GPU is mostly idle (CPU-bound steps), this
  is small; when the GPU is busy, waits can dominate and the time is not CPU work.
- **Compare like with like**: two code paths may do the same work in
  different functions (e.g. V2 computes logits inside `sample_tokens`, V1
  inside `execute_model`).
- **Steps are averaged over all step types** in the window. For a decode-only
  breakdown, use a workload where prefill steps are rare (long outputs) or
  check the per-call counts.
- Repeat runs and alternate configurations before concluding; one window or
  one run is not enough.
