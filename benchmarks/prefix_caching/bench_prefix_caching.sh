#!/bin/bash
# Prefix-caching A/B benchmark: PC on vs off per model, plus a correctness probe.
#
# Daily -- PC gain on the fixed workload (4096-token shared prefix + 512 unique
# tokens, concurrency 16, plus a cheap concurrency-1 point that isolates the
# cache-hit path):
#   MODES="on off" TP="2" ./bench_prefix_caching.sh Qwen/Qwen3-14B tiiuae/falcon-mamba-7b
#   MODES="on off" TP="4" ./bench_prefix_caching.sh Qwen/Qwen3.8-27B Qwen/Qwen3.6-35B-A3B
# Curve -- PC gain vs. hit ratio (~0/25/50/75/89%; prefix 0 = nothing reused,
# i.e. PC's overhead). Once per platform on one representative model, ~5x the
# daily run time:
#   MODES="on off" PREFIX_LENS="0 1152 2304 3456 4096" CONCURRENCIES=16 \
#     ./bench_prefix_caching.sh Qwen/Qwen3.6-35B-A3B
#
# Read the results (each run also prints this at the end); run on both
# platforms, then put them side by side:
#   python3 analyze_pc.py bench_perf_results_graph/xpu
#   python3 analyze_pc.py --compare bench_perf_results_graph/cuda bench_perf_results_graph/xpu
#
# Other knobs (env vars, combine freely):
#   EXEC_MODE=eager             --enforce-eager instead of graph mode (default)
#   RETENTION=0                 sparse Mamba/sliding-window checkpoint retention
#   CONCURRENCIES="1 4 16 32"   load sweep
#   REPEATS=3                   rerun each point, to tell a small gap from noise
#   EXTRA_SERVER_ARGS="--kv-cache-dtype fp8" TAG=fp8kv
#                               feature combinations; TAG names their result tree
#   TP=4 CARDS=0,1,2,3          parallelism and devices
#   MOE_MAX_NUM_SEQS=64         --max-num-seqs for Qwen MoE (...-A<N>B) models
#                               (default 128; the Mamba-block limit is in the log)
#   SERVER_NUMA=2 CLIENT_NUMA=3 NUMA nodes for server / client (default 6 / 5,
#                               matching cards 4-7 here; empty = no binding)
#   PROBE_ARGS="--concurrent --evict"
#                               deeper correctness phases (default: a smoke test)
#   SKIP_EXISTING=0             rerun runs whose results already exist (default
#                               1 skips them, so an interrupted sweep resumes)
#
# Results: bench_perf_results_<exec_mode>/<platform>[_ret<N>][_<TAG>]/<model>/
# Each (model, on/off) gets one server; the prefix cache is reset and the seed
# changed before every run, so each run starts cold and never hits another's.

set -uo pipefail

[[ $# -gt 0 ]] || { echo "usage: [ENV=...] $0 MODEL [MODEL ...]   (see the header for examples)"; exit 1; }
models=("$@")
read -r -a modes <<< "${MODES:-on}"

for m in "${modes[@]}"; do
  [[ "$m" == "on" || "$m" == "off" ]] || { echo "mode must be on|off, got '$m'"; exit 1; }
done

PORT=${PORT:-9999}
CARDS=${CARDS:-4,5,6,7}
TP=${TP:-2}
# NUMA binding; empty = none. The server belongs on the node its cards hang off
# (on this box cards 4-7 -> node 6, 0-3 -> node 2; see
# /sys/class/drm/card*/device/numa_node, or `nvidia-smi topo -m` on CUDA): its
# scheduler and kernel-launch threads are on the critical path and its
# host<->device copies stay local. Memory is only preferred there, since loading
# weights can outgrow one node. The client goes on another node so the load
# generator never competes with the server.
SERVER_NUMA=${SERVER_NUMA:-6}
CLIENT_NUMA=${CLIENT_NUMA:-5}
server_numa=(); [[ -n "$SERVER_NUMA" ]] && server_numa=(numactl --cpunodebind="$SERVER_NUMA" --preferred="$SERVER_NUMA")
client_numa=(); [[ -n "$CLIENT_NUMA" ]] && client_numa=(numactl --cpunodebind="$CLIENT_NUMA" --membind="$CLIENT_NUMA")
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
# Mamba/sliding-window checkpoint retention: None = dense (old behavior),
# 0 = junctions/replay boundary only (upstream default), N = every N tokens.
RETENTION=${RETENTION:-None}
# Folder of .md prose the accuracy probe cuts prompts from; empty = probe's
# default (/workspace/vllm/docs, else <script dir>/workspace/vllm/docs).
PROBE_CORPUS=${PROBE_CORPUS:-}
# Extra pc_accuracy.py flags, e.g. "--concurrent --evict" for its deeper phases.
PROBE_ARGS=${PROBE_ARGS:-}
# Skip a run whose result json already exists, so an interrupted sweep resumes.
# Set 0 to rerun after changing vLLM or the environment: skipped runs keep the
# old build's numbers.
SKIP_EXISTING=${SKIP_EXISTING:-1}
# graph = vLLM's default compile + graph capture; eager = --enforce-eager.
EXEC_MODE=${EXEC_MODE:-graph}
case "$EXEC_MODE" in
  graph) exec_flag="" ;;
  eager) exec_flag="--enforce-eager" ;;
  *)     echo "EXEC_MODE must be graph|eager, got '$EXEC_MODE'"; exit 1 ;;
esac
# Extra `vllm serve` flags for feature combinations (fp8 KV, DCP, spec decode...).
# They change what is measured, so they need a TAG naming their own result tree.
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}
TAG=${TAG:-}
[[ -z "$EXTRA_SERVER_ARGS" || -n "$TAG" ]] || {
  echo "set TAG when using EXTRA_SERVER_ARGS, so its results don't mix with the defaults"; exit 1; }
# Qwen MoE models (Qwen/...-A<N>B, e.g. Qwen3.6-35B-A3B) get --max-num-seqs.
# The hybrid ones need it under graph mode: capture reserves one Mamba state
# block per decode slot, and 24 GB cards hold fewer blocks than vLLM's default
# 256 (188 at TP4 on B60). 128 is still far above any concurrency this script
# sends, so the measurement is unaffected; it applies on every platform so both
# sides run the same config.
MOE_MAX_NUM_SEQS=${MOE_MAX_NUM_SEQS:-128}

# Per-model `vllm serve` flags, printed for word-splitting into an array.
model_args() {
  case "$1" in
    Qwen/*-A[0-9]*B*) echo "--max-num-seqs $MOE_MAX_NUM_SEQS" ;;
  esac
}

# workload (see header)
PROMPT_LEN=${PROMPT_LEN:-4608}
read -r -a prefix_lens <<< "${PREFIX_LENS:-4096}"
read -r -a concurrencies <<< "${CONCURRENCIES:-16 1}"
NUM_PREFIXES=${NUM_PREFIXES:-5}
NUM_PROMPTS=${NUM_PROMPTS:-100}
OUTPUT_LEN=${OUTPUT_LEN:-128}
# c=1 points only measure TTFT: few requests and short outputs keep them cheap.
C1_NUM_PROMPTS=${C1_NUM_PROMPTS:-20}
C1_OUTPUT_LEN=${C1_OUTPUT_LEN:-16}
# Warm-up removes the first-run bias; repeats measure run-to-run noise. One run
# is enough when on/off or platform gaps are large; rerun a point with
# REPEATS=3 before drawing a conclusion from a gap of a few percent.
REPEATS=${REPEATS:-1}

# One entry per point: "<id> <prefix> <suffix> <num_prefixes> <concurrency>
# <num_prompts> <output_len>". The id spells out all of them, so analyze_pc.py
# can recompute what the point should hit.
points=()
for p in "${prefix_lens[@]}"; do
  (( p >= 0 && p < PROMPT_LEN )) || { echo "PREFIX_LENS entries must be in [0, PROMPT_LEN), got $p"; exit 1; }
  for c in "${concurrencies[@]}"; do
    if (( c == 1 )); then n=$C1_NUM_PROMPTS; o=$C1_OUTPUT_LEN; else n=$NUM_PROMPTS; o=$OUTPUT_LEN; fi
    if (( p == 0 )); then
      np=$n; pl=$((PROMPT_LEN / 2))   # one prefix per prompt: no reuse possible
    else
      np=$NUM_PREFIXES; pl=$p
    fi
    (( n % np == 0 )) || { echo "num prompts ($n) must be a multiple of NUM_PREFIXES ($np)"; exit 1; }
    sl=$((PROMPT_LEN - pl))
    points+=("p${pl}_s${sl}_n${np}_c${c}_N${n}_o${o} $pl $sl $np $c $n $o")
  done
done

if [[ -z "${PLATFORM:-}" ]]; then
  if command -v nvidia-smi > /dev/null 2>&1; then PLATFORM=cuda; else PLATFORM=xpu; fi
fi
case "$PLATFORM" in
  cuda) DEVICE_ENV_VAR=CUDA_VISIBLE_DEVICES ;;
  xpu)  DEVICE_ENV_VAR=ZE_AFFINITY_MASK ;;
  *)    echo "PLATFORM must be cuda|xpu"; exit 1 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Non-default retention and tagged configs get their own tree, so they never
# overwrite the default results.
ret_suffix=""; [[ "$RETENTION" != "None" ]] && ret_suffix="_ret$RETENTION"
OUTROOT=${OUTDIR:-"$SCRIPT_DIR/pc_bench_results_$EXEC_MODE/$PLATFORM$ret_suffix${TAG:+_$TAG}"}
mkdir -p "$OUTROOT"

SERVER_PID=""

# Only ever reaps our own process group -- a machine-wide `pkill -f VLLM::`
# would take down other users' servers on a shared box.
stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  kill -9 -$SERVER_PID 2>/dev/null || true   # negative pid = whole process group
  kill -9 $SERVER_PID 2>/dev/null || true
  for _ in $(seq 1 20); do
    pgrep -g $SERVER_PID > /dev/null 2>&1 || break
    sleep 1
  done
  pkill -9 -g $SERVER_PID 2>/dev/null || true
  SERVER_PID=""
  sleep 3
}
trap stop_server EXIT

# run_bench <model> <seed> <prefix> <suffix> <num_prefixes> <concurrency>
#           <num_prompts> <output_len> [extra bench args...]
run_bench() {
  local model=$1 seed=$2 pl=$3 sl=$4 np=$5 c=$6 n=$7 o=$8; shift 8
  "${client_numa[@]}" vllm bench serve \
    --model "$model" --host 0.0.0.0 --port "$PORT" \
    --dataset-name prefix_repetition \
    --prefix-repetition-prefix-len "$pl" \
    --prefix-repetition-suffix-len "$sl" \
    --prefix-repetition-num-prefixes "$np" \
    --prefix-repetition-output-len "$o" \
    --num-prompts "$n" \
    --request-rate inf --max-concurrency "$c" \
    --seed "$seed" --ignore-eos "$@"
}

start_server() {
  local model=$1 mode=$2 outdir=$3
  local pc_flag log="$outdir/server_$mode.log"
  case "$mode" in
    on)  pc_flag="--enable-prefix-caching" ;;
    off) pc_flag="--no-enable-prefix-caching" ;;
  esac

  echo "== launching $model on $PLATFORM ($EXEC_MODE, prefix caching: $mode, retention $RETENTION${TAG:+, tag $TAG}) cards $CARDS =="
  # setsid so the whole engine/worker tree lands in one process group we can
  # reap; killing only the `vllm serve` parent leaves VLLM::EngineCore and
  # VLLM::Worker_TP* alive, still holding all the device memory.
  # EXTRA_SERVER_ARGS and model_args are left unquoted on purpose: they hold
  # several flags.
  local server_args=(vllm serve "$model"
      $exec_flag --prefix-cache-retention-interval "$RETENTION" --host 0.0.0.0 --port "$PORT"
      --enable-prompt-tokens-details --trust-remote-code --tensor-parallel-size "$TP"
      --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
      $pc_flag $(model_args "$model") $EXTRA_SERVER_ARGS)
  SERVER_CMD="VLLM_SERVER_DEV_MODE=1 $DEVICE_ENV_VAR=$CARDS ${server_numa[*]} ${server_args[*]}"   # for run_meta.py
  setsid env VLLM_SERVER_DEV_MODE=1 "$DEVICE_ENV_VAR=$CARDS" "${server_numa[@]}" "${server_args[@]}" > "$log" 2>&1 &
  SERVER_PID=$!

  echo -n "waiting for server"
  for i in $(seq 1 180); do
    if curl -s -m 2 "http://0.0.0.0:$PORT/health" > /dev/null 2>&1; then echo " ready ($((i*5))s)"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo " DIED, see $log"; tail -40 "$log"; return 1; fi
    sleep 5; echo -n "."
  done
  curl -s -m 2 "http://0.0.0.0:$PORT/health" > /dev/null || { echo " timeout"; tail -40 "$log"; return 1; }

  # The block size / block count / cache modes the engine actually resolved.
  curl -s 0.0.0.0:$PORT/metrics | grep -E "^vllm:cache_config_info" > "$outdir/cache_config_$mode.txt"
  # Discarded warm-up, so first-request compile/allocation costs stay out of r1.
  echo "== warm-up =="
  run_bench "$model" 1 1024 128 2 4 8 16 > "$outdir/warmup_$mode.log" 2>&1 || true
}

# Fails while blocks are still held (requests draining), hence the retries.
reset_cache() {
  for _ in 1 2 3 4 5; do
    curl -s -m 10 -X POST "http://0.0.0.0:$PORT/reset_prefix_cache" | grep -q '"success": *true' && return 0
    sleep 2
  done
  return 1
}

scrape_counters() {
  curl -s 0.0.0.0:$PORT/metrics | grep -E "^vllm:prefix_cache_(hits|queries)" > "$1"
}

# counter <scrape> hits|queries: that counter summed over engines, 0 if absent.
counter() {
  awk -v n="vllm:prefix_cache_$2_total{" 'index($0, n) == 1 {s += $NF} END {printf "%d", s}' "$1" 2>/dev/null || echo 0
}

# Sets RUN_STATUS (OK|FAIL|NOHIT|HIT-OFF) and RUN_NOTE for the summary.
run_point() {
  local model=$1 mode=$2 outdir=$3 run=$4 seed=$5 pl=$6 sl=$7 np=$8 c=$9 n=${10} o=${11}
  scrape_counters "$outdir/metrics_before_$run.txt"
  run_bench "$model" "$seed" "$pl" "$sl" "$np" "$c" "$n" "$o" \
    --label "${PLATFORM}_pc_${mode}" \
    --save-result --result-dir "$outdir" --result-filename "bench_$run.json" \
    > "$outdir/bench_$run.log" 2>&1
  local rc=$?
  scrape_counters "$outdir/metrics_after_$run.txt"

  local hits queries hit ttft
  hits=$(( $(counter "$outdir/metrics_after_$run.txt" hits) - $(counter "$outdir/metrics_before_$run.txt" hits) ))
  queries=$(( $(counter "$outdir/metrics_after_$run.txt" queries) - $(counter "$outdir/metrics_before_$run.txt" queries) ))
  hit=$(awk -v h="$hits" -v q="$queries" 'BEGIN {if (q > 0) printf "%.1f%%", 100 * h / q; else print "-"}')
  ttft=$(awk -F: '/^Median TTFT/ {gsub(/ /, "", $2); print $2}' "$outdir/bench_$run.log")
  RUN_NOTE="hit $hit  ttft_p50 ${ttft:--} ms"
  # Functional bar: with caching on, a point that shares prefixes must hit;
  # with it off, nothing may hit.
  if (( rc != 0 )); then RUN_STATUS=FAIL
  elif [[ "$mode" == on ]] && (( np < n && hits == 0 )); then RUN_STATUS=NOHIT
  elif [[ "$mode" == off ]] && (( hits > 0 )); then RUN_STATUS=HIT-OFF
  else RUN_STATUS=OK
  fi
}

# Sets PROBE_STATUS from the probe's verdict (first word of its first line).
run_probe() {
  local model=$1 mode=$2 outdir=$3
  [[ "$mode" == off ]] || reset_cache || echo "WARN: prefix cache not reset before the probe"
  # A smoke test by default; PROBE_ARGS="--concurrent --evict" for the deeper
  # phases (eviction is pointless with caching off, so it is dropped there).
  local probe_args=() a
  for a in $PROBE_ARGS; do
    [[ "$mode" == off && "$a" == --evict ]] || probe_args+=("$a")
  done
  python3 "$SCRIPT_DIR/pc_accuracy.py" --model "$model" --port "$PORT" \
    ${PROBE_CORPUS:+--corpus "$PROBE_CORPUS"} ${probe_args[@]+"${probe_args[@]}"} \
    --out "$outdir/accuracy_$mode.json" 2>&1 | tee "$outdir/accuracy_$mode.txt"
  local verdict
  verdict=$(head -1 "$outdir/accuracy_$mode.txt" | awk '{print $1}')
  # With caching on, shared-prefix (sibling) reuse must happen and match;
  # NO-SIBLING-HIT fails too, since that reuse is what prefix caching is for.
  case "$mode:$verdict" in
    on:PASS|on:WARN|off:NO-HIT) PROBE_STATUS=OK ;;
    *)                          PROBE_STATUS=FAIL ;;
  esac
}

# run_meta_<mode>.json: versions, devices, env and the engine's resolved cache
# config, so runs are only compared when their conditions match. Called after
# stop_server because it opens the devices; never fails the sweep.
write_meta() {
  local model=$1 mode=$2 outdir=$3 status=$4
  env "$DEVICE_ENV_VAR=$CARDS" python3 "$SCRIPT_DIR/run_meta.py" \
    --out "$outdir/run_meta_$mode.json" --log "$outdir/server_$mode.log" \
    --metrics "$outdir/cache_config_$mode.txt" \
    --field model="$model" --field pc_mode="$mode" --field status="$status" \
    --field platform="$PLATFORM" --field exec_mode="$EXEC_MODE" --field retention="$RETENTION" \
    --field tag="$TAG" --field extra_server_args="$EXTRA_SERVER_ARGS" \
    --field cards="$CARDS" --field tp="$TP" --field max_model_len="$MAX_MODEL_LEN" \
    --field server_numa="$SERVER_NUMA" --field client_numa="$CLIENT_NUMA" \
    --field max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    --field prompt_len="$PROMPT_LEN" --field prefix_lens="${prefix_lens[*]}" \
    --field concurrencies="${concurrencies[*]}" --field repeats="$REPEATS" \
    --field num_prefixes="$NUM_PREFIXES" --field num_prompts="$NUM_PROMPTS" \
    --field output_len="$OUTPUT_LEN" --field c1_num_prompts="$C1_NUM_PROMPTS" \
    --field c1_output_len="$C1_OUTPUT_LEN" \
    --field server_cmd="$SERVER_CMD" \
    || echo "WARN: run_meta.py failed for $model [$mode]"
}

SERVER_CMD=""
summary=()
failures=0
for model in "${models[@]}"; do
  # "Qwen/Qwen3-8B" -> "Qwen_Qwen3-8B" so each model gets its own folder.
  outdir="$OUTROOT/${model//\//_}"
  mkdir -p "$outdir"
  echo "$model" > "$outdir/model.txt"

  for mode in "${modes[@]}"; do
    # Resume: skip the whole server launch when every run and the probe exist.
    todo=0
    for pt in "${points[@]}"; do
      for rep in $(seq 1 "$REPEATS"); do
        [[ -s "$outdir/bench_${mode}_${pt%% *}_r$rep.json" ]] || todo=$((todo + 1))
      done
    done
    if [[ "$SKIP_EXISTING" == "1" && $todo -eq 0 && -s "$outdir/accuracy_$mode.json" ]]; then
      echo "== skip $model [$mode]: all runs exist =="
      summary+=("SKIP $model $mode")
      continue
    fi

    mode_status=OK
    if ! start_server "$model" "$mode" "$outdir"; then
      summary+=("FAIL $model $mode: server did not start")
      failures=$((failures + 1))
      stop_server
      write_meta "$model" "$mode" "$outdir" FAIL
      continue
    fi

    seed=100
    for pt in "${points[@]}"; do
      read -r pid pl sl np c n o <<< "$pt"
      for rep in $(seq 1 "$REPEATS"); do
        seed=$((seed + 1))   # a fresh seed per run: fresh prompts, no stale hits
        run="${mode}_${pid}_r$rep"
        if [[ "$SKIP_EXISTING" == "1" && -s "$outdir/bench_$run.json" ]]; then
          summary+=("SKIP $model $run")
          continue
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || { summary+=("FAIL $model $run: server died"); failures=$((failures + 1)); mode_status=FAIL; break 2; }
        if [[ "$mode" == on ]] && ! reset_cache; then
          summary+=("WARN $model $run: prefix cache not reset (its fresh seed still isolates it)")
        fi
        echo "== $model $run =="
        run_point "$model" "$mode" "$outdir" "$run" "$seed" "$pl" "$sl" "$np" "$c" "$n" "$o"
        summary+=("$(printf '%-5s' "$RUN_STATUS") $model $run  $RUN_NOTE")
        echo "   $RUN_STATUS  $RUN_NOTE"
        [[ "$RUN_STATUS" == OK ]] || { failures=$((failures + 1)); mode_status=FAIL; }
      done
    done

    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "== accuracy probe =="
      run_probe "$model" "$mode" "$outdir"
      summary+=("$(printf '%-5s' "$PROBE_STATUS") $model $mode probe: $(head -1 "$outdir/accuracy_$mode.txt")")
      [[ "$PROBE_STATUS" == OK ]] || { failures=$((failures + 1)); mode_status=FAIL; }
    fi
    stop_server
    write_meta "$model" "$mode" "$outdir" "$mode_status"
  done
done

echo
echo "==== sweep summary ($OUTROOT) ===="
printf '%s\n' "${summary[@]}"
echo
python3 "$SCRIPT_DIR/analyze_pc.py" "$OUTROOT" || echo "WARN: analyze_pc.py failed"
exit $((failures > 0))
