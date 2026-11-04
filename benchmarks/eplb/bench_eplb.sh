#!/bin/bash
# EPLB benchmark: for each mode, start a server, prime it past the first
# expert rearrangement, then measure.
#
#   GPU=cri PHASE=decode ./bench_eplb.sh [model]
#   GPU=b70 PHASE=prefill MODES="fake_off on" ./bench_eplb.sh
#   GPU=b70 PHASE=decode SMOKE=1 ./bench_eplb.sh    # short check before a real run
#
# Modes (all use expert parallel, so they differ only in EPLB):
#   off       no EPLB
#   fake_off  EPLB on, but never rearranges: the baseline, logs balancedness
#   on        EPLB rearranges every STEP_INTERVAL engine steps
set -euo pipefail

MODEL=${1:-Qwen/Qwen3-30B-A3B}
GPU=${GPU:-cri}            # cri | b70, only names the result folder for now
PHASE=${PHASE:-decode}     # decode | prefill
MODES=${MODES:-"off fake_off on"}
SMOKE=${SMOKE:-0}          # 1: same server settings, fewer steps

# ---------------- settings (vLLM defaults where there is one) ----------------
CARDS=0,1,2,3,4,5,6,7
DP=8
MNBT=8192                  # max_num_batched_tokens
GPU_MEM_UTIL=0.89          # not the default 0.92: DP starts DP API servers, which hold
                           # ~2.9 GiB on xpu:0 together; B70 30.3 GiB -> at most 0.90
STEP_INTERVAL=3000         # EPLB: rearrange every N engine steps
WINDOW=1000                # EPLB: load statistics over the last N steps
REDUNDANT=8                # EPLB: redundant experts, 1 per rank
PROMPT_LEN=723             # mean tokens of a prompt in data/gsm8k_5shot.jsonl (measured)

PRIME_STEPS=3200           # past the first rearrangement at step 3000
MEASURE_STEPS=1000         # must end before the second one at step 6000

case "$PHASE" in
  # Every step is decode: CONC tokens per rank. With --ignore-eos all requests
  # run OUTPUT_LEN steps, so CONC x DP requests take ~OUTPUT_LEN steps.
  decode)  CONC=256; OUTPUT_LEN=512; MAX_NUM_SEQS=256; PREFIX_CACHING=--enable-prefix-caching ;;
  # Every step prefills MNBT tokens on each of the DP ranks, so it finishes
  # about DP x MNBT / PROMPT_LEN requests (8 x 8192 / 723 = ~90). CONC does not
  # change the step count; it only has to keep MNBT full on every rank
  # (>= MNBT / PROMPT_LEN = ~12 requests per rank; 64 leaves room for uneven DP routing).
  prefill) CONC=64;  OUTPUT_LEN=1;   MAX_NUM_SEQS=128; PREFIX_CACHING=--no-enable-prefix-caching ;;
  *)       echo "PHASE must be decode|prefill, got '$PHASE'"; exit 1 ;;
esac

# Smoke: server settings (CONC, OUTPUT_LEN, MNBT, ...) stay as above, so KV and
# memory problems still show; only the interval and step counts shrink.
if [[ "$SMOKE" == 1 ]]; then
  case "$PHASE" in
    decode)  STEP_INTERVAL=1000; PRIME_STEPS=1010; MEASURE_STEPS=500 ;;   # 2 + 1 batches of 512 steps
    prefill) STEP_INTERVAL=100;  PRIME_STEPS=110;  MEASURE_STEPS=50  ;;
  esac
  WINDOW=$STEP_INTERVAL
fi

# steps_to_prompts <steps>: how many requests keep the server busy that long.
steps_to_prompts() {
  if [[ "$PHASE" == decode ]]; then
    # a batch is CONC x DP requests running OUTPUT_LEN steps side by side
    local batches=$(( ($1 + OUTPUT_LEN - 1) / OUTPUT_LEN ))   # ceil(steps / OUTPUT_LEN)
    echo $(( batches * CONC * DP ))
  else
    echo $(( $1 * DP * MNBT / PROMPT_LEN ))
  fi
}
PRIME_PROMPTS=$(steps_to_prompts $PRIME_STEPS)
MEASURE_PROMPTS=$(steps_to_prompts $MEASURE_STEPS)
# -----------------------------------------------------------------------------

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=$SCRIPT_DIR/eplb_bench_results/${GPU}_${PHASE}$([[ "$SMOKE" == 1 ]] && echo _smoke)_$(date +%m%d_%H%M)
mkdir -p "$OUT_DIR"

# GSM8K test set as 5-shot prompts, built like vLLM's tests/evals/gsm8k/gsm8k_eval.py.
# Downloaded on first use into a local cache, not kept in the repo.
DATASET=${XDG_CACHE_HOME:-$HOME/.cache}/eplb_bench/gsm8k_5shot.jsonl
if [[ ! -s "$DATASET" ]]; then
  echo "building $DATASET"
  python3 - "$DATASET" <<'EOF'
import json, os, sys, urllib.request
url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/ci-datasets/gsm8k"
load = lambda name: [json.loads(l) for l in urllib.request.urlopen(f"{url}/{name}").read().decode().splitlines()
                     if l.strip() and not l.startswith("#")]
train, test = load("train.jsonl"), load("test.jsonl")
shots = "".join(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n" for ex in train[:5])
os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True)
with open(sys.argv[1] + ".tmp", "w") as f:
    for ex in test:
        f.write(json.dumps({"prompt": f"{shots}Question: {ex['question']}\nAnswer:"}) + "\n")
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
EOF
fi

echo "== $GPU $PHASE$([[ "$SMOKE" == 1 ]] && echo " SMOKE"), modes: $MODES"
echo "   DP$DP, conc $CONC/rank, mnbt $MNBT, output $OUTPUT_LEN, interval $STEP_INTERVAL"
echo "   prime $PRIME_PROMPTS prompts (~$PRIME_STEPS steps), measure $MEASURE_PROMPTS prompts (~$MEASURE_STEPS steps)"
echo "   -> $OUT_DIR"

eplb_args() {
  local log="\"window_size\":$WINDOW,\"use_async\":false,\"log_balancedness\":true,\"log_balancedness_interval\":10"
  case "$1" in   # use_async false: async EPLB never finishes on XPU
    fake_off) echo "--enable-eplb --eplb-config {\"num_redundant_experts\":0,\"step_interval\":1000000000,$log}" ;;
    on)       echo "--enable-eplb --eplb-config {\"num_redundant_experts\":$REDUNDANT,\"step_interval\":$STEP_INTERVAL,$log}" ;;
  esac
}

SERVER_PID=""
stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  kill -- -"$SERVER_PID" 2>/dev/null || true   # the whole process group (setsid)
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  sleep 5   # let the cards release memory
}
trap stop_server EXIT
trap 'exit 130' INT TERM HUP PIPE   # so Ctrl-C, kill or a closed terminal also stop the server

start_server() {
  local mode=$1 log="$OUT_DIR/server_$1.log"
  # shellcheck disable=SC2046  # eplb_args is split on purpose; its JSON has no spaces
  setsid env PYTHONUNBUFFERED=1 ZE_AFFINITY_MASK=$CARDS vllm serve "$MODEL" \
    --host 0.0.0.0 --port 9999 --gpu-memory-utilization $GPU_MEM_UTIL \
    --data-parallel-size $DP --enable-expert-parallel $(eplb_args "$mode") $PREFIX_CACHING \
    --max-num-batched-tokens $MNBT --max-num-seqs $MAX_NUM_SEQS --max-model-len 2048 \
    > "$log" 2>&1 &
  SERVER_PID=$!

  echo -n "waiting for server ($mode)"
  for i in $(seq 1 240); do
    curl -s -m 2 http://0.0.0.0:9999/health > /dev/null && { echo " ready"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo " DIED"; tail -40 "$log"; return 1; }
    sleep 5; echo -n "."
  done
  echo " timeout"; return 1
}

# send_requests <mode> <prime|measure> <num_prompts>
send_requests() {
  vllm bench serve --model "$MODEL" --host 0.0.0.0 --port 9999 \
    --dataset-name custom --dataset-path "$DATASET" --skip-chat-template \
    --custom-output-len $OUTPUT_LEN --ignore-eos \
    --num-prompts "$3" --request-rate inf --max-concurrency $((CONC * DP)) \
    --save-result --result-dir "$OUT_DIR" --result-filename "$1_$2.json" \
    > "$OUT_DIR/client_$1_$2.log" 2>&1
}

PROBLEMS=0
warn() { echo "WARN: $*" | tee -a "$OUT_DIR/problems.txt"; PROBLEMS=$((PROBLEMS + 1)); }

prime() {
  send_requests "$1" prime $PRIME_PROMPTS
  # "Rearranged experts   in" is a real one; the startup one reads "Rearranged experts  (profile)  in".
  if [[ "$1" == on ]] && ! grep -qE "Rearranged experts +in |Skip rearrange" "$OUT_DIR/server_on.log"; then
    warn "on: no rearrangement during priming"
  fi
}

# Also keeps the part of the server log written while measuring.
measure() {
  local log="$OUT_DIR/server_$1.log" size
  size=$(stat -c %s "$log")
  send_requests "$1" measure $MEASURE_PROMPTS
  tail -c +$((size + 1)) "$log" > "$OUT_DIR/server_$1_measure.log"
}

# health <mode>: problems the summary table would hide. Call while the server is up.
health() {
  local log="$OUT_DIR/server_$1.log" kv preempt slow_all slow errors
  kv=$(grep -o 'KV cache size: [0-9,]* tokens' "$log" | tr -d , | awk '{print $4}' | sort -n | head -1 || true)
  preempt=$(curl -s -m 5 http://0.0.0.0:9999/metrics \
            | awk '/^vllm:num_preemptions_total/ {s += $NF} END {printf "%d", s}' || true)
  # "No available shared memory broadcast block": one step took > 60 s. All DP engines
  # log it at once, so count distinct timestamps. First steps (kernel JIT) are expected.
  slow_all=$(grep 'No available shared memory broadcast' "$log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | sort -u | wc -l)
  slow=$(grep 'No available shared memory broadcast' "$OUT_DIR/server_$1_measure.log" 2>/dev/null \
         | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | sort -u | wc -l || true)
  errors=$(grep -cE 'Traceback|^\S* *ERROR | ERROR [0-9-]+ ' "$log" || true)
  echo "   $1 health: KV ${kv:-?} tokens/rank (min), preempted ${preempt:-?}," \
       "steps > 60 s: $slow_all ($slow while measuring), error lines $errors" | tee -a "$OUT_DIR/health.txt"
  (( ${preempt:-0} == 0 )) || warn "$1: $preempt requests preempted: KV cache too small for CONC $CONC"
  (( ${slow:-0} == 0 ))    || warn "$1: a step took > 60 s while measuring"
  (( errors == 0 ))        || warn "$1: $errors error lines in $log"
}

analyze() {
  python3 - "$OUT_DIR" "$PHASE" $MODES <<'EOF' | tee "$OUT_DIR/summary.txt"
import json, os, re, sys
out, phase, modes = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = {}
for m in modes:
    path = os.path.join(out, f"{m}_measure.json")
    if not os.path.exists(path):
        continue
    r = json.load(open(path))
    log = open(os.path.join(out, f"server_{m}_measure.log")).read()
    b = [float(x) for x in re.findall(r"balancedness=([\d.]+)", log)]
    rows[m] = dict(
        b=f"{sum(b) / len(b):.3f}" if b else "-",
        stall=sum(float(x) for x in re.findall(r"Rearranged experts\s+in ([\d.]+) s", log)),
        tps=r["output_throughput"] if phase == "decode" else r["total_input_tokens"] / r["duration"],
        ttft=r["mean_ttft_ms"], itl=r["mean_itl_ms"])
base = rows.get("fake_off") or rows.get("off")
unit = "out tok/s" if phase == "decode" else "in tok/s"
print(f"{'mode':9} {'b':>5} {unit:>10} {'vs base':>7} {'TTFT ms':>9} {'ITL ms':>8} {'stall s':>7}")
for m, r in rows.items():
    rel = f"{r['tps'] / base['tps']:.3f}" if base else "-"
    print(f"{m:9} {r['b']:>5} {r['tps']:>10.0f} {rel:>7} {r['ttft']:>9.1f} {r['itl']:>8.2f} {r['stall']:>7.1f}")
print("b: mean balancedness while measuring. base: fake_off (else off). "
      "stall: rearrangement time while measuring, should be 0. ITL is the mean, not the median.")
for m in modes:
    if m not in rows:
        print(f"WARN: {m}: no measurement")
    elif rows[m]["stall"] > 0:
        print(f"WARN: {m}: a rearrangement fell inside the measured window")
EOF
}

# timed <label> <command...>: run it, then print and log how long it took.
timed() {
  local label=$1 t0=$SECONDS rc=0; shift
  "$@" || rc=$?
  local s=$((SECONDS - t0))
  printf '   %-16s %3dm%02ds\n' "$label" $((s / 60)) $((s % 60)) | tee -a "$OUT_DIR/timing.txt"
  return $rc
}

for mode in $MODES; do
  if ! timed "$mode start" start_server "$mode"; then
    warn "$mode: server failed to start, see $OUT_DIR/server_$mode.log"
  else
    if ! timed "$mode prime" prime "$mode"; then
      warn "$mode: priming failed, see $OUT_DIR/client_${mode}_prime.log"
    elif ! timed "$mode measure" measure "$mode"; then
      warn "$mode: measuring failed, see $OUT_DIR/client_${mode}_measure.log"
    fi
    health "$mode"
  fi
  stop_server
done
analyze
PROBLEMS=$((PROBLEMS + $(grep -c '^WARN' "$OUT_DIR/summary.txt" || true)))
printf '== total %dm%02ds\n' $((SECONDS / 60)) $((SECONDS % 60)) | tee -a "$OUT_DIR/timing.txt"
if (( PROBLEMS )); then
  echo "== $([[ "$SMOKE" == 1 ]] && echo "SMOKE FAILED: ")$PROBLEMS problem(s), see the WARN lines"
else
  echo "== $([[ "$SMOKE" == 1 ]] && echo "SMOKE PASSED: ")no problems found"
fi
