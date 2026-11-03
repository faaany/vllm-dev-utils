#!/bin/bash
# Pre-benchmark check for XPU cards that run slow under load (thermal throttling,
# a dead fan). DP/EP/TP jobs run every card in lockstep, so one slow card slows the
# whole group without any error; this catches it before the benchmark does.
#
# Loads every card at once for SECS seconds, as a DP8 job would, samples sysfs
# every second, then flags a card whose
#   - matmul TFLOPS is below MIN_TFLOPS_FRAC x the median card,
#   - mean frequency under load is below MIN_MHZ, or
#   - one fan reads 0 rpm while the other spins (at the end of the load).
# reason_thermal alone is not flagged: healthy B60s also set it after ~13 s at full
# load, while losing only 2-3% of their clock.
#
# Run inside a container with torch-xpu and xpu-smi, with all cards visible:
#   docker exec <container> bash /path/to/xpu_preflight.sh && ./your_benchmark.sh
# Exit code: 0 all cards OK, 1 a card flagged, 2 a card busy (nothing loaded).
# Logs stay in OUT (default: a new temp dir, printed at the end).
set -u
SECS=${SECS:-25}
MIN_MHZ=${MIN_MHZ:-2000}
MIN_TFLOPS_FRAC=${MIN_TFLOPS_FRAC:-0.9}
MAX_USED_MIB=${MAX_USED_MIB:-1024}   # a card using more is treated as busy
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$(mktemp -d -t xpu_preflight.XXXXXX)}
mkdir -p "$OUT"

ndev=$(xpu-smi discovery | grep -c "Device Name")
(( ndev > 0 )) || { echo "no XPU found by xpu-smi"; exit 2; }

# Never load a card someone else is using.
for ((i = 0; i < ndev; i++)); do
  used=$(xpu-smi stats -d "$i" | awk -F'current: ' '/GPU Memory Used/ {print int($2)}')
  if (( ${used:-0} > MAX_USED_MIB )); then
    echo "xpu $i is busy (${used} MiB used); nothing loaded"
    exit 2
  fi
done

sample() {   # t pci act_freq reason_thermal fan1 fan2
  while :; do
    for d in /sys/class/drm/card*/device; do
      f=$d/tile0/gt0/freq0
      [[ -d $f ]] || continue
      fans=$(cat "$d"/hwmon/hwmon*/fan1_input "$d"/hwmon/hwmon*/fan2_input 2>/dev/null | tr '\n' ' ')
      echo "$(date +%s) $(basename "$(readlink -f "$d")") $(cat "$f/act_freq") $(cat "$f/throttle/reason_thermal") $fans"
    done
    sleep 1
  done
}

temps() {    # core temperature near the end of the load
  sleep $((SECS - 5))
  for ((i = 0; i < ndev; i++)); do
    bdf=$(xpu-smi discovery -d "$i" 2>/dev/null | awk -F': ' '/PCI BDF Address/ {print $2}' | tr -d ' |')
    t=$(xpu-smi stats -d "$i" | awk -F'current: ' '/GPU Core Temperature/ {print int($2)}')
    echo "$bdf ${t:-NA}"
  done
}

echo "loading $ndev cards for ${SECS}s (logs: $OUT)"
sample > "$OUT/samples.txt" &
sampler=$!
temps > "$OUT/temps.txt" &
burns=()
for ((i = 0; i < ndev; i++)); do
  ZE_AFFINITY_MASK=$i python3 "$HERE/xpu_burn.py" "$SECS" > "$OUT/burn_$i.log" 2>&1 &
  burns+=($!)
done
wait "${burns[@]}"
kill "$sampler" 2>/dev/null
wait

python3 - "$OUT" "$MIN_MHZ" "$MIN_TFLOPS_FRAC" <<'EOF'
import collections, glob, os, statistics, sys

out, min_mhz, min_frac = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
tflops = {}
for f in glob.glob(os.path.join(out, "burn_*.log")):
    for line in open(f):
        if line.startswith("RESULT"):
            _, bus, tf = line.split()
            tflops[bus] = float(tf)
mhz, thermal, fans = collections.defaultdict(list), collections.Counter(), {}
for line in open(os.path.join(out, "samples.txt")):
    p = line.split()
    if len(p) < 4 or int(p[2]) == 0:   # act_freq 0: idle / not started yet
        continue
    mhz[p[1]].append(int(p[2]))
    thermal[p[1]] += int(p[3])
    fans[p[1]] = [int(x) for x in p[4:6]]
temps = dict(l.split() for l in open(os.path.join(out, "temps.txt")) if len(l.split()) == 2)

med = statistics.median(tflops.values()) if tflops else 0
bad = 0
print(f"{'pci':14} {'TFLOPS':>7} {'MHz':>6} {'thermal':>8} {'fans':>11} {'temp C':>7}  verdict")
for pci in sorted(mhz):
    bus = pci.split(":")[1]
    tf, f, fan = tflops.get(bus), statistics.mean(mhz[pci]), fans.get(pci, [])
    why = []
    if tf is None:
        why.append("no burn result")
    elif tf < min_frac * med:
        why.append(f"TFLOPS {tf / med:.0%} of median")
    if f < min_mhz:
        why.append(f"under {min_mhz:.0f} MHz")
    if len(fan) == 2 and min(fan) == 0 and max(fan) > 0:
        why.append("a fan reads 0 rpm")
    bad += bool(why)
    print(f"{pci:14} {tf if tf is not None else float('nan'):7.1f} {f:6.0f} "
          f"{thermal[pci] / len(mhz[pci]):8.0%} {','.join(map(str, fan)):>11} "
          f"{temps.get(pci, 'NA'):>7}  {'; '.join(why) or 'ok'}")
print(f"{bad} card(s) flagged" if bad else "all cards ok")
sys.exit(1 if bad else 0)
EOF
rc=$?
echo "logs: $OUT"
exit $rc
