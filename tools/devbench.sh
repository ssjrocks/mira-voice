#!/usr/bin/env bash
# npu branch: run NpuBench on the phone and print its result.
# usage: devbench.sh SERIAL MODEL INPUT SHAPE ACCEL(cpu|npu) [THREADS]
set -u
adb=("$HOME/Android/Sdk/platform-tools/adb" -s "$1")
"${adb[@]}" logcat -c
out=$("${adb[@]}" shell am start -n dev.local.miravoice/.MainActivity --es bench_model "$2" --es bench_input "$3" \
    --es bench_shape "$4" --es bench_accel "$5" --ei bench_threads "${6:-4}" --ei bench_iters 5 2>&1)
if grep -q "Exception occurred" <<< "$out"; then echo "am start failed: $(grep -m1 Exception -A1 <<< "$out" | tail -1)"; exit 1; fi
for _ in $(seq 1 150); do
    if "${adb[@]}" logcat -d -s MiraVoice:* AndroidRuntime:E | grep -qE 'MIRA_BENCH|FATAL'; then break; fi
    sleep 2
done
"${adb[@]}" logcat -d -s MiraVoice:* AndroidRuntime:E | grep -E 'MIRA_BENCH|FATAL' | sed 's/.*MIRA_BENCH //' | cut -c1-400 | head -3
