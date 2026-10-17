#!/usr/bin/env bash
# Trigger one synthesis on a device via intent and print the MIRA_RESULT line.
# usage: devrun.sh SERIAL GGUF THREADS HD(true|false) "TEXT" OUTNAME [extra am args...]
set -u
serial=$1 gguf=$2 threads=$3 hd=$4 text=$5 out=$6
shift 6
adb=("$HOME/Android/Sdk/platform-tools/adb" -s "$serial")
# the device shell re-parses the command line: single-quote the text, escaping embedded quotes
qtext="'${text//\'/\'\\\'\'}'"
"${adb[@]}" logcat -c
"${adb[@]}" shell am start -n dev.local.miravoice/.MainActivity --es gguf "$gguf" --ei threads "$threads" \
    --es text "$qtext" --ez hd "$hd" --ez play false --es out "$out" --ei seed 1 "$@" > /dev/null
for _ in $(seq 1 200); do
    if "${adb[@]}" logcat -d -s MiraVoice:* | grep -qE 'MIRA_RESULT|failed'; then break; fi
    sleep 3
done
"${adb[@]}" logcat -d -s MiraVoice:* | grep -E 'MIRA_RESULT|failed' | sed 's/.*MIRA_RESULT //' | cut -c1-500
