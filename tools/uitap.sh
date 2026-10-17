#!/usr/bin/env bash
# Tap a view on the phone by resource id (e.g. renameVoiceButton) or by visible text ("text:Save").
# usage: uitap.sh SERIAL ID|text:LABEL
set -eu
adb=("$HOME/Android/Sdk/platform-tools/adb" -s "$1")
"${adb[@]}" shell uiautomator dump /sdcard/mira_ui.xml > /dev/null
xml=$("${adb[@]}" shell cat /sdcard/mira_ui.xml)
"${adb[@]}" shell rm /sdcard/mira_ui.xml
if [[ $2 == text:* ]]; then key="text=\"${2#text:}\""; else key="resource-id=\"dev.local.miravoice:id/$2\""; fi
node=$(grep -o "<node[^>]*$key[^>]*>" <<< "$xml" | head -1)
[ -n "$node" ] || { echo "uitap: $2 not on screen" >&2; exit 1; }
read -r x1 y1 x2 y2 < <(sed -E 's/.*bounds="\[([0-9]+),([0-9]+)\]\[([0-9]+),([0-9]+)\]".*/\1 \2 \3 \4/' <<< "$node")
"${adb[@]}" shell input tap $(( (x1 + x2) / 2 )) $(( (y1 + y2) / 2 ))
echo "tapped $2 at $(( (x1 + x2) / 2 )),$(( (y1 + y2) / 2 ))"
