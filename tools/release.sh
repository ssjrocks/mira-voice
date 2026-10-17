#!/usr/bin/env bash
# Build the release APK and publish it as GitHub release v<versionName> (the app's updater picks it up).
# usage: tools/release.sh NOTES_FILE
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
notes=$1
version=$(sed -n 's/^ *versionName = "\(.*\)"/\1/p' "$root/android/app/build.gradle.kts")
[ -n "$version" ] || { echo "no versionName found" >&2; exit 1; }
gh release view "v$version" --repo ssjrocks/mira-voice > /dev/null 2>&1 && { echo "v$version already released" >&2; exit 1; }

(cd "$root/android" && ./gradlew --no-daemon -q assembleRelease)
mkdir -p "$root/build/dist"
apk="$root/build/dist/MiraVoice-$version.apk"
cp "$root/android/app/build/outputs/apk/release/app-release.apk" "$apk"
echo "built $apk ($(du -h "$apk" | cut -f1))"

gh release create "v$version" "$apk" --repo ssjrocks/mira-voice --title "Mira Voice $version" \
    --notes-file "$notes" --latest
