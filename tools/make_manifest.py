"""Write the app's model manifest (android/app/src/main/assets/models.json) for a models release.

usage: make_manifest.py MODELS_DIR RELEASE_TAG   (files are expected at
       https://github.com/<repo>/releases/download/<tag>/<name>)
"""
import hashlib
import json
import os
import sys

REPO = "ssjrocks/mira-voice"
src, tag = sys.argv[1], sys.argv[2]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = []
for name in sorted(os.listdir(src)):
    p = os.path.join(src, name)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    files.append({"name": name, "url": f"https://github.com/{REPO}/releases/download/{tag}/{name}",
                  "size": os.path.getsize(p), "sha256": h.hexdigest()})
out = os.path.join(root, "android/app/src/main/assets/models.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"release": tag, "files": files}, open(out, "w"), indent=1)
print(f"{out}: {len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB")
