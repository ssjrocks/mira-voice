"""Write the app's model manifest (android/app/src/main/assets/models.json).

usage: make_manifest.py DIR:TAG [DIR:TAG ...]   files in DIR are expected at
       https://github.com/<repo>/releases/download/<TAG>/<name>  (e.g. build/release-models:models-1)
"""
import hashlib
import json
import os
import sys

REPO = "ssjrocks/mira-voice"
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = []
for arg in sys.argv[1:]:
    src, tag = arg.rsplit(":", 1)
    for name in sorted(os.listdir(src)):
        p = os.path.join(src, name)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        files.append({"name": name, "url": f"https://github.com/{REPO}/releases/download/{tag}/{name}",
                      "size": os.path.getsize(p), "sha256": h.hexdigest()})
out = os.path.join(root, "android/app/src/main/assets/models.json")
json.dump({"files": files}, open(out, "w"), indent=1)
print(f"{out}: {len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB")
