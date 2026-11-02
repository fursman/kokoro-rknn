#!/usr/bin/env python3
"""Download the reference model and voices from kokoro-onnx's releases, checksummed.

    python build/fetch_models.py models

kokoro-v1.0.onnx is the file from kokoro-onnx's `model-files-v1.0` release (January 2025).
Everything here is cut from that graph: build/extract.py looks its tensors up by name,
and the newer re-export in `model-files-v1.1` has a different graph (see the README).
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download"
FILES = {
    "kokoro-v1.0.onnx": ("model-files-v1.0", "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"),
    "voices-v1.0.bin": ("model-files-v1.0", "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "models")
    out.mkdir(parents=True, exist_ok=True)
    for name, (tag, want) in FILES.items():
        path = out / name
        if not (path.exists() and sha256(path) == want):
            print(f"downloading {name} ({tag})", flush=True)
            urllib.request.urlretrieve(f"{BASE}/{tag}/{name}", path)
        got = sha256(path)
        if got != want:
            raise SystemExit(f"{name}: sha256 {got}, expected {want}")
        print(f"{name}: ok")


if __name__ == "__main__":
    main()
