#!/usr/bin/env python3
"""Blind listening samples: loudness-matched to -23 LUFS (pyloudnorm, BS.1770-4, a pure linear
gain), 16-bit mono 24 kHz, each group padded with digital silence to one length so neither
duration nor file size gives a version away, and shuffled with secrets.SystemRandom.

    python eval/make_samples.py OUTDIR group1.json [group2.json ...]

A group is {"title": ..., "items": {"label": "path.npy", ...}}; files are numbered x1, x2, ...
across groups. The key goes to OUTDIR/key.txt (don't look before listening) and the gains
to OUTDIR/loudness.json."""
import json, secrets, sys
from pathlib import Path
import numpy as np, pyloudnorm as pyln, soundfile as sf
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
meter = pyln.Meter(24000)
rng = secrets.SystemRandom()
key = ["Blinded listening key. All files 24 kHz mono 16-bit, loudness-matched",
       "to -23 LUFS integrated (pure linear gain), padded with digital silence to a common length per group.", ""]
info = {}
n = 0
for gpath in sys.argv[2:]:
    g = json.load(open(gpath))
    labels = list(g["items"]); rng.shuffle(labels)
    clips = {}
    for lab in labels:
        x = np.load(g["items"][lab]).astype(np.float64)
        L = meter.integrated_loudness(x)
        y = x * 10 ** ((-23.0 - L) / 20)
        clips[lab] = (y, L)
    length = max(y.size for y, _ in clips.values())
    key.append(g["title"])
    for lab in labels:
        n += 1
        y, L = clips[lab]
        y = np.pad(y, (0, length - y.size))
        peak = 20 * np.log10(np.abs(y).max() + 1e-12)
        assert peak < 0, f"{lab} would clip ({peak:.1f} dBFS)"
        sf.write(out / f"x{n}.wav", y.astype(np.float32), 24000, subtype="PCM_16")
        key.append(f"x{n}.wav = {lab}")
        info[f"x{n}"] = dict(label=lab, lufs_before=round(L, 2), gain_db=round(-23.0 - L, 2), peak_after_dbfs=round(peak, 2))
    key.append("")
(out / "key.txt").write_text("\n".join(key) + "\n")
json.dump(info, open(out / "loudness.json", "w"), indent=1)
print("\n".join(key))
