#!/usr/bin/env python3
"""Attribution, CPU part (fp32): how much each property of a windowed NPU pipeline costs,
emulated with the reference decoder itself, and the whole-sentence bucket graphs in fp32.

    python eval/eval_cpu.py [PREFIX ...]      e.g. /tmp/chk (uses /tmp/chk_T<b>_A.onnx for every b)

  win50_ctx2_harwin    50-frame windows (first 20), 2 frames of context, zero-padded edges,
                       harmonic source per window: the earlier NPU pipeline's scheme
  win50_ctx2_harfull   same, harmonic source sliced from the whole sentence
  win50_ctx12_harfull  same with 12 frames of context (74-frame windows, first 44)
  whole_<prefix>       whole sentence through the bucket graphs (fp32, onnxruntime)

If eval/out/orig_<g2p>_<name>.npy exist (audio from another pipeline, any length), they are
compared after trimming along a DTW path (orig_misaki / orig_espeak in RESULTS.md: the
earlier windowed NPU server). -> eval/out/cpu.json
"""
import json
import sys

import numpy as np

from common import OUT, SENTENCES, Cpu, load_ref
from kokoro_rknn.buckets import onnx_split, pad_inputs, pick
from metrics import all_metrics, dtw_metrics, lufs
from windowed import windowed

cpu = Cpu()
res = {}
for name in SENTENCES:
    ref, enc = load_ref(name)
    T = enc[0].shape[2]
    r = res[name] = {"ref_lufs": lufs(ref)}
    for g in ("misaki", "espeak"):
        p = OUT / f"orig_{g}_{name}.npy"
        if p.exists():
            r[f"orig_{g}"] = dtw_metrics(np.load(p), ref)
    for key, kw in {"win50_ctx2_harwin": dict(ctx=2, har_per_window=True),
                    "win50_ctx2_harfull": dict(ctx=2, har_per_window=False),
                    "win50_ctx12_harfull": dict(ctx=12, t_first=44, t_fix=74, har_per_window=False)}.items():
        y, seams = windowed(cpu, *enc, **kw)
        r[key] = all_metrics(y, ref, seams)
        r[key].update({k: v for k, v in dtw_metrics(y, ref).items() if k != "dlufs"})
    for prefix in sys.argv[1:]:
        Tb = pick(T)
        if Tb is None:
            continue
        dec = onnx_split(f"{prefix}_T{Tb}")
        y = cpu.tail(dec(pad_inputs(*enc, Tb)), T)
        r[f"whole_{prefix.split('/')[-1]}"] = all_metrics(y, ref)
    print(name, json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items() if vv is not None} if isinstance(v, dict) else round(v, 2)
                            for k, v in r.items()}), flush=True)
json.dump(res, open(OUT / "cpu.json", "w"), indent=1)
