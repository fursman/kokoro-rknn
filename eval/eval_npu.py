#!/usr/bin/env python3
"""The NPU decoder (fp16, whole sentence, buckets) against the reference, with timings.
Runs on the board, over rknnlite (the simple path; the server's C-API lanes give
bit-identical audio). -> eval/out/npu.json and eval/out/npu_<name>.npy

    python eval/eval_npu.py [--runs 5] [--cores 0_1_2]
"""
import argparse
import json
import time

import numpy as np

from common import MODELS, OUT, SENTENCES, Cpu, load_ref
from kokoro_rknn.buckets import BUCKETS, pad_inputs, pick, rknnlite_split
from metrics import all_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--buckets", default=",".join(map(str, BUCKETS)))
ap.add_argument("--cores", default="0_1_2")
a = ap.parse_args()
buckets = [int(b) for b in a.buckets.split(",")]
cpu = Cpu()
decs = {}
res = {}
for name, text in SENTENCES.items():
    ref, enc = load_ref(name)
    T = enc[0].shape[2]
    Tb = pick(T, buckets)
    res[name] = {"T": T, "Tb": Tb}
    if Tb is None:
        print(name, f"T={T}: longer than the largest bucket loaded (the server would decode it on the CPU)")
        continue
    if Tb not in decs:
        decs[Tb] = rknnlite_split(MODELS / "buckets" / f"dec_T{Tb}", Tb, a.cores)
    dec = decs[Tb]
    f = pad_inputs(*enc, Tb)
    dec(f)                                          # warm
    td, tt = [], []
    for _ in range(a.runs):
        t0 = time.perf_counter(); xp = dec(f); t1 = time.perf_counter(); y = cpu.tail(xp, T); t2 = time.perf_counter()
        td.append(t1 - t0); tt.append(t2 - t1)
    m = all_metrics(y, ref)
    tf = []
    for _ in range(a.runs):
        t0 = time.perf_counter(); cpu.encode(text); tf.append(time.perf_counter() - t0)
    m.update(dec_ms=1000 * float(np.median(td)), tail_ms=1000 * float(np.median(tt)),
             front_har_ms=1000 * float(np.median(tf)), audio_s=T * 0.025)
    res[name].update(m)
    np.save(OUT / f"npu_{name}.npy", y)
    print(name, json.dumps({k: round(v, 2) for k, v in m.items() if v is not None}), flush=True)
json.dump(res, open(OUT / "npu.json", "w"), indent=1)
