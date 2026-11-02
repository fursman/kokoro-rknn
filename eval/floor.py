#!/usr/bin/env python3
"""The reference's own variation, i.e. what "identical" can mean here: kokoro-onnx against
itself with onnxruntime's graph optimisations off and with 1 thread instead of 4, and the
CPU pieces (front -> har -> dec, whole sentence) against kokoro-onnx. -> eval/out/floor.json"""
import json

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

from common import MODEL, OUT, SENTENCES, VOICES, Cpu, load_ref, session, style, tokens
from metrics import all_metrics

variants = {"ort_disable_all": Kokoro.from_session(session(MODEL, 4, ort.GraphOptimizationLevel.ORT_DISABLE_ALL), str(VOICES)),
            "threads1": Kokoro.from_session(session(MODEL, 1), str(VOICES))}
cpu = Cpu()
res = {}
for name, text in SENTENCES.items():
    ref, enc = load_ref(name)
    toks = tokens(text)
    res[name] = {vn: all_metrics(k._infer(toks, style(len(toks)), 1.0)[0].astype(np.float32), ref)
                 for vn, k in variants.items()}
    res[name]["cpu_pieces"] = all_metrics(cpu.full(*enc), ref)
    print(name, {vn: {k: round(v, 2) for k, v in m.items() if v is not None} for vn, m in res[name].items()}, flush=True)
json.dump(res, open(OUT / "floor.json", "w"), indent=1)
