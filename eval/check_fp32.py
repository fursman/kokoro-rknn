#!/usr/bin/env python3
"""Is the surgery exact? A bucket's A/B graphs on onnxruntime (fp32, CPU) against the reference.

    python build/surgery.py --models models --T 96 --sin rr --no-fuse --out /tmp/chk_T96.onnx
    python eval/check_fp32.py /tmp/chk_T96 short timer

Two comparisons per sentence: against the same decoder unpadded (dec.onnx on exactly T
frames: is the masking exact?) and against kokoro-onnx (the reference; its own floor is
~25-32 dB of waveform SNR, see floor.py, because onnxruntime fuses the full model differently).
"""
import sys

from common import Cpu, load_ref
from kokoro_rknn.buckets import onnx_split, pad_inputs
from metrics import all_metrics

dec, cpu = onnx_split(sys.argv[1]), Cpu()
for name in sys.argv[2:]:
    ref, enc = load_ref(name)
    T = enc[0].shape[2]
    if T > dec.Tb:
        print(name, f"T={T} does not fit bucket {dec.Tb}")
        continue
    y = cpu.tail(dec(pad_inputs(*enc, dec.Tb)), T)
    r = lambda m: {k: round(v, 2) for k, v in m.items() if v is not None}
    print(name, T, "->", dec.Tb, "vs unpadded:", r(all_metrics(y, cpu.full(*enc))), "vs kokoro-onnx:", r(all_metrics(y, ref)), flush=True)
