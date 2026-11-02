#!/usr/bin/env python3
"""ONNX -> RKNN for the RK3588, fp16 (no quantisation). Needs rknn-toolkit2 (2.3.2 used here).

    python build/compile_rknn.py models/buckets/dec_T128_A.onnx models/buckets/dec_T128_A.rknn

Check the log: a model whose build printed "REGTASK: bit width of field value exceeds the
limit" even once hangs the NPU when it runs (job timeouts and soft resets for minutes).
build_buckets.sh counts those lines for you.
"""
import sys
import time

from rknn.api import RKNN

src, dst = sys.argv[1], sys.argv[2]
t0 = time.time()
r = RKNN(verbose=False)
r.config(target_platform="rk3588", optimization_level=3)
assert r.load_onnx(src) == 0, "load_onnx"
assert r.build(do_quantization=False) == 0, "build"
assert r.export_rknn(dst) == 0, "export"
print(f"COMPILED {dst} in {time.time() - t0:.0f} s", flush=True)
