#!/usr/bin/env python3
"""Cut kokoro-v1.0.onnx (the model kokoro-onnx runs) into the pieces this project runs.
Same weights, same ops: nothing is re-exported.

    front.onnx  tokens, style, speed -> asr (1,512,T), F0 (1,2T), N (1,2T), dur     CPU
    har.onnx    F0 (1,2T) -> har (1,22,120T+1): sine source + STFT (deterministic)   CPU
    dec.onnx    asr, F0, N, s (1,128), har -> audio (600 T samples)                  source for build/surgery.py
    tail.onnx   conv_post output (1,22,120T+1) -> audio: exp, sin, the model's iSTFT CPU

    python build/extract.py models        # reads models/kokoro-v1.0.onnx (build/fetch_models.py)
"""
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from onnx.utils import Extractor

ASR = "/encoder/MatMul_1_output_0"
F0 = "/encoder/Reshape_output_0"
N = "/encoder/Reshape_1_output_0"
DUR = "/encoder/Cast_output_0"
S = "/decoder/Slice_output_0"
HAR = "/decoder/decoder/generator/Concat_3_output_0"
X_POST = "/decoder/decoder/generator/conv_post/Conv_output_0"
RENAME = {ASR: "asr", F0: "F0", N: "N", DUR: "dur", S: "s", HAR: "har"}
SPEC = {"tokens": (TensorProto.INT64, [1, "L"]), "style": (TensorProto.FLOAT, [1, 256]),
        "speed": (TensorProto.FLOAT, [1]), "asr": (TensorProto.FLOAT, [1, 512, "T"]),
        "F0": (TensorProto.FLOAT, [1, "T2"]), "N": (TensorProto.FLOAT, [1, "T2"]),
        "dur": (TensorProto.INT64, ["B", "L"]), "s": (TensorProto.FLOAT, [1, 128]),
        "har": (TensorProto.FLOAT, [1, 22, "F"]), "audio": (TensorProto.FLOAT, ["S"]),
        X_POST: (TensorProto.FLOAT, [1, 22, "F"])}


def cut(ex, inputs, outputs, path):
    m = ex.extract_model(inputs, outputs)
    # Friendly I/O names (renamed everywhere the tensor appears).
    for node in m.graph.node:
        for i, x in enumerate(node.input):
            if x in RENAME:
                node.input[i] = RENAME[x]
        for i, x in enumerate(node.output):
            if x in RENAME:
                node.output[i] = RENAME[x]
    for v in list(m.graph.input) + list(m.graph.output):
        if v.name in RENAME:
            v.name = RENAME[v.name]
    # Declare the cut tensors' types and shapes explicitly, time axes dynamic.
    for coll in (m.graph.input, m.graph.output):
        vis = [helper.make_tensor_value_info(v.name, *SPEC[v.name]) for v in coll]
        del coll[:]
        coll.extend(vis)
    del m.graph.value_info[:]
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    ins = [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outs = [(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    print(f"{path.name}: {len(m.graph.node)} nodes, in={ins} out={outs}")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "models")
    model = onnx.shape_inference.infer_shapes(onnx.load(str(out / "kokoro-v1.0.onnx")))
    ex = Extractor(model)
    cut(ex, ["tokens", "style", "speed"], [ASR, F0, N, DUR], out / "front.onnx")
    cut(ex, [F0], [HAR], out / "har.onnx")
    cut(ex, [ASR, F0, N, S, HAR], ["audio"], out / "dec.onnx")
    cut(ex, [X_POST], ["audio"], out / "tail.onnx")


if __name__ == "__main__":
    main()
