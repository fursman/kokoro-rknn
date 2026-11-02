#!/usr/bin/env python3
"""Why windowed decoding cannot match whole-sentence decoding: the decoder's AdaIN statistics.

The decoder (models/dec.onnx) has 108 time-axis means (58 AdaIN/InstanceNorm layers, mean and
variance; 20 of them at the frame rates, 88 in the generator). Every one is computed over
whatever window the decoder is given. Here they are taken from the whole-sentence pass and
injected into windowed decodes:

  inject  none (window-local) | all 108 | only the 20 frame-rate ones | only the 88 generator ones
  ctx     frames of real context each side (windows: first 20 frames, then 40; hard cuts)

and, separately, the whole sentence is decoded with the exact statistics perturbed by
(1 + eps * N(0, 1)).

    python eval/adain_stats.py [short,medium] [2,4,6,8,12,16]
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

from common import MODELS, OUT, SENTENCES, Cpu, session
from metrics import lsd, mag_snr, wave_snr

SPF, HPF = 600, 120


def graphs():
    """dec_stats: dec.onnx plus every time-axis ReduceMean as an output; dec_inject_<sel>:
    those ReduceMeans replaced by inputs."""
    out = OUT / "adain"
    out.mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(MODELS / "dec.onnx"))
    rms = [n for n in m.graph.node if n.op_type == "ReduceMean"]
    for n in rms:
        m.graph.output.append(helper.make_tensor_value_info(n.output[0], TensorProto.FLOAT, [1, None, 1]))
    onnx.save(m, str(out / "dec_stats.onnx"))
    for sel in ("all", "front", "gen"):
        m = onnx.load(str(MODELS / "dec.onnx"))
        for n in [n for n in m.graph.node if n.op_type == "ReduceMean"]:
            if sel == "all" or (sel == "gen") == ("/generator/" in n.name):
                m.graph.node.remove(n)
                m.graph.input.append(helper.make_tensor_value_info(n.output[0], TensorProto.FLOAT, [1, None, 1]))
        onnx.save(m, str(out / f"dec_inject_{sel}.onnx"))
    print(f"{len(rms)} time-axis means; {sum('/generator/' in n.name for n in rms)} in the generator")
    return out


def plan(T, first=20, step=40):
    p, e = [], 0
    while e < T:
        e1 = min(T, e + (first if not p else step))
        p.append((e, e1))
        e = e1
    return p


def main():
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["short", "medium"]
    ctxs = [int(c) for c in (sys.argv[2] if len(sys.argv) > 2 else "2,4,6,8,12,16").split(",")]
    d = graphs()
    cpu = Cpu()
    stats = session(d / "dec_stats.onnx")
    snames = [o.name for o in stats.get_outputs()][1:]
    inj = {w: session(d / f"dec_inject_{w}.onnx") for w in ("all", "front", "gen")}
    for name in names:
        asr, F0, N, s, har, _ = cpu.encode(SENTENCES[name])
        T = asr.shape[2]
        o = stats.run(None, {"asr": asr, "F0": F0, "N": N, "s": s, "har": har})
        full, st = o[0], dict(zip(snames, o[1:]))
        show = lambda tag, y: print(f"[{name}, T={T}] {tag}: wave SNR {wave_snr(y, full):6.1f} dB, "
                                    f"|S| SNR {mag_snr(y, full):5.1f} dB, LSD {lsd(y, full):5.2f} dB", flush=True)
        for w in ("none", "all", "front", "gen"):
            for c in ctxs:
                parts = []
                for e0, e1 in plan(T):
                    lo, hi = max(0, e0 - c), min(T, e1 + c)
                    f = {"asr": np.ascontiguousarray(asr[:, :, lo:hi]), "F0": np.ascontiguousarray(F0[:, 2 * lo:2 * hi]),
                         "N": np.ascontiguousarray(N[:, 2 * lo:2 * hi]), "s": s,
                         "har": np.ascontiguousarray(har[:, :, HPF * lo:HPF * hi + 1])}
                    if w == "none":
                        a = cpu.dec.run(None, f)[0]
                    else:
                        sess = inj[w]
                        f.update({i.name: st[i.name] for i in sess.get_inputs() if i.name in st})
                        a = sess.run(None, f)[0]
                    parts.append(a[(e0 - lo) * SPF:(e1 - lo) * SPF])
                show(f"inject {w:5s} ctx {c:2d}", np.concatenate(parts))
        rng = np.random.default_rng(0)
        for eps in (1e-4, 1e-3, 1e-2, 3e-2):
            p = {k: (v * (1 + eps * rng.standard_normal(v.shape))).astype(np.float32) for k, v in st.items()}
            y = inj["all"].run(None, {"asr": asr, "F0": F0, "N": N, "s": s, "har": har, **p})[0]
            show(f"whole sentence, statistics x (1 + {eps:g} N(0,1))", y)


if __name__ == "__main__":
    main()
