"""fp32 emulation of windowed decoding with the reference decoder (models/dec.onnx), to
attribute the error of a windowed NPU pipeline: windows of t_fix frames (the first t_first),
ctx frames of context each side, zero-padded edges, a 480-sample crossfade, and the
harmonic source either recomputed per window (its phase restarts) or sliced from the
whole-sentence one (phase continuous). cpu: an object with .har and .dec sessions."""
import numpy as np
SPF, HPF = 600, 120


def windowed(cpu, asr, F0, N, s, har_full, ctx=2, t_first=20, t_fix=50, ovl=480, har_per_window=True):
    T = asr.shape[2]
    half = ovl // 2
    out = np.empty(0, np.float32)
    seams = []
    e0 = 0
    while e0 < T:
        tf = t_first if e0 == 0 else t_fix
        e1 = min(e0 + tf - 2 * ctx, T)
        w0, w1 = e0 - ctx, e1 + ctx
        lo, hi = max(0, w0), min(T, w1)
        lp, rp = lo - w0, w1 - hi
        rp += tf - (hi - lo + lp + rp)
        a = np.pad(asr[:, :, lo:hi], ((0, 0), (0, 0), (lp, rp)))
        f = np.pad(F0[:, 2 * lo:2 * hi], ((0, 0), (2 * lp, 2 * rp)))
        n = np.pad(N[:, 2 * lo:2 * hi], ((0, 0), (2 * lp, 2 * rp)))
        if har_per_window:
            har = cpu.har.run(None, {"F0": f})[0]
        else:
            h = har_full[:, :, HPF * lo:HPF * hi + 1]
            har = np.pad(h, ((0, 0), (0, 0), (HPF * lp, HPF * rp)))
        audio = cpu.dec.run(None, {"asr": a, "F0": f, "N": n, "s": s, "har": har})[0]
        left = half if e0 > 0 else 0
        right = 0 if e1 >= T else half
        off = ctx * SPF
        piece = audio[off - left: off + (e1 - e0) * SPF + right]
        if out.size >= ovl and piece.size >= ovl and e0 > 0:
            w = np.linspace(0, 1, ovl, endpoint=False, dtype=np.float32)
            seams.append(out.size - half)
            out = np.concatenate([out[:-ovl], out[-ovl:] * (1 - w) + piece[:ovl] * w, piece[ovl:]])
        else:
            out = np.concatenate([out, piece])
        e0 = e1
    return out, seams
