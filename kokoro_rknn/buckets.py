"""Bucketed, masked whole-sentence decoding: the shapes, the masks, and a simple runner.

A sentence of T frames (25 ms each) is decoded by the graphs of the smallest bucket
Tb >= T. Its inputs are zero-padded to Tb and masks come in at every time
resolution of the decoder (T, 2T, 20T, 120T+1):

    m_<k>  1 on the real frames, 0 on the padding; multiplies the input of every
           convolution with a kernel wider than 1, so the real frames see zeros past
           the end, exactly as in the unpadded model;
    w_<k>  m_<k> * (padded length / real length); every time-axis mean of the AdaIN
           layers is GlobalAveragePool(x * w), i.e. the mean over the real frames only.

Each bucket is two graphs (see build/surgery.py): A runs at the frame rates and ends
in a polyphase form of the last upsampling layer; the CPU interleaves its six phases
and adds the reflection pad; B runs the 120T+1 stage on a (1, C, H, W) view.

`Split` runs A and B with any backend: onnxruntime (to check the surgery in fp32) or
rknnlite (simple, one call at a time). The server uses `lanes.py` instead, which does
the same over the RKNN C API without copies.
"""
from pathlib import Path

import numpy as np

BUCKETS = (64, 96, 128, 160, 200, 256, 320, 400)
# The graphs' inputs, in the order build/surgery.py writes them (rknnlite takes a list).
NAMES_A = ["asr", "F0", "N", "s", "har", "m_T", "w_T", "m_2T", "w_2T", "m_20T", "w_20T"]
NAMES_B = ["x_after", "s", "har", "m_120T1", "w_120T1"]
FRAME = 600                  # samples per frame at 24 kHz
MAXW = 8191                  # widest time axis an elementwise op may broadcast over on the NPU


def lengths(T):
    return {"T": T, "2T": 2 * T, "20T": 20 * T, "120T1": 120 * T + 1}


def b_shape(T):
    """B runs on L' = H * W >= 120T + 1 samples (zeros past the end, masked), with
    W <= 8160 a multiple of 32: elementwise ops on such views ran ~35% faster than
    on odd widths."""
    L = 120 * T + 1
    H = -(-L // 8160)
    W = -(-L // (H * 32)) * 32
    return H, W, H * W


def pick(T, buckets=BUCKETS):
    """The smallest bucket that holds T frames, or None."""
    return next((b for b in sorted(buckets) if b >= T), None)


def pad_inputs(asr, F0, N, s, har, Tb):
    """Zero-pad one sentence's decoder inputs to bucket Tb and make its masks."""
    T = asr.shape[2]
    assert T <= Tb, (T, Tb)
    feeds = {
        "asr": np.pad(asr, ((0, 0), (0, 0), (0, Tb - T))).astype(np.float32),
        "F0": np.pad(F0, ((0, 0), (0, 2 * (Tb - T)))).astype(np.float32),
        "N": np.pad(N, ((0, 0), (0, 2 * (Tb - T)))).astype(np.float32),
        "s": np.ascontiguousarray(s, np.float32),
        "har": np.pad(har, ((0, 0), (0, 0), (0, 120 * (Tb - T)))).astype(np.float32),
    }
    lv, lb = lengths(T), lengths(Tb)
    for k in lb:
        m = np.zeros((1, 1, lb[k]), np.float32)
        m[..., :lv[k]] = 1.0
        feeds[f"m_{k}"] = m
        feeds[f"w_{k}"] = m * np.float32(lb[k] / lv[k])
    return feeds


def interleave(poly):
    """Graph A's polyphase output (1, 6*128, 20T) -> the 128-channel signal after the
    stride-6 upsampling and ReflectionPad1d((1, 0)): (1, 128, 120T + 1)."""
    y = poly.reshape(6, 128, -1).transpose(1, 2, 0).reshape(1, 128, -1)    # y[6q + p] = phase p at q
    return np.concatenate([y[:, :, 1:2], y], axis=2)


class Split:
    """The two-graph decoder: A -> CPU interleave -> B. run_a/run_b take a feed dict."""

    def __init__(self, run_a, run_b, names_a, names_b, Tb):
        self.run_a, self.run_b, self.names_a, self.names_b, self.Tb = run_a, run_b, names_a, names_b, Tb
        self.H, self.W, self.Lb = b_shape(Tb)

    def __call__(self, feeds):
        """feeds from pad_inputs -> conv_post output (1, 22, Lb); the first 120T+1 columns are real."""
        poly = self.run_a({k: feeds[k] for k in self.names_a})
        xa = interleave(poly)
        pad = self.Lb - xa.shape[2]
        P = (lambda a: np.pad(a, ((0, 0), (0, 0), (0, pad)))) if pad else (lambda a: a)
        view = (lambda a: a.reshape(1, a.shape[1], self.H, self.W)) if self.H > 1 else (lambda a: a)
        fb = {"x_after": view(P(xa)), "s": feeds["s"], "har": P(feeds["har"])}
        for k in ("m_120T1", "w_120T1"):
            v = P(feeds[k])
            if k[0] == "w":                 # B averages over Lb, not 120Tb + 1
                v = v * np.float32(self.Lb / (120 * self.Tb + 1))
            fb[k] = view(v)
        return self.run_b({k: fb[k] for k in self.names_b}).reshape(1, 22, -1)


def onnx_split(prefix, threads=4):
    """A bucket's A/B graphs on onnxruntime (fp32, CPU): prefix 'buckets/dec_T64'."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sa = ort.InferenceSession(f"{prefix}_A.onnx", so, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(f"{prefix}_B.onnx", so, providers=["CPUExecutionProvider"])
    na = [i.name for i in sa.get_inputs()]
    nb = [i.name for i in sb.get_inputs()]
    Tb = sa.get_inputs()[0].shape[2]
    return Split(lambda f: sa.run(None, f)[0], lambda f: sb.run(None, f)[0], na, nb, Tb)


class _Rknnlite:
    CORES = {"auto": "NPU_CORE_AUTO", "0": "NPU_CORE_0", "1": "NPU_CORE_1", "2": "NPU_CORE_2",
             "0_1": "NPU_CORE_0_1", "0_1_2": "NPU_CORE_0_1_2"}

    def __init__(self, path, cores):
        from rknnlite.api import RKNNLite
        self.r = RKNNLite(verbose=False)
        assert self.r.load_rknn(str(path)) == 0, path
        assert self.r.init_runtime(core_mask=getattr(RKNNLite, self.CORES[cores])) == 0, path

    def __call__(self, names, feeds):
        ins = [feeds[k] for k in names]
        return self.r.inference(inputs=ins, data_format=["nchw"] * len(ins))[0]


def rknnlite_split(prefix, Tb, cores="0_1_2"):
    """A bucket's A/B .rknn graphs on rknnlite. Simple, but slow on the CPU side: rknnlite
    converts every input and output (60-220 ms per sentence, against ~15 ms in lanes.py)."""
    prefix = Path(prefix)
    na, nb = NAMES_A, NAMES_B
    ma, mb = _Rknnlite(f"{prefix}_A.rknn", cores), _Rknnlite(f"{prefix}_B.rknn", cores)
    sp = Split(lambda f: ma(na, f).reshape(1, 768, 20 * Tb), lambda f: mb(nb, f), na, nb, Tb)
    sp.models = (ma, mb)
    return sp
