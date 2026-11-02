"""NPU decoder lanes over the RKNN C API.

A Lane = one NPU core with an A and a B context per bucket (the graphs from build/surgery.py).
Inputs are written straight into the contexts' DMA buffers in their native fp16 layouts
and A's output is interleaved (+ reflection pad) straight into B's input buffer, so the
only CPU work between the graphs is one fp16 shuffle. Weights are shared between lanes
(rknn_dup_context); each lane's contexts share one internal (scratch) buffer, since a
lane runs one graph at a time.
"""
import ctypes
import threading
import time
from pathlib import Path

import numpy as np

from .buckets import BUCKETS, b_shape
from .rknnc import FLAG_INTERNAL_ALLOC_OUTSIDE, Model, mem_view

try:                        # gcc -O3 -shared -fPIC -o kokoro_rknn/libinterleave.so kokoro_rknn/interleave.c
    _il = ctypes.CDLL(str(Path(__file__).resolve().parent / "libinterleave.so")).interleave
    _il.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
except OSError:
    _il = None
USE_C = _il is not None


class Bucket:
    """A and B for one bucket on one core (contexts only; buffers are bound by the Lane)."""
    def __init__(self, Tb, core, src=None, flags=0, buckets_dir=None):
        self.Tb = Tb
        self.H, self.W, self.Lb = b_shape(Tb)
        if src is None:
            self.A = Model(Path(buckets_dir) / f"dec_T{Tb}_A.rknn", core, flags=flags)
            self.B = Model(Path(buckets_dir) / f"dec_T{Tb}_B.rknn", core, flags=flags)
        else:                                   # same weights, another core
            self.A, self.B = src.A.dup(core), src.B.dup(core)
        self.ia = {n: i for i, n in enumerate(self.A.names)}
        self.ib = {n: i for i, n in enumerate(self.B.names)}


class Lane:
    """One NPU core. The lane runs one graph at a time, so all its contexts share one internal
    (scratch) buffer and, per input/output name, one I/O buffer sized for the largest bucket.
    Make lanes with make_lanes(): the other lanes share the first one's weights (rknn_dup_context),
    and every context must exist before any is bound (binding a context, then dup'ing it,
    corrupts its bindings in librknnrt 2.5.0)."""
    def __init__(self, core, buckets_dir, buckets=BUCKETS, src=None, share=True, bind=True):
        self.core = core
        self.lock = threading.Lock()
        self.share = share
        fl = FLAG_INTERNAL_ALLOC_OUTSIDE if share else 0
        self.buckets = {Tb: Bucket(Tb, core, src.buckets[Tb] if src is not None else None, fl, buckets_dir)
                        for Tb in sorted(buckets)}
        if bind:
            self.bind()

    def bind(self):
        ctxs = [m for bk in self.buckets.values() for m in (bk.A, bk.B)]
        self.internal_bytes = max(m.mem_size()["internal"] for m in ctxs)
        owner = ctxs[0]
        shared = {}
        self.io_bytes = 0
        if self.share:
            self.internal = owner.create_mem(self.internal_bytes)
            for m in ctxs:
                m.set_internal(self.internal)
                m.refresh()
            for g in ("A", "B"):
                sizes = {}
                for bk in self.buckets.values():
                    for key, n in getattr(bk, g).io_sizes().items():
                        sizes[key] = max(sizes.get(key, 0), n)
                shared[g] = {key: owner.create_mem(n) for key, n in sizes.items()}
            self.io_bytes = sum(m.contents.size for d in shared.values() for m in d.values())
        for bk in self.buckets.values():
            bk.A.bind(native=True, shared=shared.get("A"))
            bk.B.bind(native=True, shared=shared.get("B"))

    def dma_bytes(self):
        return {Tb: (bk.A.mem_size()["dma"], bk.B.mem_size()["dma"]) for Tb, bk in self.buckets.items()}

    def pick(self, T):
        return next((b for b in sorted(self.buckets) if b >= T), None)

    # ---- the steps
    def fill_a(self, bk, asr, F0, N, s, har, T):
        A, ia, Tb = bk.A, bk.ia, bk.Tb
        x = A.inp[ia["asr"]].reshape(512, Tb); x[:, :T] = asr.reshape(512, T); x[:, T:] = 0
        for k, v in (("F0", F0), ("N", N)):
            x = A.inp[ia[k]].reshape(-1); x[:2 * T] = v.reshape(-1); x[2 * T:] = 0
        A.inp[ia["s"]].reshape(-1)[:] = s.reshape(-1)[:128]
        h = A.inp[ia["har"]].reshape(22, -1); L = 120 * T + 1
        h[:, :L] = har.reshape(22, L); h[:, L:] = 0
        for k, r in (("T", 1), ("2T", 2), ("20T", 20)):
            n, nb = r * T, r * Tb
            m = A.inp[ia[f"m_{k}"]].reshape(-1); m[:n] = 1; m[n:] = 0
            w = A.inp[ia[f"w_{k}"]].reshape(-1); w[:n] = nb / n; w[n:] = 0

    def fill_b(self, bk, s, har, T, poly=None):
        """Interleave A's output (6 phases x 128 ch, 20Tb) + ReflectionPad1d((1,0)) into B's x_after."""
        A, B, ib, Tb, Lb = bk.A, bk.B, bk.ib, bk.Tb, bk.Lb
        poly = A.out[0].reshape(6, 128, 20 * Tb) if poly is None else poly
        n = 120 * Tb
        xa = B.inp[ib["x_after"]]
        if USE_C and poly.flags.c_contiguous:
            _il(poly.ctypes.data, xa.ctypes.data, 20 * Tb, Lb, int(bk.H > 1))
        elif bk.H == 1:                                   # plain (128, Lb)
            x = xa.reshape(128, Lb)
            x[:, 1:1 + n].reshape(128, 20 * Tb, 6)[...] = poly.transpose(1, 2, 0)
            x[:, 0] = poly[1, :, 0]
            x[:, 1 + n:] = 0
        else:                                           # NC1HWC2 (16, H*W, 8)
            x = xa.reshape(16, Lb, 8)
            x[:, 1:1 + n, :].reshape(16, 20 * Tb, 6, 8)[...] = poly.reshape(6, 16, 8, 20 * Tb).transpose(1, 3, 0, 2)
            x[:, 0, :] = poly[1, :, 0].reshape(16, 8)
            x[:, 1 + n:, :] = 0
        B.inp[ib["s"]].reshape(-1)[:] = s.reshape(-1)[:128]
        L = 120 * T + 1
        h = B.inp[ib["har"]].reshape(22, Lb); h[:, :L] = har.reshape(22, L); h[:, L:] = 0
        m = B.inp[ib["m_120T1"]].reshape(-1); m[:L] = 1; m[L:] = 0
        w = B.inp[ib["w_120T1"]].reshape(-1); w[:L] = Lb / L; w[L:] = 0

    def x_post(self, bk, T):
        """B's output, (22, 120T+1) float32."""
        a = bk.B.bound_out[0]
        o = mem_view(bk.B.out_mem[0], np.float16, (max(a.size, a.size_with_stride) // 2,))
        L = 120 * T + 1
        return o[:22 * bk.Lb].reshape(22, bk.Lb)[:, :L].astype(np.float32)   # plain rows (buffer padded to 24)

    def decode(self, asr, F0, N, s, har, T, timing=None):
        """One sentence (T frames, T <= the largest bucket) -> conv_post output (22, 120T+1)."""
        bk = self.buckets[self.pick(T)]
        with self.lock:
            t0 = time.perf_counter()
            self.fill_a(bk, asr, F0, N, s, har, T)
            t1 = time.perf_counter(); bk.A.run(); t2 = time.perf_counter()
            self.fill_b(bk, s, har, T)
            t3 = time.perf_counter(); bk.B.run(); t4 = time.perf_counter()
            y = self.x_post(bk, T)
            t5 = time.perf_counter()
        if timing is not None:
            timing.update(fill_a=t1 - t0, A=t2 - t1, fill_b=t3 - t2, B=t4 - t3, out=t5 - t4)
        return y


def make_lanes(buckets_dir, cores=(0, 1, 2), buckets=BUCKETS, share=True):
    """One lane per NPU core in `cores`. The first lane loads the models; the others are
    rknn_dup_context copies of it, so the weights are in NPU memory once."""
    lanes = [Lane(cores[0], buckets_dir, buckets, share=share, bind=False)]
    for c in cores[1:]:
        lanes.append(Lane(c, buckets_dir, buckets, src=lanes[0], share=share, bind=False))
    for ln in lanes:
        ln.bind()
    for ln in lanes:                    # the model files' bytes are only needed until every dup exists (-1.1 GB RSS)
        for bk in ln.buckets.values():
            for m in (bk.A, bk.B):
                m.__dict__.pop("_buf", None)
    return lanes
