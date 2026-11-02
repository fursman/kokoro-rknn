"""Kokoro-82M with the decoder on the RK3588 NPU, one sentence per NPU core at a time.

`NpuKokoro` is kokoro-onnx's `Kokoro` with only `_infer` (tokens -> audio for one
batch) replaced, so the G2P, batching, silence trim and pauses are kokoro-onnx's own:

    front (CPU, onnxruntime)   tokens, style, speed -> asr, F0, N, durations
    harmonic source (CPU)      F0 -> har
    decoder (NPU)              one call for the whole sentence, padded to the next bucket
    tail (CPU)                 exp/sin and the model's own iSTFT

The four graphs are cut out of kokoro-v1.0.onnx (build/extract.py), so the weights and
ops are the reference's own; the decoder is rewritten for the NPU (build/surgery.py).

    from kokoro_rknn.engine import load
    k = load("models")                       # ~25 s: 3 lanes, 8 buckets each
    samples, rate = k.create("Hello there.", voice="af_heart", lang="en-us")
"""
import heapq
import itertools
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

from .buckets import BUCKETS

log = logging.getLogger("kokoro-rknn")


def session(path, threads, spin=False):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    # Spinning threads burned ~30% more CPU while the NPU worked, for no speed.
    so.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class LanePool:
    """Free lanes, granted to the waiting caller with the lowest (priority, arrival). A caller
    gets the first free lane in the order the lanes were given: with cores 1,2,0, core 0
    (which speech recognition on NPU_CORE_AUTO then has to itself) is only used when
    cores 1 and 2 are both busy."""

    def __init__(self, lanes):
        self.order = {id(ln): i for i, ln in enumerate(lanes)}
        self.free = list(lanes)
        self.cv = threading.Condition()
        self.waiting = []
        self.seq = itertools.count()
        self.busy = 0

    def acquire(self, prio=0):
        with self.cv:
            me = (prio, next(self.seq))
            heapq.heappush(self.waiting, me)
            while not (self.free and self.waiting[0] == me):
                self.cv.wait()
            heapq.heappop(self.waiting)
            self.busy += 1
            lane = self.free.pop(0)
            self.cv.notify_all()
            return lane

    def release(self, lane):
        with self.cv:
            self.free.append(lane)
            self.free.sort(key=lambda ln: self.order[id(ln)])
            self.busy -= 1
            self.cv.notify_all()


class CpuSet:
    """One worker's onnxruntime sessions. More workers than lanes let the next sentences'
    front run while the NPU decodes."""

    def __init__(self, models, threads):
        self.front = session(models / "front.onnx", threads)
        self.har = session(models / "har.onnx", threads)
        self.tail = session(models / "tail.onnx", threads)
        self.tail_in = self.tail.get_inputs()[0].name


class NpuKokoro(Kokoro):
    """kokoro_onnx.Kokoro with the decoder on per-core NPU lanes."""

    def setup_npu(self, models, cores=(1, 2, 0), buckets=BUCKETS, workers=0, threads=4):
        from .lanes import make_lanes
        models = Path(models)
        t0 = time.time()
        self.lanes = make_lanes(models / "buckets", cores, buckets)
        log.info(f"{len(self.lanes)} lanes on NPU cores {list(cores)} in {time.time() - t0:.1f}s "
                 f"(scratch {self.lanes[0].internal_bytes / 1e6:.0f} MB + io {self.lanes[0].io_bytes / 1e6:.0f} MB per lane)")
        self.pool = LanePool(self.lanes)
        self.cpu = queue.Queue()
        for _ in range(workers or len(cores)):
            self.cpu.put(CpuSet(models, threads))
        self.local = threading.local()
        self.stats = {"npu": 0, "cpu_fallback": 0}
        self.max_T = max(buckets)
        for ln in self.lanes:                   # the first call of each graph is slow: make it now
            for T in ln.buckets:
                ln.decode(np.zeros((1, 512, 8), np.float32), np.zeros((1, 16), np.float32),
                          np.zeros((1, 16), np.float32), np.zeros((1, 128), np.float32),
                          np.zeros((1, 22, 961), np.float32), 8)
        return self

    def _infer(self, tokens, style, speed):
        style = np.asarray(style, np.float32).reshape(1, 256)
        cs = self.cpu.get()
        prio = getattr(self.local, "prio", 0)
        try:
            tok = np.array([0, *tokens, 0], np.int64)
            asr, F0, N, dur = cs.front.run(None, {"tokens": tok[None], "style": style,
                                                  "speed": np.array([speed], np.float32)})
            T = asr.shape[2]
            if T > self.max_T:          # longer than the largest bucket: the reference model on the CPU
                self.stats["cpu_fallback"] += 1
                log.info(f"T={T} > largest bucket: CPU decode")
                return super()._infer(tokens, style, speed)
            har = cs.har.run(None, {"F0": F0})[0]
            lane = self.pool.acquire(prio)
            try:
                x = lane.decode(asr, F0, N, style[:, :128], har, T)
            finally:
                self.pool.release(lane)
            audio = cs.tail.run(None, {cs.tail_in: np.ascontiguousarray(x[None])})[0].reshape(-1)
        finally:
            self.cpu.put(cs)
        self.stats["npu"] += 1
        return audio.astype(np.float32), (dur.reshape(-1) if self.has_timings else None)

    def create_prio(self, prio, *a, **kw):
        """create(), with this thread's sentences queued for a lane at priority `prio`
        (lower first): a request's first piece goes before anyone's later pieces."""
        self.local.prio = prio
        return self.create(*a, **kw)


def load(models="models", cores=(1, 2, 0), buckets=BUCKETS, workers=0, threads=4):
    """models: the directory made by build/ (kokoro-v1.0.onnx, voices-v1.0.bin, front/har/tail.onnx,
    buckets/dec_T*_{A,B}.rknn). The full model stays loaded for sentences longer than the
    largest bucket, which are decoded on the CPU."""
    models = Path(models)
    k = NpuKokoro.from_session(session(models / "kokoro-v1.0.onnx", threads), str(models / "voices-v1.0.bin"))
    return k.setup_npu(models, cores, buckets, workers, threads)
