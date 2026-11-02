"""Shared by the evaluation scripts: the test sentences, the reference, and the CPU pieces.

The reference is kokoro-onnx (kokoro-v1.0.onnx) on the CPU: Kokoro._infer on kokoro-onnx's
own espeak-ng tokens, whole sentence, untrimmed. Every variant here gets the same tokens
and the same front outputs, so lengths are identical and waveform metrics are meaningful.

Paths: KOKORO_RKNN_MODELS (default models/) and KOKORO_RKNN_EVAL (default eval/out/).
Run the scripts from the repository root: python eval/make_refs.py, ...
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODELS = Path(os.environ.get("KOKORO_RKNN_MODELS", ROOT / "models"))
OUT = Path(os.environ.get("KOKORO_RKNN_EVAL", ROOT / "eval" / "out"))
MODEL = MODELS / "kokoro-v1.0.onnx"
VOICES = MODELS / "voices-v1.0.bin"
VOICE = "af_heart"
SR = 24000

SENTENCES = {
    "salt": "The evening is settling in over Salt Spring Island. Rain is moving in from the southwest, and by Thursday it should be steady, with highs around six degrees.",
    "sure": "Sure, I can help with that. Let me check the forecast and get back to you in just a moment.",
    "short": "Sure, I can help with that.",
    "ok": "Okay.",
    "timer": "Do you want me to set a timer for twenty-five minutes?",
    "medium": "Let me check the forecast and get back to you in just a moment.",
    "ferry": "If you leave now, take the ferry from Fulford Harbour; otherwise, the next sailing is at four thirty, and it is usually busy on Fridays.",
    "library": "The library opens at nine tomorrow morning, and the book you reserved should be waiting for you at the front desk.",
}


def session(path, threads=4, opt=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    if opt is not None:
        so.graph_optimization_level = opt
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


_k = None


def kokoro(threads=4):
    global _k
    if _k is None:
        from kokoro_onnx import Kokoro
        _k = Kokoro.from_session(session(MODEL, threads), str(VOICES))
    return _k


def tokens(text):
    k = kokoro()
    return k.tokenizer.tokenize(k.tokenizer.phonemize(text, "en-us"))


def style(n_tokens):
    v = np.load(VOICES)[VOICE]                  # (510, 1, 256): row n-1 for n tokens, as kokoro-onnx
    return v[min(n_tokens, len(v)) - 1].reshape(1, 256).astype(np.float32)


def reference(text):
    """kokoro-onnx, whole sentence, the raw model output before its silence trim."""
    toks = tokens(text)
    audio, _ = kokoro()._infer(toks, style(len(toks)), 1.0)
    return audio.astype(np.float32)


class Cpu:
    """The pieces cut from the same model (build/extract.py), on onnxruntime."""

    def __init__(self, threads=4):
        self.threads = threads
        self.front = session(MODELS / "front.onnx", threads)
        self.har = session(MODELS / "har.onnx", threads)
        self.tail_s = session(MODELS / "tail.onnx", threads)
        self._dec = None

    @property
    def dec(self):
        """The reference decoder (dec.onnx, ~200 MB; loaded when first needed)."""
        if self._dec is None:
            self._dec = session(MODELS / "dec.onnx", self.threads)
        return self._dec

    def encode(self, text):
        """Front + harmonic source on the whole sentence: asr, F0, N, s, har, dur."""
        toks = tokens(text)
        st = style(len(toks))
        asr, F0, N, dur = self.front.run(None, {"tokens": np.array([[0, *toks, 0]], np.int64), "style": st,
                                               "speed": np.array([1.0], np.float32)})
        har = self.har.run(None, {"F0": F0})[0]
        return asr, F0, N, np.ascontiguousarray(st[:, :128]), har, dur.reshape(-1)

    def full(self, asr, F0, N, s, har):
        """The reference decoder on the whole sentence (fp32)."""
        return self.dec.run(None, {"asr": asr, "F0": F0, "N": N, "s": s, "har": har})[0].reshape(-1)

    def tail(self, x_post, T):
        x = np.ascontiguousarray(x_post[:, :, :120 * T + 1], np.float32)
        return self.tail_s.run(None, {self.tail_s.get_inputs()[0].name: x})[0].reshape(-1)


def load_ref(name):
    z = np.load(OUT / "ref" / f"{name}_enc.npz")
    return np.load(OUT / "ref" / f"{name}.npy"), (z["asr"], z["F0"], z["N"], z["s"], z["har"])
