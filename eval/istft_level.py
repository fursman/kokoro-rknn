#!/usr/bin/env python3
"""Output level of kokoro-onnx's ONNX exports against PyTorch Kokoro (KModel), and why.

    pip install kokoro==0.9.4 kokoro-onnx==0.6.1 onnx pyloudnorm
    python eval/istft_level.py models/kokoro-v1.0.onnx [other.onnx ...] --voices models/voices-v1.0.bin

1. End to end: the same espeak-ng tokens (kokoro-onnx's tokenizer), the same voice row and
   speed, through each ONNX file (Kokoro._infer, untrimmed) and through KModel (default
   torch.istft path; its harmonic source is random, so seed 0). Prints the loudness
   difference (BS.1770, LU) and the lengths.
2. Only the iSTFT: KModel's own (magnitude, phase) for one sentence, inverted by torch.istft
   and by each ONNX file's iSTFT sub-graph (cut from the generator's Exp/Sin outputs to
   'audio'), when the file has those tensors. Prints the least-squares gain between them.

With kokoro-onnx's model-files-v1.0 kokoro-v1.0.onnx the iSTFT gain is exactly 1.5 (+3.52 dB):
its window_sum buffer is a 20-sample constant, so the overlap-add normalisation only reaches
the first output samples. The model-files-v1.1 re-export matches KModel within 0.1 LU.
"""
import argparse

import numpy as np
import onnx
import onnxruntime as ort
import pyloudnorm as pyln
import torch
from kokoro import KModel
from kokoro.istftnet import TorchSTFT
from kokoro_onnx import Kokoro
from onnx.utils import Extractor

TEXTS = ["The evening is settling in over Salt Spring Island. Rain is moving in from the southwest, and by Thursday it should be steady, with highs around six degrees.",
         "Sure, I can help with that. Let me check the forecast and get back to you in just a moment.",
         "Sure, I can help with that.",
         "Do you want me to set a timer for twenty-five minutes?"]
SPEC, PHASE = "/decoder/decoder/generator/Exp_output_0", "/decoder/decoder/generator/Sin_output_0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--voices", required=True)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--kmodel-dir", help="a local copy of hexgrad/Kokoro-82M (config.json, kokoro-v1_0.pth)")
    a = ap.parse_args()
    meter = pyln.Meter(24000)
    lufs = lambda x: float(meter.integrated_loudness(np.asarray(x, np.float64)))
    kw = dict(config=f"{a.kmodel_dir}/config.json", model=f"{a.kmodel_dir}/kokoro-v1_0.pth") if a.kmodel_dir else {}
    km = KModel(repo_id="hexgrad/Kokoro-82M", **kw).eval()
    pack = np.load(a.voices)[a.voice]
    kos = {p: Kokoro(p, a.voices) for p in a.onnx}
    tok = next(iter(kos.values())).tokenizer

    print("1. end to end: LUFS(onnx) - LUFS(KModel), same tokens, voice and speed")
    for text in TEXTS:
        toks = tok.tokenize(tok.phonemize(text, "en-us"))
        style = pack[len(toks) - 1].reshape(1, 256).astype(np.float32)
        torch.manual_seed(0)
        ref, _ = km.forward_with_tokens(torch.LongTensor([[0, *toks, 0]]), torch.from_numpy(style), 1.0)
        ref = ref.numpy()
        for p, k in kos.items():
            y, _ = k._infer(toks, style, 1.0)
            print(f"   {lufs(y) - lufs(ref):+6.2f} LU  len {y.size} vs {ref.size}  {p}  {text[:40]!r}")

    print("2. the iSTFT alone: gain of each export's iSTFT against torch.istft on KModel's own spectrum")
    captured = {}
    gen = km.decoder.generator
    inverse = gen.stft.inverse

    def grab(mag, phase):
        captured["mag"], captured["phase"] = mag.clone(), phase.clone()
        return inverse(mag, phase)
    gen.stft.inverse = grab
    toks = tok.tokenize(tok.phonemize(TEXTS[1], "en-us"))
    style = pack[len(toks) - 1].reshape(1, 256).astype(np.float32)
    torch.manual_seed(0)
    km.forward_with_tokens(torch.LongTensor([[0, *toks, 0]]), torch.from_numpy(style), 1.0)
    mag, ph = captured["mag"], captured["phase"]
    yt = TorchSTFT(filter_length=20, hop_length=5, win_length=20).inverse(mag, ph).numpy().ravel()
    for p in a.onnx:
        m = onnx.shape_inference.infer_shapes(onnx.load(p))
        names = {o for n in m.graph.node for o in n.output}
        if not {SPEC, PHASE} <= names:
            print(f"   {p}: no {SPEC} / {PHASE} tensors (another graph layout); skipped")
            continue
        sub = Extractor(m).extract_model([SPEC, PHASE], ["audio"])
        s = ort.InferenceSession(sub.SerializeToString(), providers=["CPUExecutionProvider"])
        ye = s.run(None, {SPEC: mag.numpy(), PHASE: ph.numpy()})[0].ravel()
        n = min(ye.size, yt.size)
        mid = slice(1000, n - 1000)
        g = float(np.dot(ye[mid], yt[mid]) / np.dot(yt[mid], yt[mid]))
        resid = float(10 * np.log10(np.sum((ye[mid] - g * yt[mid]) ** 2) / np.sum(ye[mid] ** 2)))
        print(f"   {p}: gain {g:.6f} ({20 * np.log10(g):+.2f} dB), residual after the gain {resid:.0f} dB")


if __name__ == "__main__":
    main()
