#!/usr/bin/env python3
"""Reference audio and front outputs for every test sentence -> eval/out/ref/.

    <name>.npy      kokoro-onnx, whole sentence, untrimmed (run twice: is it deterministic?)
    <name>_enc.npz  asr, F0, N, s, har, dur from front.onnx / har.onnx (the same model)
"""
import json

import numpy as np

from common import OUT, SENTENCES, SR, Cpu, kokoro, reference, tokens

out = OUT / "ref"
out.mkdir(parents=True, exist_ok=True)
cpu = Cpu()
info = {}
for name, text in SENTENCES.items():
    a, b = reference(text), reference(text)
    asr, F0, N, s, har, dur = cpu.encode(text)
    np.save(out / f"{name}.npy", a)
    np.savez(out / f"{name}_enc.npz", asr=asr, F0=F0, N=N, s=s, har=har, dur=dur)
    info[name] = dict(chars=len(text), tokens=len(tokens(text)), T=int(asr.shape[2]), seconds=a.size / SR,
                      repeat_identical=bool(np.array_equal(a, b)), length_matches_front=int(asr.shape[2]) * 600 == a.size,
                      phonemes=kokoro().tokenizer.phonemize(text, "en-us"))
    print(name, info[name], flush=True)
json.dump(info, open(out / "info.json", "w"), indent=1)
