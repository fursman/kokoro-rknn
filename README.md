# kokoro-rknn

**Kokoro-82M text-to-speech with its decoder on the Rockchip RK3588 NPU, at the
quality of the CPU version.** The decoder of [Kokoro](https://github.com/hexgrad/kokoro)
(as exported by [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)) is rewritten
so that it decodes each sentence whole, in one fixed-shape NPU call. It matches the CPU
version about as closely as the CPU version matches itself under other onnxruntime
settings. Three sentences can be synthesised at once, one per NPU core.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![RK3588](https://img.shields.io/badge/NPU-RK3588-orange.svg)

Measured on a Turing RK1 (RK3588), against kokoro-onnx on the same board's CPU, voice `af_heart`:

| | kokoro-onnx (CPU, 4 × A76) | earlier windowed NPU pipeline | **kokoro-rknn** |
|---|---|---|---|
| Log-spectral distance from kokoro-onnx, 7 sentences | 0.68–1.36 dB ¹ | 10.7–14.3 dB ² | **0.72–1.50 dB** |
| Loudness difference | 0 | −3.6 to −5.9 dB | **within 0.07 dB** |
| First audio, one request ("Sure, I can help …", 5 s of speech) | 3.72 s | streamed | **2.93 s** |
| Seconds of audio made per second, one board | ~1.2–1.4 s (one sentence at a time) | | **4.6 s** with 3 in flight (5.1 s saturated) |
| CPU per second of audio | the four A76 cores, one sentence at a time | | **0.19 core-seconds** |

¹ kokoro-onnx against itself with other onnxruntime settings: the floor for "identical".
² aligned by DTW; it used another G2P, so lengths differ.

In an informal blind test (one listener, loudness-matched), the listener could not reliably
tell this version from CPU Kokoro over four A/B rounds; in one three-way round that included
the earlier NPU version, they labelled all three correctly. Method, raw numbers and caveats are in
[eval/RESULTS.md](eval/RESULTS.md).

This builds on two projects, credited throughout: [Kokoro](https://github.com/hexgrad/kokoro)
by hexgrad (Apache-2.0), the model, and [kokoro-server](https://github.com/marty1885/kokoro-server)
by marty1885, whose RK3588 work (GlobalAveragePool normalisation, the Snake folding, the
ConvTranspose rewrite) this started from. See [Credits and licence](#credits-and-licence).

---

- [Quick start](#quick-start)
- [Using it](#using-it): Python, the server and its protocol
- [How it works](#how-it-works)
- [Several sentences at once](#several-sentences-at-once): 3 cores, the ~4 GB address space, shared weights
- [Building the NPU graphs](#building-the-npu-graphs): toolkit versions, the REGTASK pitfall
- [Evaluation](#evaluation) · [Limitations](#limitations) · [Credits and licence](#credits-and-licence)

## Quick start

On an RK3588 board (tested: Ubuntu 22.04, kernel 5.10, RKNPU driver 0.9.2, librknnrt 2.5.0,
Python 3.12):

```sh
sudo apt install -y git python3-venv gcc

# Rockchip's NPU runtime (we used 2.5.0)
sudo curl -L -o /usr/lib/librknnrt.so \
  https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

git clone https://github.com/fursman/kokoro-rknn && cd kokoro-rknn
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
gcc -O3 -march=armv8.2-a -shared -fPIC -o kokoro_rknn/libinterleave.so kokoro_rknn/interleave.c
.venv/bin/python build/fetch_models.py models      # kokoro-v1.0.onnx + voices-v1.0.bin, checksummed
```

No compiled models are distributed: build them once, on the board or on an x86-64 Linux
machine, as in [Building the NPU graphs](#building-the-npu-graphs) (~15 minutes of
compiling on the board). Then:

```sh
taskset -c 4-7 .venv/bin/python -m kokoro_rknn.server --models models --port 8771
python3 examples/tts_client.py "Sure, I can help with that. Let me check the forecast and get back to you in just a moment." --server board:8771 --wav hello.wav
```

```
{"first": 2.93, "total": 2.94, "audio_s": 4.97, "rtf": 0.59, "stall": 0.0}
```

(for "Sure, I can help with that. Let me check the forecast and get back to you in just a
moment.", from another machine on the LAN)

`systemd/kokoro-rknn.service` runs it at boot.

## Using it

### From Python

```python
from kokoro_rknn.engine import load

k = load("models")                  # ~30 s: 3 NPU lanes, 8 buckets each, warmed up
samples, rate = k.create("The library opens at nine tomorrow morning.", voice="af_heart", lang="en-us")
```

`k` is a [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) `Kokoro` with one
method replaced (`_infer`), so `create`, the voices, G2P (espeak-ng), batching, silence
trimming and pauses are kokoro-onnx's own. The server calls it from several threads at
once; up to three sentences are decoded at a time, one per NPU core.

### As a server

| Request | Response |
|---|---|
| `GET /health` | `{"ok": true, "engine": "kokoro-rknn", "rate": 24000, "voices": [...], "lanes": 3, "lanes_busy": 0, ...}` |
| `POST /tts {"text": "...", "voice": "af_heart", "speed": 1.0}` | 200, raw little-endian float32 mono at `X-Sample-Rate`, chunked |

Text up to 120 characters is synthesised whole; longer text is cut at clause breaks into
balanced pieces ([kokoro_rknn/text.py](kokoro_rknn/text.py)), and each piece is sent as soon
as it is made. A client that hangs up stops the request before its next piece.

| Flag | Default | |
|---|---|---|
| `--models DIR` | `models` | made by `build/` |
| `--cores` | `1,2,0` | one decoder lane per NPU core, in order of preference |
| `--workers` | `6` | CPU session sets (front, harmonic source, iSTFT); more than lanes, so the next sentences' front runs while the NPU decodes |
| `--threads` | `4` | onnxruntime threads per session (fewer change the output slightly, at the reference's own floor) |
| `--ahead` | `1` | pieces of one request made at once |
| `--buckets` | `64,…,400` | which compiled lengths to load |
| `--host`, `--port`, `--voice` | `0.0.0.0`, `8771`, `af_heart` | |

Pin the process to the four Cortex-A76 cores (`taskset -c 4-7`, or `CPUAffinity=4-7` in
the unit). On the A55s the CPU work becomes the bottleneck: throughput falls 40% and first
audio rises by 2–3 s.

## How it works

```
 text ─▶ espeak-ng ─▶ front ─────▶ harmonic ─▶ decoder ────────────────────────▶ iSTFT ─▶ audio
         (kokoro-onnx) BERT, LSTMs  source      graph A ─▶ interleave ─▶ graph B   (CPU)
                       (CPU)        (CPU)       (NPU)      (CPU, 1-9 ms)  (NPU)
                       whole sentence           whole sentence, padded to the next bucket
```

All four pieces are cut out of kokoro-onnx's `kokoro-v1.0.onnx` ([build/extract.py](build/extract.py)),
so the weights and ops are the reference's own; only the decoder is rewritten
([build/surgery.py](build/surgery.py)).

**1. Why whole sentences.** Kokoro's decoder has 58 AdaIN/InstanceNorm layers, and every one
normalises over the time axis of whatever it is given. A window is therefore normalised
with its own statistics, not the sentence's, and the output is very sensitive to them:
perturbing the exact statistics by 1% already costs ~2–2.4 dB of log-spectral distance. With
the whole sentence's statistics injected, 12 frames of context per window would be exact;
without them, windowed decoding with 12–16 frames of context is still 5.5–8 dB away
([eval/adain_stats.py](eval/adain_stats.py)). So each sentence is decoded in one call.

**2. Buckets with masked statistics.** The NPU needs fixed shapes, so the decoder is compiled
for 8 lengths, from 64 to 400 frames (1.6–10 s). A sentence is zero-padded to the next one, and
masks come in at each of the decoder's time resolutions:
- every time-axis mean becomes `GlobalAveragePool(x · w)` with `w = mask × padded/real length`,
  so the statistics are those of the real frames only;
- the input of every convolution wider than 1 is multiplied by the mask, so the real frames
  see zeros past the end, as in the unpadded model.

In fp32 the padded, masked decode matches the unpadded one to 80 dB of waveform SNR or better
([eval/check_fp32.py](eval/check_fp32.py)): the same computation up to rounding. This works for
any static-shape runtime, not just this NPU.

**3. The 8191 width limit.** On the RK3588, an elementwise op that broadcasts over a time
axis wider than 8191 overflows a 13-bit register field. rknn-toolkit2 prints
`REGTASK: The bit width of field value exceeds the limit` while compiling, and **the
compiled model hangs the NPU** (minutes of job timeouts and soft resets). The limit in the
message is 0x1fff. The decoder's last
stage runs at 120T+1 samples, i.e. 12,001 wide for 100 frames. The workarounds:
- every elementwise op at that rate runs on a free `(1, C, H, W)` reshape of the tensor,
  with W ≤ 8160 a multiple of 32, and its masks are fed in already 4-D;
- the graph is split in two at the last upsampling layer (stride 6). Graph A runs at the frame
  rates and ends in a **polyphase** form of that layer: one convolution with 6 × 128 output
  channels. The CPU interleaves the six phases and adds the reflection pad (a 40-line C helper).
  Graph B runs the 120T+1 stage. Interleaving on the NPU forced a CPU transpose, and the
  reflection pad overflowed too.

Convolutions, GlobalAveragePool and casts are not affected by the limit.

**4. Sin, range-reduced.** The Snake activations compute sin²(αx). There is no NPU Sin, and
|αx| reaches ~120 (0.9% of all inputs exceed π; 10–14% in some layers), so a polynomial on [−π, π]
with the input clipped is wrong exactly there: that alone adds 0.5–1.5 dB LSD. sin² has
period π, so here `r = y − π·round(y/π)` and a degree-3 polynomial in r² (max error 8.5e-4).
Round and Floor fall back to the CPU in librknnrt, but `Cast float → int16 → float` runs on the NPU
and rounds to nearest (measured), so the whole thing stays on the NPU.

**5. Fewer passes.** The NPU time went mostly to elementwise ops (Mul 46%, Add 14%, convolutions
only 15%). So each AdaIN becomes one per-channel affine `x·A + B`, the Snake's α/π folds into it
and π/α into the next convolution's weights. The variance stays two-pass: a one-pass
E[x²] − μ² was 13% faster but measurably worse in fp16.

**6. The rest stays on the CPU, whole sentence:** the front (ALBERT, LSTMs, F0/energy), the
harmonic source and the model's own iSTFT. A sentence longer than the largest bucket
(10 s of speech) is decoded on the CPU by the full model.

## Several sentences at once

The RK3588 has three NPU cores. The server gives each one a **lane**: the A and B graphs of
every bucket, pinned to that core ([kokoro_rknn/lanes.py](kokoro_rknn/lanes.py)). A sentence
takes whichever lane is free; the first piece of any request goes before anyone's later
pieces. With `--cores 1,2,0`, core 0 is used only when cores 1 and 2 are both busy, which
leaves it to a speech recogniser such as [moonshine-rknn](https://github.com/fursman/moonshine-rknn).

| sentences in flight | audio made per second | first audio p50 / p90 |
|---|---|---|
| 1 | 1.68 s | 2.93 / 3.56 s |
| 2 | 3.15 s | 3.13 / 3.75 s |
| 3 | 4.58 s | 3.29 / 3.95 s |
| 4 | 5.07 s (NPU saturated) | 3.99 / 5.01 s |

One process running the decoder through rknnlite, one sentence at a time, stays at ~1.9 s/s
however many requests wait, and its first audio grows linearly with the queue.

**The ~4 GB address space.** All NPU memory, for every process on the board, comes from one
IOVA space of about 4 GB. When it runs out, `rknn_init` fails with -6 and the kernel logs
`rknpu_gem_get_pages: dma map ... fail` (harmless: nothing runs). Eight buckets' A and B graphs
take ~1.4 GB per core if each core loads its own, so three would not fit. So:
- lanes 2 and 3 are `rknn_dup_context` copies of lane 1: **the weights are loaded once**;
- a lane runs one graph at a time, so its 16 contexts share **one scratch buffer**
  (`RKNN_FLAG_INTERNAL_ALLOC_OUTSIDE`, 104 MB) and one set of I/O buffers (32 MB);
- all three lanes take ~1.4 GB together, which leaves room for a speech recogniser.

**No copies.** Inputs are written straight into the NPU's DMA buffers in its native fp16
layout, over the C API ([kokoro_rknn/rknnc.py](kokoro_rknn/rknnc.py), ctypes), and graph A's
output is interleaved straight into graph B's input. The decoder's CPU work fell from 63–223 ms
per sentence (rknnlite's conversions) to 12–17 ms. onnxruntime's threads don't spin while the NPU
works, which saved another ~30% of the CPU. The audio is bit-identical to the rknnlite path.

**librknnrt 2.5.0 quirks** found on the way, each of which cost a crash:
1. `rknn_dup_context` segfaults if the source context was initialised from a file *path*: initialise from a buffer.
2. Binding a context's I/O memory and then dup'ing it corrupts the source's bindings: create every context and dup first, then bind.
3. With `RKNN_FLAG_INTERNAL_ALLOC_OUTSIDE`, native output attributes report size 0 until `rknn_set_internal_mem` has been called: query again after it.
4. Buffers from `rknn_create_mem2(..., NON_CACHEABLE)` make numpy die with SIGBUS.
5. The model file's bytes can be freed once every dup exists and is bound (−1.1 GB of RSS).

**Next to speech recognition.** With moonshine-rknn on the same board (`NPU_CORE_AUTO`),
streamed recognition was unaffected: the final transcript still came ~1 ms after the audio
ended, except 3 of 138 utterances (0.2–0.7 s), all when synthesis was using its core.
Whole-clip recognition slowed from RTF 0.37–0.39 to 0.64 with three sentences on all three
cores, and stayed at 0.44 when synthesis kept off core 0.

**What did not help:** pipelining graph A of one sentence with graph B of another across cores
(a 3-sentence reply finished at 6.3 s instead of 4.3 s); splitting one sentence's resblocks over
the three cores (bit-identical, ~10% faster, not worth three cores); and ALBERT on the NPU.
That last one cut the front's CPU by 55–60%, but on the real NPU its fp16 error was ~10× what
the toolkit's simulator predicted: durations flipped by a frame and F0 shifted, 3.8–4.3 dB LSD.

## Building the NPU graphs

`extract.py` and `surgery.py` need only `onnx` and `numpy`. Compiling needs **rknn-toolkit2
2.3.2** (from PyPI; x86-64 or aarch64 Linux, Python ≤ 3.12), in a venv of its own because it
pins its own numpy, onnx and torch.

```sh
python3 -m venv .venv-build && .venv-build/bin/pip install -r requirements-build.txt
python3 -m venv .venv-tk && .venv-tk/bin/pip install rknn-toolkit2==2.3.2 "setuptools<70"

.venv-build/bin/python build/fetch_models.py models   # if not done yet
.venv-build/bin/python build/extract.py models        # front/har/dec/tail.onnx
build/build_buckets.sh models .venv-build/bin/python .venv-tk/bin/python
```

```
T=64 A: 0 REGTASK errors; COMPILED models/buckets/dec_T64_A.rknn in 34 s
T=64 B: 0 REGTASK errors; COMPILED models/buckets/dec_T64_B.rknn in 25 s
...
```

Each graph took 20–40 s to compile on a desktop (small buckets) and 25–90 s on the board.
The output is `models/buckets/dec_T*_{A,B}.rknn`, about 1.2 GB; the intermediate `.onnx` files
take another ~1.6 GB and are not needed at run time. Copy `models/` to the board if you built elsewhere. On
the board the toolkit also needed `opencv-python-headless`.

**Never run a model whose compile log mentions `REGTASK ... exceeds the limit`**, even once:
it hangs the NPU for minutes, and anything else on the NPU stalls too. `build_buckets.sh`
counts those lines and deletes such a model. All the graphs built here have none; you would
only see them after changing the surgery or adding buckets above 400 frames without the 4-D views.

Notes:
- Toolkit 2.5.0 also compiles these graphs but took ~3× longer on the one we timed; we
  didn't evaluate its output. librknnrt 2.5.0 runs the 2.3.2 builds.
- `surgery.py --sin rr --no-fuse` builds a variant for checking in fp32 on the CPU
  ([eval/check_fp32.py](eval/check_fp32.py)). The deployed `--sin rr_npu` uses the int16 cast,
  which onnxruntime truncates instead of rounding, so check that one on the NPU.
- Other bucket lengths work: `build_buckets.sh models ... 48 72`, then `--buckets`. Finer
  buckets waste less padding (on average 12%).

## Evaluation

[eval/RESULTS.md](eval/RESULTS.md) has every table: the reference's own floor, the attribution
of the earlier NPU pipeline's error, the NPU results per sentence, speed, loudness, concurrency
and the listening test. The scripts, run from the repository root:

```sh
pip install -r requirements-eval.txt
python eval/make_refs.py              # kokoro-onnx reference audio + front outputs -> eval/out/ref
python eval/floor.py                  # kokoro-onnx against itself: what "identical" can mean
python eval/eval_cpu.py               # windowed decoding emulated in fp32: what each property costs
python eval/adain_stats.py            # windowed decoding with the whole sentence's statistics injected
python eval/eval_npu.py               # on the board: the NPU decoder against the reference, with timings
python eval/loadtest.py --servers board:8771 --conc 3 --reqs 6
python eval/istft_level.py models/kokoro-v1.0.onnx --voices models/voices-v1.0.bin   # needs torch + kokoro
```

The metrics ([eval/metrics.py](eval/metrics.py)) are log-spectral distance (LSD), STFT-magnitude
SNR, waveform SNR, mel-cepstral distortion, multi-resolution STFT distance and BS.1770 loudness.
All variants get the same tokens and front outputs, so lengths are identical and even waveform
SNR is meaningful. kokoro-onnx against itself, with onnxruntime's optimisations off or one thread
instead of four, is the floor: LSD 0.68–1.36 dB, waveform SNR 24–32 dB.

## Limitations

- **Built from kokoro-onnx's `model-files-v1.0` export** (January 2025). `build/extract.py`
  looks its tensors up by name; the re-export in `model-files-v1.1` is a different graph and
  isn't supported yet. That v1.0 export is 3.1–3.3 LU louder than PyTorch Kokoro: its iSTFT
  skips the overlap-add normalisation (×1.5, [eval/istft_level.py](eval/istft_level.py)). This project reproduces it, so it matches
  kokoro-onnx v1.0, not PyTorch, in level.
- **No streaming within a sentence.** First audio waits for the whole sentence. The real-time
  factor is ~0.5–0.6, so a 10 s sentence takes ~4 s to start; cut long replies into sentences
  in the client.
- Sentences longer than 400 frames (10 s) are decoded on the CPU.
- Tested on Turing RK1s only, in English, mostly with `af_heart`. Other RK3588 boards use the
  same NPU and runtime and should work, but we haven't run them.
- The listening test was informal: one listener, a handful of rounds.
- RSS is ~4 GB, mostly six CPU workers' onnxruntime sessions plus the full model kept for the
  CPU fallback; `--workers 3` uses less.

## Credits and licence

- **[Kokoro](https://github.com/hexgrad/kokoro)** and the Kokoro-82M weights, by hexgrad:
  Apache-2.0. This project runs the model unchanged apart from the graph rewrites above.
- **[kokoro-server](https://github.com/marty1885/kokoro-server)** by marty1885: the first
  Kokoro decoder on the RK3588 NPU, and where this started. Its `build.py` and notes showed
  that GlobalAveragePool keeps the normalisation on the NPU, that Snake's 1/α folds into the
  next convolution, and how to replace the stride-10 ConvTranspose. Two parts of
  [build/surgery.py](build/surgery.py) are adapted from it and marked there: that ConvTranspose
  rewrite, and the clipped-polynomial Sin (only used to reproduce the earlier build in the
  evaluation). kokoro-server has no licence file; because much of it derives from Kokoro, we
  treat what we adapted as carrying Kokoro's Apache-2.0 terms, and say so in [NOTICE](NOTICE).
  Everything else here is our own code.
- **[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)** by thewh1teagle (MIT): the
  ONNX export we cut from, and the `Kokoro` class the engine extends.
- **[rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)** and librknnrt: Rockchip.

This repository is licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE).
