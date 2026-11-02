# Results

Everything here was measured on Turing RK1 boards (RK3588; Ubuntu 22.04, kernel 5.10.160,
RKNPU driver 0.9.2, librknnrt 2.5.0, Python 3.12, kokoro-onnx 0.6.1, onnxruntime 1.31.1),
voice `af_heart`, speed 1.0, in November 2026. Graphs were compiled with rknn-toolkit2 2.3.2.
Raw files are in [results/](results/); the scripts are next to this file.

Names in the raw files: **npu2** is the decoder run through rknnlite, one sentence at a time,
on all three cores (`buckets.rknnlite_split`); **npu3** is this repository's server (C-API
lanes, one sentence per core). Their audio is bit-identical (checked on 4 texts), so every
quality number applies to both.

## The reference and its floor

The reference is kokoro-onnx on `kokoro-v1.0.onnx` (from kokoro-onnx's `model-files-v1.0`
release): `Kokoro._infer` on kokoro-onnx's own espeak-ng tokens, whole sentence, untrimmed.
Repeated runs are bit-identical (this export's harmonic source is deterministic).

Every variant gets the same tokens and the same front outputs (cut from the same model), so
lengths are identical and waveform SNR is meaningful. **The floor** is the reference against
itself under other onnxruntime settings; anything inside it is "the same model" as far as
these numbers can tell.

| metric | definition |
|---|---|
| wave SNR | waveform SNR |
| \|S\| SNR | SNR of STFT magnitudes, 1024/256 Hann (ignores phase) |
| LSD | log-spectral distance, dB, over frames within 50 dB of the loudest |
| MCD | 80 mel bands, natural-log amplitude, orthonormal DCT, c1–c24 (compare only within this file) |
| MR-STFT | spectral convergence + log-magnitude L1 over 512/1024/2048 |
| ΔLUFS | BS.1770-4 integrated loudness difference (pyloudnorm) |

Test sentences ([common.py](common.py)): salt (411 frames), sure (223), short (84), ok (57),
timer (131), medium (158), ferry (358), library (268). One frame is 25 ms.

[results/quality/floor.json](results/quality/floor.json):

| sentence | variant | wave SNR | \|S\| SNR | LSD | MCD | MR-STFT | ΔLUFS |
|---|---|---|---|---|---|---|---|
| salt | ORT_DISABLE_ALL | 25.6 | 32.6 | 1.09 | 1.16 | 0.08 | −0.00 |
| salt | 1 thread | 28.7 | 35.7 | 0.80 | 0.83 | 0.06 | −0.00 |
| salt | the cut pieces (front → har → dec) | 24.1 | 31.2 | 1.36 | 1.44 | 0.10 | −0.00 |
| sure | ORT_DISABLE_ALL | 30.2 | 36.5 | 0.76 | 0.77 | 0.05 | −0.00 |
| sure | 1 thread | 26.6 | 32.6 | 1.18 | 1.26 | 0.08 | −0.00 |
| sure | the cut pieces | 26.5 | 33.0 | 1.17 | 1.27 | 0.08 | −0.00 |
| short | ORT_DISABLE_ALL | 25.7 | 31.7 | 1.13 | 1.42 | 0.08 | −0.00 |
| short | the cut pieces | 25.7 | 31.8 | 1.10 | 1.36 | 0.08 | −0.00 |
| timer | ORT_DISABLE_ALL | 27.5 | 34.7 | 0.97 | 1.06 | 0.07 | −0.01 |
| medium | the cut pieces | 25.1 | 32.4 | 1.27 | 1.42 | 0.09 | −0.00 |
| ferry | 1 thread | 32.1 | 37.9 | 0.68 | 0.64 | 0.05 | 0.00 |
| library | the cut pieces | 29.3 | 35.0 | 1.02 | 1.07 | 0.07 | −0.00 |

(selected rows; "ok" is bit-exact in every variant.) **Floor over 7 sentences: wave SNR
24.1–32.1 dB, |S| SNR 31.2–37.9 dB, LSD 0.68–1.36 dB, MCD 0.64–1.44, ΔLUFS 0.00.** Waveform
SNR is low because the harmonic source integrates F0 into phase over the whole sentence, so a
1e-7 change in F0 becomes a slow phase drift; it is inaudible.

## The NPU decoder against the reference

Whole sentence, fp16, buckets 64–400, fused graphs ([results/quality/npu.json](results/quality/npu.json)):

| sentence | T / bucket | wave SNR | \|S\| SNR | LSD | MCD | ΔLUFS | floor LSD (cut pieces) |
|---|---|---|---|---|---|---|---|
| salt | 411 / – (longer than 400: the server cuts this text in two pieces) | – | – | – | – | – | 1.36 |
| sure | 223 / 256 | 25.7 | 32.4 | 1.43 | 1.66 | +0.07 | 1.17 |
| short | 84 / 96 | 25.4 | 31.7 | 1.46 | 1.77 | +0.01 | 1.10 |
| ok | 57 / 64 | 41.8 | 46.5 | 0.72 | 0.71 | +0.01 | 0.00 |
| timer | 131 / 160 | 27.5 | 34.6 | 1.21 | 1.34 | −0.02 | 0.96 |
| medium | 158 / 160 | 24.8 | 32.2 | 1.50 | 1.77 | −0.01 | 1.27 |
| ferry | 358 / 400 | 26.4 | 32.3 | 1.37 | 1.48 | +0.01 | 1.09 |
| library | 268 / 320 | 28.9 | 34.7 | 1.34 | 1.50 | +0.00 | 1.02 |

LSD is 0.1–0.4 dB above the floor of the same sentence: what fp16 costs.

**End to end, server against server** (the same request to this NPU server and to a CPU
server running kokoro-onnx `create()`, both trimmed; [results/quality/server_vs_cpu_server.json](results/quality/server_vs_cpu_server.json)):
identical lengths for all 8 sentences, LSD 0.72–1.55 dB, ΔLUFS within ±0.02.

## Why the earlier windowed NPU pipeline sounded worse

The earlier pipeline (kokoro-server's design, as we deployed it) decoded 50-frame windows
with 2 frames of context, zero-padded at the sentence edges, with the harmonic source
recomputed per window, a clipped polynomial Sin, fp16, a numpy iSTFT and misaki G2P. Each
property was emulated in fp32 with the reference decoder itself
([eval_cpu.py](eval_cpu.py), [windowed.py](windowed.py); [results/quality/attribution_cpu.json](results/quality/attribution_cpu.json)).
Ranges are min–max over the sentences:

| config | wave SNR | \|S\| SNR | LSD | LSD (DTW) | ΔLUFS |
|---|---|---|---|---|---|
| A0 the earlier pipeline as deployed (misaki G2P) | – | – | – | 10.75–14.25 | −5.93 to −3.55 |
| A1 same with espeak-ng tokens | – | – | – | 9.74–14.21 | −5.77 to −3.58 |
| A2 reference decoder fp32, 50-frame windows, ctx 2, zero pad, har per window | −3.3–1.4 | 4.4–11.0 | 7.72–12.94 | 7.52–12.45 | −2.88 to −0.25 |
| A3 = A2, whole-sentence harmonic source | −0.4–1.7 | 4.4–11.0 | 7.64–12.94 | 7.41–12.56 | −2.91 to −0.26 |
| A4 = A3, 12 frames of context (74-frame windows) | 0.1–2.6 | 7.1–12.9 | 5.51–8.19 | 5.69–8.22 | −0.51 to 0.49 |
| A5 whole sentence, masked buckets, fp32, clipped Sin | 20.1–24.3 | 26.8–30.3 | 1.54–1.99 | | −0.05 to 0.09 |
| A6 whole sentence, masked buckets, fp32, exact Sin | 25.1–97.5 | 31.8–105.3 | 0.00–1.27 | | −0.01 to 0.00 |
| A7 = A6, fused, range-reduced degree-3 sin² (fp32) | 25.1–50.2 | 31.8–56.4 | 0.12–1.27 | | −0.00 to 0.01 |
| A8 NPU fp16, whole, clipped Sin (T=160 only) | 20.4–23.2 | 26.8–29.9 | 1.75–2.15 | | −0.07 to −0.02 |
| A9 NPU fp16, whole, range-reduced Sin (T=160 only) | 24.6–37.8 | 31.5–45.0 | 0.72–1.52 | | −0.02 |
| A10 NPU fp16, whole, fused, buckets 64–400 (this repository) | 24.8–41.8 | 31.7–46.5 | 0.72–1.50 | | −0.02 to 0.07 |

In order of impact:
1. **Window-local AdaIN statistics, including the zero padding** (A2): 7.7–12.9 dB LSD even in fp32.
2. **Too little context** (A3 → A4): 12 frames instead of 2 still leaves 5.5–8.2 dB.
3. **The clipped Sin** (A6 vs A5, A9 vs A8): 0.5–1.5 dB LSD. About 0.9% of Snake inputs have |αx| > π (up to ~122; 10–14% in some layers), where clipping zeroes sin².
4. **The harmonic source restarting per window** (A2 → A3): up to ~3 dB of waveform SNR, little spectrally.
5. **misaki vs espeak-ng** (A0 → A1): 0–1.2 dB (DTW-aligned).
6. **The level** (next section).

A0 and A1 come from running that pipeline (built with kokoro-server's `build.py`) and are
compared after trimming along a DTW path; the scripts that ran it are not included here, as
they are thin wrappers around kokoro-server's code.

### The level

[results/quality/level.json](results/quality/level.json): same front outputs, the same
50-frame windowing, one stage swapped at a time (fp32, CPU), ΔLUFS against kokoro-onnx:

| stage | short | medium | timer | sure | library | mean |
|---|---|---|---|---|---|---|
| the earlier pipeline (misaki) | −5.93 | −3.63 | −4.53 | −4.31 | −4.48 | −4.58 |
| kokoro-src decoder + its harmonic source + numpy iSTFT | −3.51 | −4.32 | −3.42 | −3.66 | −3.92 | −3.76 |
| … + the reference's iSTFT instead | +0.02 | −0.79 | +0.11 | −0.14 | −0.38 | −0.24 |
| reference decoder, windowed | −0.08 | −0.79 | +0.07 | −0.24 | −0.54 | −0.31 |

So ~3.5 dB is the iSTFT, and window-local normalisation costs up to 0.8 dB more. The numpy
iSTFT is not wrong: it is the reference export that is hot. On the same spectrum,
`kokoro-v1.0.onnx`'s iSTFT gives exactly 1.5× PyTorch Kokoro's `torch.istft` (+3.52 dB;
residual −123 dB). Its overlap-add normalisation buffer (`window_sum`) is a 20-sample
constant in the graph, so it only applies to the first output samples, and the 1.5× of a
Hann window at 75% overlap is never divided out. End to end, PyTorch Kokoro (hexgrad/kokoro
0.9.4, `KModel`) with the same tokens and voice is 3.1–3.3 LU quieter than that export.
kokoro-onnx's newer re-export (`model-files-v1.1`) matches PyTorch to within 0.1 LU. This
repository is cut from the v1.0 export and reproduces its level. Script: [istft_level.py](istft_level.py)
(run with kokoro 0.9.4, kokoro-onnx 0.6.1, torch 2.15 CPU):

| sentence | v1.0 export − KModel | v1.1 export − KModel |
|---|---|---|
| salt | +3.14 LU | −0.07 LU |
| sure | +3.29 LU | +0.07 LU |
| short | +3.22 LU | −0.07 LU |
| timer | +3.20 LU | −0.09 LU |

iSTFT alone, v1.0 export against `torch.istft` on KModel's own spectrum: gain 1.499999
(+3.52 dB), residual after that gain −122 dB.

## Why windowed decoding can't match: the AdaIN statistics

[adain_stats.py](adain_stats.py) takes the decoder's 108 time-axis means (58 AdaIN/InstanceNorm
layers: 20 means at the frame rates, 88 in the generator) from the whole-sentence pass and
injects them into windowed decodes (first window 20 frames, then 40, hard cuts), against the
whole-sentence decode. fp32, run on x86-64 (the same numbers came out on the board):

| statistics | ctx (frames per side) | short: \|S\| SNR / LSD | medium: \|S\| SNR / LSD |
|---|---|---|---|
| window-local (none injected) | 2 | 4.8 / 13.22 | 7.0 / 9.69 |
| | 8 | 6.9 / 11.20 | 10.2 / 7.59 |
| | 16 | 12.4 / 6.45 | 10.6 / 6.69 |
| all 108 from the whole sentence | 2 | 16.9 / 2.33 | 22.2 / 1.67 |
| | 4 | 21.8 / 1.33 | 29.4 / 0.79 |
| | 8 | 35.5 / 0.37 | 41.9 / 0.16 |
| | **12** | **78.2 / 0.00** (wave SNR 75.1) | **78.6 / 0.00** (wave SNR 74.3) |
| | 16 | 124.2 / 0.00 | 122.0 / 0.00 |
| only the 20 frame-rate ones | 8 | 8.4 / 9.41 | 10.7 / 6.63 |
| only the 88 generator ones | 8 | 8.3 / 7.72 | 12.8 / 5.71 |

With exact statistics the decoder's receptive field is ~12 frames per side (300 ms).
Without them no context size helps much, and both groups of layers matter.

How exact must the statistics be? Whole sentence, all statistics multiplied by (1 + ε·N(0,1)):

| ε | short: wave SNR / \|S\| SNR / LSD | medium |
|---|---|---|
| 0.01% | 60.9 / 68.9 / 0.03 | 60.2 / 68.3 / 0.03 |
| 0.1% | 40.1 / 46.6 / 0.27 | 39.6 / 46.3 / 0.25 |
| 1% | 17.5 / 25.8 / 2.38 | 17.8 / 26.1 / 2.38 |
| 3% | 9.5 / 17.0 / 4.11 | 9.7 / 17.6 / 3.89 |

A 1% error already costs more than the floor. Getting the statistics to ~0.1% means having every
layer's activations over the whole sentence, which is the whole decode, so a streaming
decoder can't get them in advance. Statistics estimated from a sample of the sentence did
no better (50% coverage: 14.8 dB |S| SNR on the long sentence).

## Speed

**One request at a time**, over HTTP from another machine on the LAN, median of 3 warm runs.
CPU = kokoro-onnx `create()` with 4 threads on the four A76s; NPU = the rknnlite path, which
spreads one sentence over all three cores ([results/quality/http_npu.json](results/quality/http_npu.json),
[http_cpu.json](results/quality/http_cpu.json)):

| sentence | audio | CPU first / total / RTF | NPU first / total / RTF | decoder (bucket) | front + har |
|---|---|---|---|---|---|
| salt | 8.92 s | 3.90 / 6.72 / 0.75 | 2.62 / 4.65 / 0.52 | 2 pieces | 635 ms |
| sure | 4.97 s | 3.72 / 3.72 / 0.75 | 2.60 / 2.61 / 0.52 | 2181 ms (256) | 409 ms |
| short | 1.56 s | 1.39 / 1.39 / 0.89 | 0.95 / 0.95 / 0.61 | 792 ms (96) | 147 ms |
| ok | 0.75 s | 0.93 / 0.93 / 1.25 | 0.75 / 0.76 / 1.01 | 680 ms (64) | 125 ms |
| timer | 2.62 s | 2.21 / 2.22 / 0.85 | 1.59 / 1.59 / 0.61 | 1331 ms (160) | 263 ms |
| medium | 3.31 s | 2.63 / 2.64 / 0.80 | 1.61 / 1.62 / 0.49 | 1330 ms (160) | 268 ms |
| ferry | 8.28 s | 5.87 / 5.88 / 0.71 | 3.96 / 3.96 / 0.48 | 3371 ms (400) | 586 ms |
| library | 6.02 s | 4.41 / 4.42 / 0.73 | 3.16 / 3.16 / 0.53 | 2643 ms (320) | 461 ms |

The decoder costs 8.3–10.6 ms per padded frame on the NPU. The server's lanes put a sentence
on one core, which is ~10% slower for a lone request (first audio 2.93 s for "sure" instead
of 2.60 s) and is what lets three run at once.

## Several sentences at once

Closed loop from another machine: N clients, each sending its next request when the previous
one finished, cycling through salt (8.9 s of audio, 2 pieces), sure (5.0 s) and a long reply
(13.8 s, 3 pieces). "TTS CPU" is the server process; NPU load from `/sys/kernel/debug/rknpu/load`
([results/concurrency/](results/concurrency/)):

| server | N | audio s per s | sure first p50 / max | long first / total | TTS CPU (cores) | NPU load % (core 0 / 1 / 2) |
|---|---|---|---|---|---|---|
| rknnlite, one at a time | 1 | 1.79 | 3.01 / 3.34 s | 3.10 / 7.30 s | 0.535 | 74 / 14 / 11 |
| | 2 | 1.91 | 5.43 / 5.82 s | 5.33 / 14.60 s | 0.556 | 78 / 12 / 12 |
| | 3 | 1.90 | 8.25 / 8.52 s | 7.50 / 21.23 s | 0.566 | 78 / 12 / 12 |
| **lanes (this server), 4 threads, A76** | 1 | 1.69 | 2.93 / 2.93 s | 3.46 / 8.21 s | 0.316 | 32 / 26 / 25 |
| | 2 | 3.25 | 3.06 / 3.32 s | 3.60 / 8.51 s | 0.629 | 53 / 56 / 52 |
| | 3 | 4.58 | 3.13 / 3.65 s | 3.94 / 9.17 s | 0.888 | 77 / 79 / 80 |
| lanes, 2 threads, A55 only | 3 | 2.69 | 5.08 / 6.50 s | 6.67 / 15.21 s | 2.267 | 45 / 47 / 47 |
| lanes, 4 threads × 6 workers (the unit file) | 4 | 4.87 | 3.48 s | 4.16 / – s | 0.99 | ~87 |
| | 6 | 4.88 | 4.23 s | 5.04 / – s | 1.00 | ~87 |

- **CPU per second of audio:** 0.19 core-seconds with lanes, against 0.30 for the rknnlite
  server, and the four A76 cores for one sentence at a time with kokoro-onnx on the CPU.
- **The bottleneck is the NPU** on the A76s: at saturation the process uses at most one of
  the four cores. Moved to the A55s, the CPU becomes the bottleneck.
- **Load test with random arrivals** ([results/concurrency/pool_open.json](results/concurrency/pool_open.json)):
  at 0.3 requests/s per board (mean 9 s of audio each), first audio p50/p90 3.29/3.90 s, as
  unloaded; at 0.45/s, 3.87/5.43 s.
- **Per sentence, CPU** (`profile_*.json`, "sure"): front 805 ms of CPU (ALBERT 62% of it),
  harmonic source 157 ms, decoder glue 12 ms, iSTFT tail 11 ms.

### Next to speech recognition

[moonshine-rknn](https://github.com/fursman/moonshine-rknn) on the same board (`--npu-core auto`),
6 utterances per condition streamed at the pace of speech, plus the same as whole clips, while
N synthesis clients ran ([results/stt_alongside/](results/stt_alongside/)):

| synthesis lanes | N | streamed final p50 / max | whole-clip RTF p50 / max |
|---|---|---|---|
| – | 0 | 1 / 1 ms | 0.386 / 0.428 |
| cores 0, 1, 2 | 3 | 1 / 210 ms | 0.637 / 0.714 |
| cores 1, 2 | 3 | 1 / 1 ms | 0.438 / 0.514 |
| cores 1, 2, then 0 (default) | 2 | 1 / 1 ms | 0.46 / 0.501 |
| | 3 | 1 / 2 ms | 0.633 / 0.741 |

Word error rate was ~0.06 in every condition (number formatting and place names), unchanged
by load.

## Blind listening

Informal, one listener. Files were 24 kHz mono 16-bit, loudness-matched to −23 LUFS with a
pure linear gain, padded with silence to one length per group, and shuffled with
`secrets.SystemRandom` ([make_samples.py](make_samples.py)); the samples were the salt and
sure sentences, the CPU version from a kokoro-onnx server and the NPU version from this
decoder's server.

- **This version against CPU Kokoro (kokoro-onnx), 4 A/B rounds:** the listener could not
  reliably tell which was which.
- **Three-way, CPU / this version / the earlier windowed NPU version, one round (salt):** the
  listener labelled all three correctly. The earlier version is clearly distinguishable; one
  round can't say whether there is also a small audible difference between this version and
  the CPU, or whether that part was luck (1 in 6 for all three by chance).

This is a sanity check, not a listening study.
