"""Objective comparison against the reference (kokoro-onnx whole-sentence output).

All metrics take x (test) and ref (reference), float32 at 24 kHz, same length
(both are the untrimmed model output for the same tokens, so the durations and
hence the lengths are identical).

  wave_snr   waveform SNR, dB (phase-sensitive)
  mag_snr    SNR of STFT magnitudes (1024/256 Hann), dB (ignores phase)
  lsd        log-spectral distance, dB, frames within 50 dB of the loudest ref frame
  mcd        mel-cepstral distortion, dB (80 mel, c1..c24, same frames)
  mrstft     multi-resolution STFT distance (spectral convergence + log-mag L1,
             averaged over 512/128, 1024/256, 2048/512)
  dlufs      integrated loudness difference x - ref, LU (BS.1770-4, pyloudnorm)
  seam_db    (windowed decodes only) max over seams of |log-energy step across the
             seam (5 ms frames)| in x minus the same in ref, dB
"""
import numpy as np

SR = 24000


def stft_mag(x, n_fft=1024, hop=256):
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    x = np.pad(x.astype(np.float32), n_fft // 2, mode="reflect")
    n = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * win, axis=1)).astype(np.float64)


def _loud_frames(S, top_db=50.0):
    e = 10 * np.log10(np.sum(S ** 2, axis=1) + 1e-20)
    return e > e.max() - top_db


def mel_fb(n_mels=80, n_fft=1024, sr=SR, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, bins.size))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0, None)
    return fb


_FB = None


def mcd(x, ref, n_ceps=24):
    global _FB
    if _FB is None:
        _FB = mel_fb()
    Sx, Sr = stft_mag(x), stft_mag(ref)
    keep = _loud_frames(Sr)
    from scipy.fft import dct
    cx = dct(np.log(Sx[keep] ** 2 @ _FB.T + 1e-10), type=2, norm="ortho", axis=1)[:, 1:n_ceps + 1]
    cr = dct(np.log(Sr[keep] ** 2 @ _FB.T + 1e-10), type=2, norm="ortho", axis=1)[:, 1:n_ceps + 1]
    # natural-log mel energies: c = 2 ln|.|, MCD = 10/ln10 * sqrt(2 sum (dc)^2) with c in ln(amplitude)
    d = (cx - cr) / 2.0
    return float(np.mean(10 / np.log(10) * np.sqrt(2 * np.sum(d ** 2, axis=1))))


def lsd(x, ref):
    Sx, Sr = stft_mag(x), stft_mag(ref)
    keep = _loud_frames(Sr)
    lx = 20 * np.log10(Sx[keep] + 1e-5)
    lr = 20 * np.log10(Sr[keep] + 1e-5)
    return float(np.mean(np.sqrt(np.mean((lx - lr) ** 2, axis=1))))


def mag_snr(x, ref):
    Sx, Sr = stft_mag(x), stft_mag(ref)
    return float(10 * np.log10(np.sum(Sr ** 2) / (np.sum((Sx - Sr) ** 2) + 1e-30)))


def wave_snr(x, ref):
    return float(10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / (np.sum((x.astype(np.float64) - ref) ** 2) + 1e-30)))


def mrstft(x, ref):
    out = []
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        Sx, Sr = stft_mag(x, n_fft, hop), stft_mag(ref, n_fft, hop)
        sc = np.linalg.norm(Sr - Sx) / (np.linalg.norm(Sr) + 1e-20)
        lm = np.mean(np.abs(np.log(Sx + 1e-5) - np.log(Sr + 1e-5)))
        out.append(sc + lm)
    return float(np.mean(out))


def lufs(x):
    import pyloudnorm as pyln
    return float(pyln.Meter(SR).integrated_loudness(x.astype(np.float64)))


def seam_db(x, ref, seams, frame=120):
    """seams: sample positions. Energy step across each seam in x, minus ref's step there."""
    if not seams:
        return None
    worst = 0.0
    for s in seams:
        if s - frame < 0 or s + frame > x.size:
            continue
        def step(a):
            e1 = 10 * np.log10(np.mean(a[s - frame:s].astype(np.float64) ** 2) + 1e-12)
            e2 = 10 * np.log10(np.mean(a[s:s + frame].astype(np.float64) ** 2) + 1e-12)
            return abs(e2 - e1)
        worst = max(worst, step(x) - step(ref))
    return float(worst)


def all_metrics(x, ref, seams=None):
    n = min(x.size, ref.size)
    x, ref = x[:n], ref[:n]
    return dict(wave_snr=wave_snr(x, ref), mag_snr=mag_snr(x, ref), lsd=lsd(x, ref), mcd=mcd(x, ref),
                mrstft=mrstft(x, ref), dlufs=lufs(x) - lufs(ref), seam_db=seam_db(x, ref, seams or []))


def _mceps(x, n_ceps=24):
    global _FB
    if _FB is None:
        _FB = mel_fb()
    from scipy.fft import dct
    S = stft_mag(x)
    return S, dct(np.log(S ** 2 @ _FB.T + 1e-10), type=2, norm="ortho", axis=1)[:, 1:n_ceps + 1] / 2.0


def _dtw_path(D):
    n, m = D.shape
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0
    for i in range(1, n + 1):
        row = D[i - 1]
        for j in range(1, m + 1):
            C[i, j] = row[j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return path[::-1]


def trim(x, top_db=60.0):
    """kokoro-onnx's trim (librosa: 2048/512, 60 dB below the loudest frame)."""
    pad = np.pad(x.astype(np.float32), 1024)
    n = 1 + (pad.size - 2048) // 512
    idx = np.arange(2048)[None, :] + 512 * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(pad[idx] ** 2, axis=1))
    loud = np.flatnonzero(rms > rms.max() * 10 ** (-top_db / 20))
    return x[loud[0] * 512:min(x.size, (loud[-1] + 1) * 512)] if loud.size else x


def dtw_metrics(x, ref):
    """For outputs of different length (other G2P / dropped frames): both trimmed, mel-cepstra
    aligned by DTW (band-free, Euclidean), then MCD and LSD along the path."""
    x, ref = trim(x), trim(ref)
    Sx, cx = _mceps(x)
    Sr, cr = _mceps(ref)
    D = np.sqrt(((cx[:, None, :] - cr[None, :, :]) ** 2).sum(-1))
    path = _dtw_path(D)
    ix = np.array([p[0] for p in path]); ir = np.array([p[1] for p in path])
    keep = _loud_frames(Sr)[ir]
    d = cx[ix[keep]] - cr[ir[keep]]
    mcd_ = float(np.mean(10 / np.log(10) * np.sqrt(2 * np.sum(d ** 2, axis=1))))
    lx = 20 * np.log10(Sx[ix[keep]] + 1e-5); lr = 20 * np.log10(Sr[ir[keep]] + 1e-5)
    lsd_ = float(np.mean(np.sqrt(np.mean((lx - lr) ** 2, axis=1))))
    return dict(mcd_dtw=mcd_, lsd_dtw=lsd_, dlufs=lufs(x) - lufs(ref), dur_ratio=x.size / ref.size)
