#!/usr/bin/env python3
"""Time a /tts request the way a player experiences it, and optionally save it.
Standard library only.

    python3 examples/tts_client.py "Hello there." [--server board:8771] [--wav out.wav] [--voice af_heart]

Prints the time to the first audio byte, the total time, the audio length, the real-time
factor and stalls: playback is assumed to start when the first audio arrives, and any
time a chunk arrives after the audio before it has finished playing counts as a gap.
"""
import argparse
import array
import http.client
import json
import sys
import time
import wave


def run(server, text, voice="af_heart", speed=1.0):
    host, _, port = server.rpartition(":")
    c = http.client.HTTPConnection(host or "127.0.0.1", int(port), timeout=60)
    body = json.dumps({"text": text, "voice": voice, "speed": speed})
    t0 = time.perf_counter()
    c.request("POST", "/tts", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {r.read()[:200]!r}")
    sr = int(r.getheader("X-Sample-Rate", "24000"))
    buf = bytearray()
    first = None
    play_end = None          # wall time at which the audio received so far finishes playing
    stall = 0.0
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        now = time.perf_counter() - t0
        n_before = len(buf) // 4
        buf += chunk
        n = len(buf) // 4 - n_before
        if n <= 0:
            continue
        if first is None:
            first = now
            play_end = now
        if now > play_end:
            stall += now - play_end
            play_end = now
        play_end += n / sr
    total = time.perf_counter() - t0
    audio = array.array("f", bytes(buf[: len(buf) // 4 * 4]))      # float32
    if sys.byteorder != "little":
        audio.byteswap()
    return {"first": first, "total": total, "audio_s": len(audio) / sr,
            "rtf": total / max(1e-9, len(audio) / sr), "stall": stall}, audio, sr


def write_wav(path, audio, sr):
    pcm = array.array("h", (int(max(-1.0, min(1.0, x)) * 32767) for x in audio))
    if sys.byteorder != "little":
        pcm.byteswap()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--server", default="127.0.0.1:8771")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--wav")
    a = ap.parse_args()
    res, audio, sr = run(a.server, a.text, a.voice, a.speed)
    if a.wav:
        write_wav(a.wav, audio, sr)
    print(json.dumps({k: round(v, 2) for k, v in res.items()}))
