"""HTTP server: Kokoro-82M with the decoder on the RK3588 NPU.

    GET  /health   {"ok": true, "engine": "kokoro-rknn", "rate": 24000, "voices": [...], "voice": ...,
                    "lanes": 3, "lanes_busy": n, "requests": n, "npu": n, "cpu_fallback": n}
    POST /tts      {"text": "...", "voice": "af_heart", "speed": 1.0}
                   -> 200, raw little-endian float32 mono at X-Sample-Rate, chunked,
                      one chunk per piece of text (see text.py)

A client that hangs up stops the request before its next piece.

    python -m kokoro_rknn.server --models models --port 8771
"""
import argparse
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from aiohttp import web

from .buckets import BUCKETS
from .engine import load
from .text import pieces

log = logging.getLogger("kokoro-rknn")
SR = 24000


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="models")
    ap.add_argument("--buckets", default=",".join(map(str, BUCKETS)))
    ap.add_argument("--cores", default="1,2,0", help="one decoder lane per NPU core listed, in order of preference")
    ap.add_argument("--workers", type=int, default=6, help="CPU session sets (front, harmonic source, tail)")
    ap.add_argument("--threads", type=int, default=4, help="onnxruntime threads per session")
    ap.add_argument("--ahead", type=int, default=1, help="pieces of one request made at once (1: in turn)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--voice", default="af_heart")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    cores = [int(c) for c in a.cores.split(",")]
    k = load(a.models, cores, [int(b) for b in a.buckets.split(",")], a.workers, a.threads)
    k.create("Ready.", voice=a.voice, lang="en-us")
    voices = sorted(k.get_voices())
    log.info(f"ready in {time.time() - t0:.1f}s: buckets {a.buckets}, lanes on NPU cores {cores}, "
             f"{a.workers} CPU workers x {a.threads} threads, ahead {a.ahead}")
    pool = ThreadPoolExecutor(max_workers=max(8, 4 * a.workers))
    active = {"requests": 0}

    async def health(_req):
        return web.json_response({"ok": True, "engine": "kokoro-rknn", "rate": SR, "voices": voices,
                                  "voice": a.voice, "lanes": len(k.lanes), "lanes_busy": k.pool.busy,
                                  "requests": active["requests"], **k.stats})

    async def speak(req):
        body = await req.json()
        text = str(body.get("text", "")).strip()
        voice = str(body.get("voice") or a.voice)
        speed = float(body.get("speed") or 1.0)
        if not text:
            return web.json_response({"error": "no text"}, status=400)
        if voice not in voices:
            voice = a.voice
        t0 = time.perf_counter()
        resp = web.StreamResponse(headers={"Content-Type": "application/octet-stream",
                                           "X-Sample-Rate": str(SR), "X-Sample-Format": "f32le"})
        await resp.prepare(req)
        loop = asyncio.get_running_loop()
        samples, first = 0, None
        parts = pieces(text)
        active["requests"] += 1

        def run(i, piece):
            return k.create_prio(i, piece, voice=voice, speed=speed, lang="en-us")[0]
        futs = {}
        try:
            for i in range(len(parts)):
                for j in range(i, min(len(parts), i + max(1, a.ahead))):
                    if j not in futs:
                        futs[j] = loop.run_in_executor(pool, run, j, parts[j])
                audio = await futs.pop(i)
                if first is None:
                    first = time.perf_counter() - t0
                samples += len(audio)
                await resp.write(np.asarray(audio, dtype="<f4").tobytes())
        except (ConnectionResetError, asyncio.CancelledError):
            log.info(f"client left after {samples / SR:.2f}s of audio: {text[:60]!r}")
            raise
        except Exception as e:
            log.warning(f"synthesis failed: {e}")
        finally:
            active["requests"] -= 1
        await resp.write_eof()
        wall = time.perf_counter() - t0
        if samples:
            log.info(f"{samples / SR:5.2f}s audio in {wall:4.2f}s (RTF {wall / (samples / SR):.2f}, "
                     f"first {first * 1000:.0f} ms) {text[:60]!r}")
        return resp

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/tts", speak)
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
