#!/usr/bin/env python3
"""Closed-loop load on one or more TTS servers: N clients per server, each sending requests back
to back (cycling through the texts), R requests each. Records time to first audio byte, total
time and audio length per request; throughput = audio seconds made per wall-clock second.

    python eval/loadtest.py --servers board:8771[,board2:8771...] --conc 3 --reqs 6 --out n3.json
"""
import argparse, http.client, json, threading, time
import statistics as st
TEXTS = {
    "salt": "The evening is settling in over Salt Spring Island. Rain is moving in from the southwest, and by Thursday it should be steady, with highs around six degrees.",
    "sure": "Sure, I can help with that. Let me check the forecast and get back to you in just a moment.",
    "long": "If you leave now, take the ferry from Fulford Harbour; otherwise, the next sailing is at four thirty, and it is usually busy on Fridays. The library opens at nine tomorrow morning, and the book you reserved should be waiting for you at the front desk.",
    "short": "Sure, I can help with that.",
    "library": "The library opens at nine tomorrow morning, and the book you reserved should be waiting for you at the front desk.",
}


def once(hp, text, timeout=120):
    host, port = hp.split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=timeout)
    t0 = time.perf_counter()
    c.request("POST", "/tts", body=json.dumps({"text": text, "voice": "af_heart", "speed": 1.0}),
              headers={"Content-Type": "application/json"})
    r = c.getresponse(); first = None; n = 0
    while True:
        d = r.read1(65536)
        if not d: break
        if first is None: first = time.perf_counter() - t0
        n += len(d)
    total = time.perf_counter() - t0; c.close()
    return first, total, n / 4 / 24000


def run(servers, conc, reqs, texts, stagger=0.0, t_ref=None):
    recs, lock = [], threading.Lock()
    t_ref = t_ref or time.perf_counter()

    def client(hp, ci):
        time.sleep(stagger * ci)
        for r in range(reqs):
            name = texts[(ci + r) % len(texts)]
            ts = time.perf_counter() - t_ref
            try:
                f, t, a = once(hp, TEXTS[name])
                rec = dict(server=hp, client=ci, text=name, start=round(ts, 3), first=f, total=t, audio=a)
            except Exception as e:
                rec = dict(server=hp, client=ci, text=name, start=round(ts, 3), error=str(e))
            with lock: recs.append(rec)
    th = [threading.Thread(target=client, args=(hp, ci)) for hp in servers for ci in range(conc)]
    t0 = time.perf_counter(); [x.start() for x in th]; [x.join() for x in th]; wall = time.perf_counter() - t0
    return recs, wall


def summary(recs, wall):
    ok = [r for r in recs if "error" not in r]
    s = {"requests": len(recs), "errors": len(recs) - len(ok), "wall_s": round(wall, 2),
         "audio_s": round(sum(r["audio"] for r in ok), 2),
         "throughput_audio_s_per_s": round(sum(r["audio"] for r in ok) / wall, 3)}
    by = {}
    for r in ok: by.setdefault(r["text"], []).append(r)
    s["per_text"] = {k: dict(n=len(v), audio_s=round(v[0]["audio"], 2),
                             first_med=round(st.median(x["first"] for x in v), 3), first_max=round(max(x["first"] for x in v), 3),
                             total_med=round(st.median(x["total"] for x in v), 3), total_max=round(max(x["total"] for x in v), 3),
                             rtf_med=round(st.median(x["total"] / x["audio"] for x in v), 3)) for k, v in by.items()}
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", required=True); ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--reqs", type=int, default=3); ap.add_argument("--texts", default="salt,sure,long")
    ap.add_argument("--stagger", type=float, default=0.0); ap.add_argument("--out")
    a = ap.parse_args()
    servers = a.servers.split(",")
    for hp in servers: once(hp, "Hello.")                      # warm
    recs, wall = run(servers, a.conc, a.reqs, a.texts.split(","), a.stagger)
    s = summary(recs, wall)
    print(json.dumps(s, indent=1))
    if a.out: json.dump({"summary": s, "records": recs, "args": vars(a)}, open(a.out, "w"), indent=1)
