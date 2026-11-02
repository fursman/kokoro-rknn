"""How a request's text is cut before synthesis.

Kokoro's phrasing comes from seeing a whole sentence, so text is synthesised whole when
it fits (kokoro-onnx's create() then does the G2P, model-sized batches and pauses). Only
text that would take too long to start is cut, because first audio waits for a whole
piece: at clause breaks first, into balanced parts, never with a short tail.
"""
import re

# Cut by a fixed length, the last few words came out as a flat fragment of their own.
MAX_PIECE = 120      # characters: ~7 s of speech
MIN_PIECE = 45       # no part shorter than this


def _balanced(words, n):
    """Split a list of words into n runs of about equal length in characters."""
    total = sum(len(w) + 1 for w in words)
    out, cur, size = [], [], 0
    for w in words:
        cur.append(w); size += len(w) + 1
        if len(out) < n - 1 and size >= total * (len(out) + 1) / n:
            out.append(" ".join(cur)); cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def pieces(text):
    if len(text) <= MAX_PIECE:
        return [text]
    clauses = [c for c in re.split(r"(?<=[,;:\u2014])\s+|\s+(?=\u2014)", text) if c]
    out, cur = [], ""
    for c in clauses:                       # pack clauses greedily up to MAX_PIECE
        if cur and len(cur) + 1 + len(c) > MAX_PIECE:
            out.append(cur); cur = c
        else:
            cur = f"{cur} {c}".strip()
    if cur:
        out.append(cur)
    # No stub anywhere: a short part joins its shorter neighbour.
    i = 0
    while len(out) > 1 and i < len(out):
        if len(out[i]) < MIN_PIECE:
            j = i - 1 if i == len(out) - 1 or (i > 0 and len(out[i - 1]) <= len(out[i + 1])) else i + 1
            a_, b_ = sorted((i, j))
            out[a_:b_ + 1] = [f"{out[a_]} {out[b_]}"]
            i = 0
        else:
            i += 1
    final = []
    for piece in out:                       # a clause still too long: even word runs
        if len(piece) > MAX_PIECE * 1.25:
            final.extend(_balanced(piece.split(), -(-len(piece) // MAX_PIECE)))
        else:
            final.append(piece)
    return final
