"""Recursive character splitter: prefer paragraph, then line, sentence, word boundaries.
Overlap is built from trailing pieces of the previous chunk, so it also respects boundaries."""
from __future__ import annotations

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _pieces(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:  # no separator left: hard slice
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    if sep not in text:
        return _pieces(text, size, rest)
    parts = text.split(sep)
    parts = [p + sep for p in parts[:-1]] + [parts[-1]]  # keep separator => lossless join
    out: list[str] = []
    for p in parts:
        if p:
            out += _pieces(p, size, rest) if len(p) > size else [p]
    return out


def split_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    if size <= 0 or not (0 <= overlap < size):
        raise ValueError("need size > 0 and 0 <= overlap < size")
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for piece in _pieces(text, size, SEPARATORS):
        if cur and cur_len + len(piece) > size:
            chunks.append("".join(cur))
            tail: list[str] = []
            t = 0
            for q in reversed(cur):          # carry trailing pieces totalling <= overlap
                if t + len(q) > overlap:
                    break
                tail.insert(0, q)
                t += len(q)
            cur, cur_len = tail, t
        cur.append(piece)
        cur_len += len(piece)
    if cur:
        chunks.append("".join(cur))
    out = [c.strip() for c in chunks if c.strip()]
    return out
