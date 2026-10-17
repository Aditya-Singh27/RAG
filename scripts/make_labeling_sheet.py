"""Create a BLIND labeling sheet from a candidate pool.

  python scripts/make_labeling_sheet.py --pool data/labeling/pool.json --n 30 --exclude data/judge_suites/human_labeled_cases.json

Writes  data/labeling/sheet.csv   (what you fill in: question, reference answer, candidate answer, then blank label columns)
        data/labeling/key.json    (id -> where each candidate came from + its pool entry; DO NOT OPEN until you have labelled all rows)
        data/labeling/LABELING_GUIDE.md
The sheet never contains judge scores, model names, or provenance. Rows are shuffled; sources are balanced so the sample
includes good, wrong and borderline candidates. Do not run the judge on these items before you finish labelling.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

GUIDE = """# Labeling guide (write your own notes in the `notes` column)

You see: a question, a reference answer written from the source documents, and a candidate answer. Judge the candidate only.

PASS (human_score 4 or 5)
  - the key facts agree with the reference answer, with no contradicted or invented claims
  - it answers what was asked (a refusal is a FAIL when the reference answer exists)
  - any format/length instruction in the question is respected
FAIL (human_score 1, 2 or 3)
  - a key fact is wrong, missing, or invented; or it does not answer the question; or it breaks an explicit constraint
human_score (optional but recommended): 5 fully correct | 4 correct, small omission | 3 borderline, lean FAIL | 2 mostly wrong | 1 wrong
Rules: label in one sitting if you can; do not look at key.json, any judge output, or the judge's earlier scores; if unsure, pick
the side your first reading suggested and write why in `notes`. Fill human_pass with exactly PASS or FAIL.
"""
COLS = ["id", "question", "reference_answer", "candidate_answer", "human_pass", "human_score", "notes"]


def _sig(item: dict) -> tuple:
    return (item["input"].strip().lower(), item["model_output"].strip().lower())


def select(pool: list[dict], n: int, seed: int, exclude: set) -> list[dict]:
    rng = random.Random(seed)
    by_src = defaultdict(list)
    for p in pool:
        if _sig(p) not in exclude and (p["input"], p["model_output"]) not in {(x["input"], x["model_output"]) for x in by_src[p["source"]]}:
            by_src[p["source"]].append(p)
    for v in by_src.values():
        rng.shuffle(v)
    chosen, srcs = [], sorted(by_src)
    while len(chosen) < n and any(by_src[s] for s in srcs):      # round-robin => balanced across sources
        for s in srcs:
            if by_src[s] and len(chosen) < n:
                chosen.append(by_src[s].pop())
    rng.shuffle(chosen)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/labeling/pool.json")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude", nargs="*", default=[], help="JSON files of already-used cases (keys: input, model_output)")
    ap.add_argument("--outdir", default="data/labeling")
    a = ap.parse_args()
    pool = json.load(open(a.pool, encoding="utf-8"))
    excl = {_sig(c) for f in a.exclude for c in json.load(open(f, encoding="utf-8"))}
    items = select(pool, a.n, a.seed, excl)
    if len(items) < a.n:
        print(f"WARNING: only {len(items)} candidates available after exclusions")
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    key = {}
    with open(out / "sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(COLS)
        for i, it in enumerate(items, 1):
            cid = f"L{i:02d}"
            w.writerow([cid, it["input"], it["expected_output"], it["model_output"], "", "", ""])
            key[cid] = it
    (out / "key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "LABELING_GUIDE.md").write_text(GUIDE, encoding="utf-8")
    print(f"{len(items)} rows -> {out / 'sheet.csv'}   (read {out / 'LABELING_GUIDE.md'}; leave key.json closed until done)")


if __name__ == "__main__":
    main()
