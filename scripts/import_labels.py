"""Turn the filled-in sheet into labelled cases for the judge validator.

  python scripts/import_labels.py --labeler "Your Name" --out data/judge_suites/human_labeled_cases_v2.json

`--labeler` is required and is stored on every case: `label_source: "human"` is only written when a named person owns the labels.
Rejects blanks, anything other than PASS/FAIL, scores outside 1-5. Warns when PASS/FAIL disagrees with the score (PASS needs >= 4).
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def load_labels(sheet: str, key: str, labeler: str, allow_partial: bool = False):
    keymap = json.load(open(key, encoding="utf-8"))
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    problems, warns, cases = [], [], []
    for r in rows:
        cid = r["id"].strip()
        verdict = (r.get("human_pass") or "").strip().upper()
        score_raw = (r.get("human_score") or "").strip()
        if verdict == "":
            if not allow_partial:
                problems.append(f"{cid}: human_pass is blank")
            continue
        if verdict not in ("PASS", "FAIL"):
            problems.append(f"{cid}: human_pass must be PASS or FAIL, got {verdict!r}")
            continue
        score = None
        if score_raw:
            try:
                score = float(score_raw)
                if not 1 <= score <= 5:
                    raise ValueError
            except ValueError:
                problems.append(f"{cid}: human_score must be a number 1-5, got {score_raw!r}")
                continue
            if (verdict == "PASS") != (score >= 4):
                warns.append(f"{cid}: {verdict} with score {score:g} (PASS expects >= 4) - check you meant this")
        src = keymap[cid]
        c = {"id": cid, "input": r["question"], "model_output": r["candidate_answer"], "expected_output": r["reference_answer"],
             "reference_context": src.get("reference_context"), "gold_pass": verdict == "PASS",
             "label_source": "human", "labeled_by": labeler, "label_notes": (r.get("notes") or "").strip()}
        if score is not None:
            c["gold_score"] = score
        cases.append(c)
    return cases, problems, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="data/labeling/sheet.csv")
    ap.add_argument("--key", default="data/labeling/key.json")
    ap.add_argument("--labeler", required=True, help="your name; you are attesting you labelled these yourself, blind")
    ap.add_argument("--out", default="data/judge_suites/human_labeled_cases_v2.json")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    cases, problems, warns = load_labels(a.sheet, a.key, a.labeler, a.allow_partial)
    if problems:
        sys.exit("NOT written:\n  " + "\n  ".join(problems))
    for w in warns:
        print("warning:", w)
    p = Counter(c["gold_pass"] for c in cases)
    if len(p) < 2:
        print("warning: all labels identical -> Cohen's kappa will be undefined; the set needs both PASS and FAIL")
    Path(a.out).write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(cases)} labelled cases -> {a.out}  (PASS={p[True]}, FAIL={p[False]}, borderline score=3: {sum(c.get('gold_score') == 3 for c in cases)})")


if __name__ == "__main__":
    main()
