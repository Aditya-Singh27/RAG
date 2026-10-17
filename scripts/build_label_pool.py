"""Build a pool of candidate answers to label. Candidates come from the REAL generator (so they have realistic errors) plus
two mechanical degradations of one sample (cut it in half / drop a sentence) that create borderline cases.
Nothing here is labelled; labelling is done by a human in scripts/make_labeling_sheet.py's CSV.

  python scripts/build_label_pool.py --questions data/eval_questions.json --out data/labeling/pool.json
Generator comes from GENERATOR_PROVIDER / GENERATOR_MODEL / GENERATOR_BASE_URL / GENERATOR_API_KEY (same vars as .env.example).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def truncate_half(text: str) -> str:
    s = sentences(text)
    return " ".join(s[: max(1, len(s) // 2)]) if len(s) > 1 else text[: max(1, len(text) // 2)]


def drop_sentence(text: str) -> str | None:
    s = sentences(text)
    return " ".join(s[:1] + s[2:]) if len(s) >= 3 else None      # drop the 2nd sentence; None if too short to matter


def build_pool(questions: list[dict], llm, samples: int = 2, temperature: float = 0.9) -> list[dict]:
    pool = []
    for q in questions:
        if q["category"] not in ("in_domain", "cross_doc") or not q.get("gold_answer"):
            continue
        base = dict(input=q["query"], expected_output=q["gold_answer"],
                    reference_context=" ".join(q.get("evidence", [])) or None, question_id=q["id"])
        outs = []
        for i in range(samples):
            r = llm.generate(prompt=q["query"], system="Answer the question concisely.", temperature=temperature, max_tokens=300)
            outs.append(r.text.strip())
            pool.append({**base, "model_output": outs[-1], "source": f"generator_sample_{i}"})
        pool.append({**base, "model_output": truncate_half(outs[0]), "source": "degraded_truncate"})
        d = drop_sentence(outs[0])
        if d:
            pool.append({**base, "model_output": d, "source": "degraded_drop_sentence"})
    return [p for p in pool if p["model_output"]]


def main():
    from common.llm import LoggedLLM, make_llm
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/eval_questions.json")
    ap.add_argument("--out", default="data/labeling/pool.json")
    ap.add_argument("--samples", type=int, default=2)
    a = ap.parse_args()
    inner = make_llm(os.environ["GENERATOR_PROVIDER"], os.environ["GENERATOR_MODEL"],
                     os.environ.get("GENERATOR_BASE_URL", ""), os.environ.get("GENERATOR_API_KEY", ""))
    llm = LoggedLLM(inner, "logs/label_pool.jsonl", "label_pool_generator")
    pool = build_pool(json.load(open(a.questions, encoding="utf-8")), _Adapter(llm), a.samples)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(pool)} candidates -> {a.out}")


class _Adapter:                                    # LoggedLLM.generate wants a tag; keep build_pool provider-agnostic
    def __init__(self, llm): self.llm = llm
    def generate(self, prompt, system="", temperature=0.0, max_tokens=1024):
        return self.llm.generate(prompt, system, temperature, max_tokens, tag="label_pool")


if __name__ == "__main__":
    main()
