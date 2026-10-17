import csv, json, subprocess, sys
from pathlib import Path
import pytest
from common.llm import ScriptedLLM
from scripts import build_label_pool as bp, make_labeling_sheet as ms, import_labels as il

ROOT = Path(__file__).resolve().parent.parent


def _pool(n=50):
    srcs = ["generator_sample_0", "generator_sample_1", "degraded_truncate"]
    return [{"input": f"q{i}", "expected_output": f"gold {i}", "reference_context": "ctx", "model_output": f"answer {i}", "source": srcs[i % 3]} for i in range(n)]


def _run(tmp, *args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})


def test_degradations():
    assert bp.truncate_half("One. Two. Three. Four.") == "One. Two."
    assert bp.drop_sentence("One. Two. Three.") == "One. Three."
    assert bp.drop_sentence("Only one.") is None


def test_pool_builder_uses_generator_and_skips_ood():
    qs = [{"id": "a", "category": "in_domain", "query": "Q?", "gold_answer": "G", "evidence": ["e1", "e2"]},
          {"id": "o", "category": "out_of_domain", "query": "Z?", "gold_answer": "refusal"}]
    llm = ScriptedLLM(fn=lambda p, s: "First. Second. Third.")
    pool = bp.build_pool(qs, llm, samples=2)
    assert {p["source"] for p in pool} == {"generator_sample_0", "generator_sample_1", "degraded_truncate", "degraded_drop_sentence"}
    assert all(p["question_id"] == "a" and p["reference_context"] == "e1 e2" for p in pool) and len(llm.calls) == 2


def test_sheet_is_blind_balanced_deterministic_and_excludes(tmp_path):
    pool = tmp_path / "pool.json"; pool.write_text(json.dumps(_pool()))
    used = tmp_path / "used.json"; used.write_text(json.dumps([{"input": "q0", "model_output": "answer 0"}]))
    out = tmp_path / "o"
    r = _run(tmp_path, "scripts/make_labeling_sheet.py", "--pool", pool, "--n", 30, "--outdir", out, "--exclude", used)
    assert r.returncode == 0, r.stderr
    text = (out / "sheet.csv").read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == 30 and list(rows[0]) == ms.COLS
    for banned in ("generator_sample", "degraded", "source", "judge", "score:"):
        assert banned not in text
    assert all(r["human_pass"] == "" for r in rows) and "q0" not in {r["question"] for r in rows}
    key = json.loads((out / "key.json").read_text())
    from collections import Counter
    assert max(Counter(v["source"] for v in key.values()).values()) - min(Counter(v["source"] for v in key.values()).values()) <= 1
    out2 = tmp_path / "o2"; _run(tmp_path, "scripts/make_labeling_sheet.py", "--pool", pool, "--n", 30, "--outdir", out2, "--exclude", used)
    assert (out / "sheet.csv").read_bytes() == (out2 / "sheet.csv").read_bytes()


def _filled(tmp_path, edits):
    pool = tmp_path / "pool.json"; pool.write_text(json.dumps(_pool(12)))
    out = tmp_path / "o"; _run(tmp_path, "scripts/make_labeling_sheet.py", "--pool", pool, "--n", 6, "--outdir", out)
    rows = list(csv.DictReader(open(out / "sheet.csv", encoding="utf-8-sig")))
    for i, r in enumerate(rows):
        r.update(edits(i))
    with open(out / "sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=ms.COLS); w.writeheader(); w.writerows(rows)
    return out


def test_import_happy_path_marks_human_and_names_labeler(tmp_path):
    out = _filled(tmp_path, lambda i: {"human_pass": "pass" if i % 2 else "FAIL", "human_score": "5" if i % 2 else "2", "notes": "n"})
    cases, problems, warns = il.load_labels(str(out / "sheet.csv"), str(out / "key.json"), "A. Person")
    assert not problems and not warns and len(cases) == 6
    assert all(c["label_source"] == "human" and c["labeled_by"] == "A. Person" for c in cases)
    assert {c["gold_pass"] for c in cases} == {True, False} and "gold_score" in cases[0] and cases[0]["reference_context"] == "ctx"


def test_import_rejects_blank_and_invalid(tmp_path):
    out = _filled(tmp_path, lambda i: {"human_pass": "" if i == 0 else "maybe" if i == 1 else "PASS", "human_score": "9" if i == 2 else ""})
    _, problems, _ = il.load_labels(str(out / "sheet.csv"), str(out / "key.json"), "X")
    assert any("blank" in p for p in problems) and any("PASS or FAIL" in p for p in problems) and any("1-5" in p for p in problems)
    _, problems, _ = il.load_labels(str(out / "sheet.csv"), str(out / "key.json"), "X", allow_partial=True)
    assert not any("blank" in p for p in problems)


def test_import_warns_on_inconsistent_score_and_needs_labeler(tmp_path):
    out = _filled(tmp_path, lambda i: {"human_pass": "PASS", "human_score": "2" if i == 0 else "5"})
    _, problems, warns = il.load_labels(str(out / "sheet.csv"), str(out / "key.json"), "X")
    assert not problems and len(warns) == 1
    r = _run(tmp_path, "scripts/import_labels.py", "--sheet", out / "sheet.csv", "--key", out / "key.json", "--out", tmp_path / "x.json")
    assert r.returncode != 0 and "labeler" in r.stderr
