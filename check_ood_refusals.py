import json
from pathlib import Path

report_path = Path("results/rag_evaluation_report.json")
if not report_path.exists():
    print("Report not found yet.")
    exit(0)

data = json.loads(report_path.read_text(encoding="utf-8"))
cases = data.get("cases", [])
ood_passed = [c for c in cases if c.get("category") == "out_of_domain" and not c.get("no_context_triggered")]

print(f"--- Out-of-Domain Questions Passing MIN_SCORE Threshold ({len(ood_passed)} total) ---")
for c in ood_passed:
    print(f"[{c['id']}] Query: {c['query']}")
    print(f"  Refused Flag: {c['refused']}")
    print(f"  Model Answer: {c['model_answer']}")
    am = c.get("answer_metrics", {})
    print(f"  Judge Scores: Faithfulness={am.get('faithfulness')}, Correctness={am.get('correctness')}, Relevance={am.get('relevance')}")
    print()
