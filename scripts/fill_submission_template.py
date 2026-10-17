"""Fill submission_template.docx from the repo's result files.

Rules this script enforces (so the document cannot claim more than the repo shows):
  * every number is read from results/*.json - a missing key raises KeyError, there are NO default values;
  * things only the candidate can supply (name, time spent, screenshots, reflections, AI-usage disclosure)
    are left as visibly marked placeholders and are never generated;
  * design-decision boxes contain only facts that are in the code or in the result files, plus a marked
    slot for the candidate's own reasoning.

Run:  python scripts/fill_submission_template.py   ->  submission_template_completed.docx
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

TODO = "[ TO BE COMPLETED BY CANDIDATE"
TODO_COLOR = (180, 35, 24)


def set_text(cell, text, bold=False, size=9.5, color=(15, 23, 42)):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(text)
    r.bold, r.font.name, r.font.size = bold, "Arial", Pt(size)
    r.font.color.rgb = RGBColor(*color)


def todo(cell, what, hints=""):
    """Visible placeholder. `hints` lists verified facts the candidate may draw on - not an answer."""
    msg = f"{TODO} - {what} ]"
    if hints:
        msg += "\nFacts available in the repo you can draw on: " + hints
    set_text(cell, msg, size=9, color=TODO_COLOR)


def set_image(cell, path, width=6.2):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))


def replace_para(doc, startswith, new_text):
    for p in doc.paragraphs:
        if p.text.strip().startswith(startswith):
            for r in p.runs[1:]:
                r._element.getparent().remove(r._element)
            if p.runs:
                p.runs[0].text = new_text
            else:
                p.add_run(new_text)
            return
    raise KeyError(f"paragraph not found: {startswith}")


def ci(x, nd=3):
    return f"[{x[0]:.{nd}f}, {x[1]:.{nd}f}]"


def env_names(path=".env.example"):
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]+)=", line.strip())
        if m:
            names.append(m.group(1))
    return names


def main():
    rag = json.load(open("results/rag_evaluation_report.json", encoding="utf-8"))["summary"]
    judge = json.load(open("results/judge_evaluation_report.json", encoding="utf-8"))
    cost = json.load(open("results/cost_comparison.json", encoding="utf-8"))
    lat = json.load(open("results/latency_benchmark_real.json", encoding="utf-8"))
    syn = json.load(open("results/latency_benchmark_synth.json", encoding="utf-8"))
    base = json.load(open("results/retrieval_baselines.json", encoding="utf-8"))  # noqa: F841 (kept for provenance)

    d = docx.Document("submission_template.docx")
    meta, ret = rag["meta"], rag["retrieval_answerable_only"]
    aq, ref = rag["answer_quality_answerable_only"], rag["refusal_behaviour"]
    cfg = judge["pipeline_configuration"]

    # ---------------- candidate details ----------------
    t0 = d.tables[0]
    todo(t0.rows[1].cells[1], "your full name")
    todo(t0.rows[2].cells[1], "your email")
    todo(t0.rows[3].cells[1], "submission date")
    set_text(t0.rows[4].cells[1], "https://github.com/Aditya-Singh27/RAG")
    todo(t0.rows[5].cells[1], "total time you actually spent")
    set_text(t0.rows[6].cells[1], "LanceDB (embedded, on-disk)")
    set_text(t0.rows[7].cells[1],
             f"judge: {cfg['judge_model']} ({cfg['judge_provider']}) | generator: {cfg['generator_model']} "
             f"({cfg['generator_provider']}, local Ollama)")

    hw = lat["hardware"]
    names = env_names()
    prereq = (f"Prerequisites: Python {hw['python']} on {hw['platform']} ({hw['cpu_count']} CPUs) - the machine the reported "
              "latencies were measured on. Ollama running locally with llama3.2:latest pulled. A Gemini API key.")
    envs = "Environment variables (names only): " + ", ".join(names)
    store = ("Vector store chosen + one-line why: LanceDB - embedded, on-disk columnar store with upsert and cosine search, "
             "so no always-on database server is needed.")
    for pre, text in [("Prerequisites", prereq), ("Environment variables", envs)]:
        for p in d.paragraphs:  # appears twice (P1 and P2): fill both
            if p.text.strip().startswith(pre):
                for r in p.runs[1:]:
                    r._element.getparent().remove(r._element)
                p.runs[0].text = text
    replace_para(d, "Vector store chosen", store)
    replace_para(d, "k used", f"k used: k = {rag['k']}")
    replace_para(d, "Judge & generator configured independently",
                 "Judge & generator configured independently? Yes - separate provider/model/base-URL/key settings: "
                 "JUDGE_PROVIDER, JUDGE_MODEL vs GENERATOR_PROVIDER, GENERATOR_MODEL, GENERATOR_BASE_URL, GENERATOR_API_KEY "
                 "(see .env.example and common/config.py).")
    replace_para(d, "Judging mode used",
                 "Judging mode used: pointwise (rubric, per-criterion scores) and pairwise A-vs-B (each pair run in both orders); "
                 "reference-based and reference-free variants for the sycophancy probe.")
    ab = judge["ab_comparison_verdict"]
    replace_para(d, "Declared winner",
                 f"Declared winner + one-line justification: {ab['declared_winner']}. Config B won {ab['wins_b']} of "
                 f"{ab['total_cases']} cases vs {ab['wins_a']} for Config A, but the exact sign test gives p = {ab['sign_test_p']} "
                 f"(alpha = {ab['alpha']}), so the pipeline does not declare a winner at this sample size.")

    # ---------------- diagrams ----------------
    set_image(d.tables[1].rows[0].cells[0], "results/figures/problem1_architecture.png")
    set_image(d.tables[21].rows[0].cells[0], "results/figures/problem2_architecture.png")

    # ---------------- setup commands ----------------
    set_text(d.tables[2].rows[0].cells[0],
             "git clone https://github.com/Aditya-Singh27/RAG.git\ncd RAG\npython -m venv .venv\n"
             ".venv\\Scripts\\activate        # Linux/macOS: source .venv/bin/activate\n"
             "pip install -r requirements.txt\nollama pull llama3.2:latest\n"
             "copy .env.example .env         # then set GEMINI_API_KEY (and model ids) in .env", size=8.5)
    set_text(d.tables[3].rows[0].cells[0],
             "python -m rag.ingest data/corpus\n# running it a second time re-embeds nothing (doc_hash unchanged)", size=8.5)
    set_text(d.tables[4].rows[0].cells[0],
             '# CLI (query is a positional argument; see rag/service.py)\n'
             'python -m rag.service "What is IVF-PQ in LanceDB?" --k 5\n\n'
             '# or HTTP (rag/api.py exposes POST /query, GET /health, GET /stats, POST /ingest)\n'
             'uvicorn rag.api:app --port 8000\n'
             'curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\\"query\\": \\"What is IVF-PQ in LanceDB?\\", \\"k\\": 5}"\n'
             '# check the request body field names against rag/api.py before submitting', size=8.5)
    set_text(d.tables[22].rows[0].cells[0], "# same install as Problem 1 (see above)\nollama pull llama3.2:latest", size=8.5)
    set_text(d.tables[23].rows[0].cells[0],
             "python -m judge.run\n# writes results/judge_evaluation_report.json; every LLM call is appended to LLM_LOG_PATH (JSONL)", size=8.5)
    set_text(d.tables[24].rows[0].cells[0],
             "python scripts/generate_real_ab.py     # builds data/judge_suites/ab_comparison_cases.json from two prompt configs\n"
             "python -m judge.run                    # runs the A/B comparison with the sign-test gate", size=8.5)

    # ---------------- 1.3.1 retrieval ----------------
    t5 = d.tables[5]
    n = ret["hit_rate@5"]["n"]
    set_text(t5.rows[1].cells[1], f"{ret['hit_rate@5']['mean']:.3f}", bold=True)
    set_text(t5.rows[1].cells[2], f"Hit@5 over {n} answerable questions; Evidence recall@5 {ret['evidence_recall@5']['mean']:.3f}. "
             "Relevant chunk = stored chunk containing a verbatim gold evidence quote. Corpus is only "
             f"{meta['corpus_chunks']} chunks (random top-5 hits ~17.9%), so this benchmark is easy.")
    set_text(t5.rows[2].cells[1], f"{ret['mrr']['mean']:.4f}", bold=True)
    set_text(t5.rows[2].cells[2], f"Bootstrap 95% CI {ci(ret['mrr']['ci95'])}.")
    set_text(t5.rows[3].cells[1], f"{ret['ndcg@5']['mean']:.4f}", bold=True)
    set_text(t5.rows[3].cells[2], f"Bootstrap 95% CI {ci(ret['ndcg@5']['ci95'])}; ideal DCG computed from the relevant chunks in the store.")
    set_text(t5.rows[4].cells[1], f"{ret['context_precision']['mean']:.3f} (share)  /  {ret['context_precision_rank_aware']['mean']:.3f} (rank-aware)", bold=True)
    set_text(t5.rows[4].cells[2], "Share = fraction of the 5 retrieved chunks that are relevant (capped by how many relevant chunks exist). "
             "Rank-aware = average precision over relevant ranks.")

    # ---------------- 1.3.2 answer quality ----------------
    t6 = d.tables[6]
    judge_txt = f"LLM judge: {meta['judge']} ({meta['judge_provider']}), different family from generator {meta['generator']}"
    f_, rel = aq["faithfulness"], aq["relevance"]
    set_text(t6.rows[1].cells[1], f"{f_['mean']:.2f} / 5", bold=True)
    set_text(t6.rows[1].cells[2], judge_txt)
    set_text(t6.rows[1].cells[3], f"{aq['evaluated_questions']} answerable questions only; CI {ci(f_['ci95'], 2)}; histogram {f_['spread']['histogram']}. "
             "Judge not yet validated against independent human labels for this task.")
    set_text(t6.rows[2].cells[1], f"{rel['mean']:.2f} / 5", bold=True)
    set_text(t6.rows[2].cells[2], judge_txt)
    set_text(t6.rows[2].cells[3], f"All {rel['n']} scores equal 5 - this metric does not discriminate here.")
    set_text(t6.rows[3].cells[1], f"{aq['exact_match_answerable']['mean']:.3f}", bold=True)
    set_text(t6.rows[3].cells[2], "gold answers")
    set_text(t6.rows[3].cells[3], aq["em_note"])
    set_text(t6.rows[4].cells[1], f"{aq['token_f1_answerable']['mean']:.3f}", bold=True)
    set_text(t6.rows[4].cells[2], "gold answers")
    set_text(t6.rows[4].cells[3], f"Token F1 vs gold answer, bootstrap CI {ci(aq['token_f1_answerable']['ci95'])}; cited free-form answers are longer than gold, which lowers F1.")

    # ---------------- 1.3.3 cost ----------------
    t7 = d.tables[7]
    for i, row in enumerate(cost["rows"], start=1):
        set_text(t7.rows[i].cells[1], f"${row['self_hosted_ec2_usd']:.2f} (always-on instance + S3)\n${row['self_hosted_s3_only_usd']:.2f} (no instance: S3 only)", bold=True)
        set_text(t7.rows[i].cells[2], f"${row['pinecone_pod_usd']:.2f} (pods)\n${row['pinecone_serverless_usd']:.2f} (serverless)")
        set_text(t7.rows[i].cells[3], f"{row['ratio_vs_pod']}x cheaper than pods; vs Pinecone Serverless the break-even is "
                 f"{row['break_even_queries_month']:,} queries/month (below that, serverless is cheaper than the fixed instance). Estimate, not measured.")
    a = cost["assumptions"]
    sh, mg = a["self_hosted"], a["managed"]
    text = (
        "Pricing assumptions (estimates from the cited vendor pages, not measurements; verify before relying on them):\n"
        f"- Queries/month: {a['queries_per_month']['value']:,}.\n"
        f"- Bytes per vector: {a['bytes_per_vector']['value']} - {a['bytes_per_vector']['note']}.\n"
        f"- Self-hosted instance ${sh['instance_monthly_usd']['value']}/mo ({sh['instance_monthly_usd']['note']}); S3 storage "
        f"${sh['storage_usd_per_gb_month']['value']}/GB-month; S3 GET ${sh['get_usd_per_1000_requests']['value']} per 1,000, "
        f"{sh['gets_per_query']['value']} GETs/query assumed.\n"
        + "\n".join(f"- Pinecone pods <= {t['max_vectors']:,} vectors: ${t['monthly_usd']}/mo ({t['note']})" for t in mg["tiers"])
        + f"\n- Pinecone Serverless: storage ${mg['serverless']['storage_usd_per_gb_month']['value']}/GB-month, reads "
        f"${mg['serverless']['usd_per_million_read_units']['value']} per million read units "
        f"(read units per query: see assumptions file).\n"
        "Sources and access dates are in data/pricing_assumptions.json. Note: the latency numbers below were measured on a local "
        "disk, while the S3-only cost row assumes object-storage reads; those two have not been measured together."
    )
    set_text(d.tables[8].rows[0].cells[0], text, size=8.5)

    # ---------------- 1.3.4 latency ----------------
    t9 = d.tables[9]
    vs, emb = lat["vector_search_only"], lat["query_embedding_api"]
    set_text(t9.rows[1].cells[1], f"{vs['p50_ms']:.1f} (search only)\n{emb['p50_ms']:.0f} (query embedding API)", bold=True)
    set_text(t9.rows[1].cells[2], f"k={lat['k']}, {lat['vectors_in_store']} vectors in store (flat search), {lat['queries']} queries x {lat['repeats']} repeats, "
             f"{hw['platform']}, {hw['cpu_count']} CPUs.")
    set_text(t9.rows[2].cells[1], f"{vs['p95_ms']:.1f} (search only)\n{emb['p95_ms']:.0f} (query embedding API)", bold=True)
    r100 = syn["runs"][0]
    set_text(t9.rows[2].cells[2], f"Same run. Scale: only {r100['vectors']:,} synthetic vectors were benchmarked (flat p50 {r100['flat_exact']['p50_ms']:.0f} ms; "
             f"IVF-PQ refine_factor=20 p50 {r100['ivf_pq']['settings']['refine_factor_20']['p50_ms']:.1f} ms at recall@5 "
             f"{r100['ivf_pq']['settings']['refine_factor_20']['recall@5_vs_exact']:.2f}; without refine recall was "
             f"{r100['ivf_pq']['settings']['no_refine']['recall@5_vs_exact']:.2f}). 1M and 10M were NOT measured.")
    L = rag["latency_ms"]
    set_text(t9.rows[3].cells[1], f"{L['total_p95']:.0f}", bold=True)
    set_text(t9.rows[3].cells[2], f"End-to-end over {L['n']} questions (p50 {L['total_p50']:.0f} ms); dominated by generation (p95 {L['generation_p95']:.0f} ms). "
             f"Slowest single requests: {', '.join(str(s['id']) + ' ' + format(s['total_ms'], ',.0f') + ' ms' for s in L['slowest'])}; "
             "p95 of this few requests is close to the maximum.")

    # ---------------- design decisions: candidate's own reasoning ----------------
    hints = {
        10: ("why LanceDB over the other stores", "embedded/no server; cost table (results/cost_comparison.json); 28-row store uses flat search, IVF-PQ only benchmarked on synthetic vectors"),
        11: ("chunking size/overlap and what degraded elsewhere", "default 800/100 (rag/chunking.py). The repo contains NO chunk-size sweep - if you did not run one, say so rather than describe one"),
        12: ("embedding model and dimension trade-off", "gemini-embedding-001 at 768 dims, query/document task types, vectors re-normalised. No comparison against another embedding model exists in the repo"),
        13: ("how you handle no relevant context", f"stage 1 score gate: {ref['stage1_threshold_refused']} out-of-domain refused; stage 2: {ref['stage2_prompt_grounding_refused']} of the rest declined in text; false refusals {ref['false_refusals_on_answerable']}; threshold set on 7 dev negatives, evaluated on 8"),
        14: ("how re-ingest avoids duplicates", "chunk_id = hash of doc_id|index|text; doc_hash over text + chunk params + embed model; upsert then delete stale rows (rag/store.py, rag/ingest.py); tests/test_ingest.py"),
        15: ("trade-offs accepted and when you would switch to a managed DB", "cost table; no 1M/10M latency measurement; single-node embedded store"),
    }
    for i, (what, h) in hints.items():
        todo(d.tables[i].rows[0].cells[0], f"your reasoning: {what}", h)

    # ---------------- evidence & reflection (candidate only) ----------------
    for i, what in [(16, "paste a REAL screenshot of one query: run the service/CLI and capture latency, chunk count, token usage"),
                    (17, "paste a REAL screenshot: run `python -m rag.ingest data/corpus` twice; the second run should show nothing re-embedded"),
                    (34, "paste a REAL screenshot of one raw prompt + raw response from logs/llm_calls.jsonl"),
                    (35, "paste a REAL screenshot of the position-bias output from python -m judge.run")]:
        todo(d.tables[i].rows[0].cells[0], what)
    for i, what in [(18, "walk through one query in your own words, without AI help"),
                    (19, "one thing that broke and exactly how you debugged it - your own account"),
                    (20, "which AI tools you used and for what; what is entirely your own - state this truthfully"),
                    (36, "explain your position-bias check in your own words, without AI help"),
                    (37, "your own read of the adversarial probe result"),
                    (38, "which AI tools you used and for what; what is entirely your own - state this truthfully")]:
        todo(d.tables[i].rows[0].cells[0], what)
    pr = judge["sycophancy_adversarial_evaluation"]["with_reference"]
    set_text(d.tables[37].rows[0].cells[0],
             f"{TODO} - your own read of the adversarial probe result ]\nMeasured: with a reference answer the judge was fooled in "
             f"{pr['fooled_count']}/{pr['total_adversarial_cases']} probes (95% CI {ci(pr['fool_rate_ci95'])}); without a reference "
             f"{judge['sycophancy_adversarial_evaluation']['without_reference']['fooled_count']}/"
             f"{judge['sycophancy_adversarial_evaluation']['without_reference']['total_adversarial_cases']}. "
             "With n = 8 the upper bound of the interval is still about one third.", size=9, color=TODO_COLOR)

    # ---------------- P2 rubric: read from code (judge/rubrics.py STANDARD_RUBRIC) ----------------
    from judge.rubrics import STANDARD_RUBRIC
    t25 = d.tables[25]
    keys = list(STANDARD_RUBRIC)
    while len(t25.rows) - 1 < len(keys):
        t25.add_row()
    for row, k in zip(t25.rows[1:], keys):
        c = STANDARD_RUBRIC[k]
        set_text(row.cells[0], c.name.replace("_", " "), bold=True)
        set_text(row.cells[1], f"{c.description}\n1: {c.anchor_1}\n3: {c.anchor_3}\n5: {c.anchor_5}", size=8.5)
        set_text(row.cells[2], f"{c.weight:.2f}")

    # ---------------- P2 bias table ----------------
    t26 = d.tables[26]
    pb, ip = judge["position_bias_evaluation"]["details"], judge["identical_pair_probe"]
    set_text(t26.rows[1].cells[1], "Every pair is judged in both orders; consensus winner; hard flips (winner changes) and soft flips recorded. Identical-pair probe checks slot preference.")
    set_text(t26.rows[1].cells[2], f"Single-order hard flips {pb['hard_flips']}/{pb['n_pairs']} (95% CI {ci(pb['single_order_flip_ci95'])}). No flips were observed, so there is no before->after change to report. "
             f"Identical-pair probe: first slot won {ip['first_slot_wins']}, second {ip['second_slot_wins']}, ties {ip['ties']} of {ip['n_judgements']} (sign test p = {ip['sign_test_p']}; underpowered).")
    vb = judge["verbosity_bias_evaluation"]
    set_text(t26.rows[2].cells[1], "Paired pointwise probe: same content, terse vs padded; baseline rubric vs calibrated rubric.")
    set_text(t26.rows[2].cells[2], f"Verbose minus terse score: {vb['before_mitigation']['mean_diff_verbose_minus_terse']} (CI {ci(vb['before_mitigation']['ci95'])}) -> "
             f"{vb['after_mitigation']['mean_diff_verbose_minus_terse']} (CI {ci(vb['after_mitigation']['ci95'])}), n={vb['after_mitigation']['n']}. Both penalise the verbose answer; "
             "the mitigation did not remove the effect, it changed its size.")
    set_text(t26.rows[3].cells[1], "Judge and generator are different model families, configured independently.")
    set_text(t26.rows[3].cells[2], f"No before->after measurement. Problem 1 answers (generator {meta['generator']}) were judged by "
             f"{meta['judge']}; no same-family judge run is reported here, so the size of any self-enhancement effect is unknown.")
    sy = judge["sycophancy_adversarial_evaluation"]
    set_text(t26.rows[4].cells[1], "Per-criterion rationale before score; probe with a confidently wrong answer, with and without a reference answer.")
    set_text(t26.rows[4].cells[2], f"Fooled: with reference {sy['with_reference']['fooled_count']}/{sy['with_reference']['total_adversarial_cases']}, "
             f"without {sy['without_reference']['fooled_count']}/{sy['without_reference']['total_adversarial_cases']} (CI {ci(sy['with_reference']['fool_rate_ci95'])}). No before->after.")
    sc = judge["score_clustering_evaluation"]
    set_text(t26.rows[5].cells[1], "Few-shot score anchors in the judging prompt.")
    set_text(t26.rows[5].cells[2], f"Std {sc['before_mitigation']['std']} -> {sc['after_mitigation']['std']}; distinct values {sc['before_mitigation']['distinct_values']} -> "
             f"{sc['after_mitigation']['distinct_values']}; share at extremes {sc['before_mitigation']['frac_at_extremes']} -> {sc['after_mitigation']['frac_at_extremes']} (n={sc['after_mitigation']['n']}). "
             "Spread across the scale improved; the standard deviation went down, so it is not 'more spread' in that sense.")

    # ---------------- P2 validation ----------------
    t27, ha, tr = d.tables[27], judge["human_agreement_validation"], judge["test_retest_consistency"]
    src = ha["label_sources"]
    set_text(t27.rows[1].cells[1], f"{ha['agreement_rate']*100:.0f}% ({ha['confusion_matrix']['true_positive']+ha['confusion_matrix']['true_negative']}/{ha['total_cases']})", bold=True)
    set_text(t27.rows[1].cells[2], f"n = {ha['total_cases']}, Wilson 95% CI {ci(ha['agreement_ci95_wilson'])}. The report records label_sources={src}, but data/judge_suites/human_labeled_cases.json is tagged 'author' - "
             "these are author-written labels, not independent human labels, so re-run judge.run to refresh the report. "
             "Replace with agreement on the 30-row blind sheet once you have labeled it and re-run.")
    set_text(t27.rows[2].cells[1], f"{ha['cohens_kappa']}", bold=True)
    set_text(t27.rows[2].cells[2], f"Bootstrap CI {ha['kappa_ci95_bootstrap']}; mean absolute score error {ha['mean_absolute_score_error']}. Kappa of 1.0 on 10 cases is weak evidence.")
    set_text(t27.rows[3].cells[1], f"{tr['flip_count']}/{tr['total_cases']} verdicts flipped", bold=True)
    set_text(t27.rows[3].cells[2], f"{tr['runs_per_case']} runs per case at temperature 0.0; {tr['judge_calls_made']} real judge calls for {tr['judgements_requested']} judgements "
             f"(cache_suspected={tr['cache_suspected']}). Scores still varied: mean range {tr['mean_score_range']}, max {tr['max_score_range']}.")
    set_text(t27.rows[4].cells[1], f"Fooled {sy['with_reference']['fooled_count']}/{sy['with_reference']['total_adversarial_cases']} (with reference); {sy['without_reference']['fooled_count']}/{sy['without_reference']['total_adversarial_cases']} (without)", bold=True)
    set_text(t27.rows[4].cells[2], f"Confidently-wrong answers vs correct ones, n=8; CI upper bound {sy['with_reference']['fool_rate_ci95'][1]}.")

    # ---------------- P2 A/B ----------------
    t28 = d.tables[28]
    set_text(t28.rows[1].cells[0], ab["label_a"])
    set_text(t28.rows[1].cells[1], "not computed (pairwise run)")
    set_text(t28.rows[1].cells[2], f"{ab['mean_score_a']}")
    set_text(t28.rows[1].cells[3], f"{ab['win_rate_a']*100:.0f}% ({ab['wins_a']}/{ab['total_cases']})")
    set_text(t28.rows[2].cells[0], ab["label_b"])
    set_text(t28.rows[2].cells[1], "not computed (pairwise run)")
    set_text(t28.rows[2].cells[2], f"{ab['mean_score_b']}  (diff {ab['mean_score_diff_b_minus_a']}, CI {ci(ab['score_diff_ci95'], 2)})")
    set_text(t28.rows[2].cells[3], f"{ab['win_rate_b']*100:.0f}% ({ab['wins_b']}/{ab['total_cases']})")

    # ---------------- P2 design decisions ----------------
    h2 = {
        29: ("which judging mode and why", "pointwise + pairwise both implemented; pairwise runs both orders (judge/bias.py)"),
        30: ("how you parse structured verdicts and recover from malformed JSON", f"see judge/pipeline.py; parse failures counted, never defaulted; report shows judge_calls={cfg['judge_calls']}"),
        31: ("why this judge family vs the generator family", f"judge {cfg['judge_model']}; generator {cfg['generator_model']}; Problem 1 answer judge is {meta['judge']} vs generator {meta['generator']}"),
        32: ("which bias worried you most and how confident you are", "position: 0 flips observed in 10 pairs (CI up to 0.278), identical-pair probe underpowered; verbosity: both rubrics still penalise verbose answers"),
        33: ("would you let this judge gate a release, and where a human stays in the loop", "small samples (n=8-10 per probe); no independent human agreement yet; sign-test gate returns INCONCLUSIVE at n=10"),
    }
    for i, (what, h) in h2.items():
        todo(d.tables[i].rows[0].cells[0], f"your reasoning: {what}", h)

    d.save("submission_template_completed.docx")
    left = sum(1 for t in d.tables for r in t.rows for c in r.cells if c.text.startswith(TODO))
    print(f"saved submission_template_completed.docx; {left} placeholder cells left for the candidate")


if __name__ == "__main__":
    main()
