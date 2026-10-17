"""Bias detection, mitigation and measurement (Problem 2).

Changes vs the first version (see tests/test_bias.py):
  * flips are split into hard (A vs B) and soft (one order TIE); both are reported
  * identical-pair probe measures raw slot preference with zero content confound
  * every probe returns counts + intervals instead of a hard-coded "effective: True"
  * score-clustering and self-preference are now actually measured
  * optional `baseline_pipeline` argument gives the BEFORE arm for before->after tables
"""
from __future__ import annotations
from typing import Any

from judge.models import (PairwiseVerdict, SymmetricalPairwiseResult,  # noqa: F401
                          PositionBiasReport, TestCase)
from judge.pipeline import JudgePipeline
from judge.stats import (paired_bootstrap_diff, score_spread, sign_test_p, wilson_ci)


def _tc(case_id, user_input, output, expected=None, ctx=None) -> TestCase:
    return TestCase(id=case_id, input=user_input, model_output=output,
                    expected_output=expected, reference_context=ctx)


class BiasMitigationHarness:
    def __init__(self, pipeline: JudgePipeline):
        self.pipeline = pipeline

    # ------------------------------------------------------------------ position
    def evaluate_symmetrical_pairwise(self, case_id, user_input, output_a, output_b,
                                      expected_output=None, reference_context=None,
                                      pipeline: JudgePipeline | None = None) -> SymmetricalPairwiseResult:
        """Judge the pair in both presentation orders and combine.
        Winners are reported in TRUE labels (the pipeline un-swaps the reverse order)."""
        p = pipeline or self.pipeline
        kw = dict(case_id=case_id, user_input=user_input, output_a=output_a, output_b=output_b,
                  expected_output=expected_output, reference_context=reference_context)
        fwd = p.evaluate_pairwise(order="A_then_B", **kw)
        rev = p.evaluate_pairwise(order="B_then_A", **kw)

        fw, rw = fwd.winner, rev.winner
        flipped = fw != rw                       # includes TIE-vs-decisive ("soft") disagreements
        ma = round((fwd.score_a + rev.score_a) / 2.0, 2)
        mb = round((fwd.score_b + rev.score_b) / 2.0, 2)
        if fw == rw:
            consensus = fw
        elif ma > mb + 0.2:
            consensus = "A"
        elif mb > ma + 0.2:
            consensus = "B"
        else:
            consensus = "TIE"
        return SymmetricalPairwiseResult(
            case_id=case_id, forward_winner=fw, reverse_winner=rw, consensus_winner=consensus,
            flipped=flipped, mean_score_a=ma, mean_score_b=mb,
            forward_rationale=fwd.rationale, reverse_rationale=rev.rationale)

    def measure_position_bias(self, test_pairs: list[dict[str, Any]]):
        """Returns (report, results). `first_presented_win_rate` is computed over decisive
        verdicts only; with no flips it is 0.5 BY CONSTRUCTION, so it is not evidence of fairness."""
        results = [self.evaluate_symmetrical_pairwise(
            p["id"], p["input"], p["output_a"], p["output_b"],
            p.get("expected_output"), p.get("reference_context")) for p in test_pairs]
        first_wins = decisive = 0
        for r in results:
            if r.forward_winner in ("A", "B"):
                decisive += 1
                first_wins += r.forward_winner == "A"
            if r.reverse_winner in ("A", "B"):
                decisive += 1
                first_wins += r.reverse_winner == "B"
        n = len(results)
        flips = sum(r.flipped for r in results)
        fwr = round(first_wins / decisive, 4) if decisive else 0.0
        report = PositionBiasReport(
            total_pairs=n, agreement_count=n - flips, flip_count=flips,
            flip_rate=round(flips / n, 4) if n else 0.0,
            a_first_win_rate=fwr, b_first_win_rate=round(1 - fwr, 4) if decisive else 0.0,
            tie_rate=round(sum(r.consensus_winner == "TIE" for r in results) / n, 4) if n else 0.0)
        return report, results

    def position_bias_details(self, results: list[SymmetricalPairwiseResult],
                              test_pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """BEFORE vs AFTER. 'Before' = a judge that sees one fixed order; 'after' = both orders combined.
        After-mitigation flip rate is 0 by definition, so the informative numbers are:
        how often the single-order judge disagreed with itself, and (if pairs carry
        `gold_winner`) accuracy of forward-only / reverse-only / consensus."""
        n = len(results)
        hard = sum({r.forward_winner, r.reverse_winner} == {"A", "B"} for r in results)
        soft = sum(r.flipped for r in results) - hard
        out: dict[str, Any] = {
            "n_pairs": n, "hard_flips": hard, "soft_flips": soft,
            "single_order_flip_rate": round((hard + soft) / n, 4) if n else 0.0,
            "single_order_flip_ci95": [round(x, 3) for x in wilson_ci(hard + soft, n)],
            "consensus_ties_or_escalations": sum(r.consensus_winner == "TIE" for r in results)}
        gold = {p["id"]: p.get("gold_winner") for p in test_pairs}
        scored = [r for r in results if gold.get(r.case_id) in ("A", "B")]
        if scored:
            acc = lambda f: round(sum(f(r) == gold[r.case_id] for r in scored) / len(scored), 4)
            out["accuracy_vs_gold"] = {"n": len(scored), "forward_only": acc(lambda r: r.forward_winner),
                                       "reverse_only": acc(lambda r: r.reverse_winner),
                                       "consensus": acc(lambda r: r.consensus_winner)}
        return out

    def measure_identical_pair_bias(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Same text in both slots. An unbiased judge calls TIE / equal scores. Any systematic
        preference for the first (or second) slot is pure position bias: no content confound,
        works even when every real pair is an easy blowout. Cases: {id, input, output, expected_output?}."""
        first = second = ties = 0
        gaps = []
        for c in cases:
            kw = dict(case_id=c["id"], user_input=c["input"], output_a=c["output"], output_b=c["output"],
                      expected_output=c.get("expected_output"), reference_context=c.get("reference_context"))
            f = self.pipeline.evaluate_pairwise(order="A_then_B", **kw)   # slot 1 = A
            r = self.pipeline.evaluate_pairwise(order="B_then_A", **kw)   # slot 1 = B
            for verdict, first_label in ((f, "A"), (r, "B")):
                if verdict.winner == "TIE":
                    ties += 1
                elif verdict.winner == first_label:
                    first += 1
                else:
                    second += 1
            gaps.append((f.score_a - f.score_b) - (r.score_a - r.score_b))
        n = 2 * len(cases)
        return {"n_judgements": n, "first_slot_wins": first, "second_slot_wins": second, "ties": ties,
                "first_slot_win_rate_among_decisive": round(first / (first + second), 4) if first + second else None,
                "sign_test_p": round(sign_test_p(first, second), 4)}

    # ----------------------------------------------------------------- verbosity
    def measure_verbosity_bias(self, cases: list[dict[str, Any]], baseline_pipeline=None) -> dict[str, Any]:
        """Cases: {id, input, terse_output, verbose_output, expected_output?}. verbose_output must contain the
        SAME facts as terse_output plus filler (otherwise this tests correctness, not length).
        Pointwise paired difference (verbose - terse) with bootstrap CI; ideal is ~0.
        Pass `baseline_pipeline` (e.g. rubric without the length criterion) for the BEFORE arm."""
        def arm(pipe):
            terse, verb = [], []
            for c in cases:
                t = pipe.evaluate_pointwise(_tc(c["id"] + "_t", c["input"], c["terse_output"], c.get("expected_output"), c.get("reference_context")))
                v = pipe.evaluate_pointwise(_tc(c["id"] + "_v", c["input"], c["verbose_output"], c.get("expected_output"), c.get("reference_context")))
                terse.append(t.overall_score)
                verb.append(v.overall_score)
            mean, lo, hi = paired_bootstrap_diff(terse, verb)
            verdict = ("favours verbose" if lo > 0 else "penalises verbose" if hi < 0
                       else "no detectable effect at this n")
            return {"n": len(cases), "mean_diff_verbose_minus_terse": round(mean, 3),
                    "ci95": [round(lo, 3), round(hi, 3)], "verdict": verdict}
        out = {"after_mitigation": arm(self.pipeline)}
        if baseline_pipeline is not None:
            out["before_mitigation"] = arm(baseline_pipeline)
        pw = [self.evaluate_symmetrical_pairwise(c["id"], c["input"], c["terse_output"], c["verbose_output"],
                                                 c.get("expected_output")) for c in cases]
        tw, vw = sum(r.consensus_winner == "A" for r in pw), sum(r.consensus_winner == "B" for r in pw)
        out["pairwise"] = {"terse_wins": tw, "verbose_wins": vw, "ties": len(pw) - tw - vw,
                           "sign_test_p": round(sign_test_p(tw, vw), 4)}
        return out

    # --------------------------------------------------------------- sycophancy
    def measure_sycophancy_and_style_bias(self, cases: list[dict[str, Any]], use_reference: bool = True) -> dict[str, Any]:
        """Cases: {id, input, correct_output, confidently_wrong_output, expected_output?}.
        use_reference=False hides the gold answer, so the judge must detect the error itself.
        For true sycophancy put the user's false belief INSIDE `input`; the wrong output agrees with it."""
        rows, fooled, ties = [], 0, 0
        for c in cases:
            r = self.evaluate_symmetrical_pairwise(
                c["id"], c["input"], c["correct_output"], c["confidently_wrong_output"],
                c.get("expected_output") if use_reference else None)
            fooled += r.consensus_winner == "B"
            ties += r.consensus_winner == "TIE"
            rows.append({"case_id": c["id"], "winner": r.consensus_winner, "flipped": r.flipped,
                         "correct_score": r.mean_score_a, "wrong_score": r.mean_score_b,
                         "rationale": r.forward_rationale})
        n = len(cases)
        return {"use_reference": use_reference, "total_adversarial_cases": n, "fooled_count": fooled,
                "tied_count": ties, "rejected_count": n - fooled - ties,
                "fool_rate": round(fooled / n, 4) if n else 0.0,
                "fool_rate_ci95": [round(x, 3) for x in wilson_ci(fooled, n)], "probe_details": rows}

    # ---------------------------------------------------------------- clustering
    def measure_score_clustering(self, cases: list[dict[str, Any]], baseline_pipeline=None) -> dict[str, Any]:
        """Spread of pointwise scores on a mixed-quality set. Compare an un-anchored judge (baseline)
        with the few-shot-anchored one. Cases: {id, input, model_output, expected_output?}."""
        def scores(pipe):
            return [pipe.evaluate_pointwise(_tc(c["id"], c["input"], c["model_output"], c.get("expected_output"),
                                               c.get("reference_context"))).overall_score for c in cases]
        out = {"after_mitigation": score_spread(scores(self.pipeline))}
        if baseline_pipeline is not None:
            out["before_mitigation"] = score_spread(scores(baseline_pipeline))
        return out

    # ---------------------------------------------------------- self-enhancement
    def measure_self_preference(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Cases: {id, input, output_judge_family, output_other_family, expected_output?} - same prompt answered by
        the judge's own family and by another family. Preference alone is confounded by real quality
        differences; interpret it next to human labels (add `gold_winner`: 'own'|'other'|'tie')."""
        own = other = ties = 0
        agree_gold = scored = 0
        for c in cases:
            r = self.evaluate_symmetrical_pairwise(c["id"], c["input"], c["output_judge_family"],
                                                   c["output_other_family"], c.get("expected_output"))
            own += r.consensus_winner == "A"
            other += r.consensus_winner == "B"
            ties += r.consensus_winner == "TIE"
            g = c.get("gold_winner")
            if g in ("own", "other", "tie"):
                scored += 1
                agree_gold += {"A": "own", "B": "other", "TIE": "tie"}[r.consensus_winner] == g
        out = {"n": len(cases), "own_family_wins": own, "other_family_wins": other, "ties": ties,
               "own_family_win_rate_among_decisive": round(own / (own + other), 4) if own + other else None,
               "sign_test_p": round(sign_test_p(own, other), 4)}
        if scored:
            out["agreement_with_human_gold"] = {"n": scored, "rate": round(agree_gold / scored, 4)}
        return out
