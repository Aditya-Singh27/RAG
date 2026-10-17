"""Cost model that refuses to invent prices.

Every price lives in data/pricing_assumptions.json as {"value": number|null, "source_url": str, "accessed": "YYYY-MM-DD"}.
A null value prints 'needs source' and the dependent total is not computed. Fill the file from the vendors' own pricing pages
(and re-check the date): this script only does arithmetic.

  python -m eval.cost_model data/pricing_assumptions.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def val(entry):
    return None if entry is None or entry.get("value") is None else float(entry["value"])


def _need(*xs):
    return all(x is not None for x in xs)


def self_hosted(n, a):
    sh, bpv = a["self_hosted"], a["bytes_per_vector"]
    gb = n * bpv["value"] / a["bytes_per_gb"]
    inst, st = val(sh["instance_monthly_usd"]), val(sh["storage_usd_per_gb_month"])
    get_p, gets = val(sh["get_usd_per_1000_requests"]), sh["gets_per_query"]["value"]
    if not _need(inst, st, get_p, gets):
        return gb, None, "needs source"
    reads = a["queries_per_month"]["value"] * gets / 1000 * get_p
    return gb, round(inst + gb * st + reads, 2), f"instance {inst} + storage {gb * st:.2f} + reads {reads:.2f}"


def managed(n, a, gb):
    m = a["managed"]
    tiers = [t for t in m.get("tiers", []) if t.get("monthly_usd") is not None]
    for t in sorted(tiers, key=lambda t: t["max_vectors"]):
        if n <= t["max_vectors"]:
            return {"model": f"tier<= {t['max_vectors']}", "usd": float(t["monthly_usd"]), "source": t.get("source_url")}
    sv = m.get("serverless", {})
    sg, ru, rpq = val(sv.get("storage_usd_per_gb_month")), val(sv.get("usd_per_million_read_units")), sv.get("read_units_per_query", {}).get("value")
    if _need(sg, ru, rpq):
        q = a["queries_per_month"]["value"]
        return {"model": "serverless", "usd": round(gb * sg + q * rpq / 1e6 * ru, 2), "source": sv["storage_usd_per_gb_month"].get("source_url")}
    return {"model": "none", "usd": None, "source": "needs source"}


def build(a):
    if a["bytes_per_vector"].get("value") is None or a["queries_per_month"].get("value") is None:
        raise SystemExit("bytes_per_vector and queries_per_month are null in the pricing file. Measure bytes/vector "
                         "(eval.bench_latency synth -> bytes_per_vector_on_disk) and state a query volume first.")
    rows = []
    for n in a["vector_counts"]:
        gb, sh, expl = self_hosted(n, a)
        mg = managed(n, a, gb)
        rows.append({"vectors": n, "stored_gb": round(gb, 3), "self_hosted_usd": sh, "self_hosted_breakdown": expl,
                     "managed_usd": mg["usd"], "managed_model": mg["model"],
                     "difference_usd": None if sh is None or mg["usd"] is None else round(mg["usd"] - sh, 2),
                     "ratio": None if sh is None or mg["usd"] is None or sh == 0 else round(mg["usd"] / sh, 1)})
    return rows


def markdown(rows):
    f = lambda x: "needs source" if x is None else f"${x:,.2f}"
    lines = ["| Vectors | Stored GB | Self-hosted $/mo | Managed $/mo | Difference | Ratio |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['vectors']:,} | {r['stored_gb']} | {f(r['self_hosted_usd'])} | {f(r['managed_usd'])} | {f(r['difference_usd'])} | {r['ratio'] or '-'} |")
    return "\n".join(lines)


if __name__ == "__main__":
    a = json.load(open(sys.argv[1]))
    rows = build(a)
    Path("results").mkdir(exist_ok=True)
    Path("results/cost_comparison.json").write_text(json.dumps({"assumptions": a, "rows": rows}, indent=2))
    print(markdown(rows))
