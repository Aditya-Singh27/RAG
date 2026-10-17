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
        return gb, None, None, "needs source"
    reads = a["queries_per_month"]["value"] * gets / 1000 * get_p
    storage_usd = round(gb * st, 2)
    s3_only_usd = round(storage_usd + reads, 2)
    total_ec2_usd = round(inst + storage_usd + reads, 2)
    expl = f"instance ${inst:.2f} + storage ${storage_usd:.2f} + reads ${reads:.2f}"
    return gb, total_ec2_usd, s3_only_usd, expl


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


def managed_pod(n, a):
    m = a["managed"]
    tiers = [t for t in m.get("tiers", []) if t.get("monthly_usd") is not None]
    for t in sorted(tiers, key=lambda t: t["max_vectors"]):
        if n <= t["max_vectors"]:
            return {"model": f"pod_tier<={t['max_vectors']}", "usd": float(t["monthly_usd"]), "source": t.get("source_url")}
    return {"model": "pod_none", "usd": None, "source": "needs source"}


def managed_serverless(n, a, gb):
    sv = a["managed"].get("serverless", {})
    sg, ru, rpq = val(sv.get("storage_usd_per_gb_month")), val(sv.get("usd_per_million_read_units")), sv.get("read_units_per_query", {}).get("value")
    if _need(sg, ru, rpq):
        q = a["queries_per_month"]["value"]
        storage_usd = round(gb * sg, 2)
        read_usd = round(q * rpq / 1e6 * ru, 2)
        return {"model": "serverless", "usd": round(storage_usd + read_usd, 2), "storage_usd": storage_usd, "read_usd": read_usd, "source": sv["storage_usd_per_gb_month"].get("source_url")}
    return {"model": "none", "usd": None, "source": "needs source"}


def calc_break_even_queries(n, a, gb):
    """Calculate query volume where self-hosted EC2+S3 breaks even with Pinecone Serverless."""
    sh = a["self_hosted"]
    sv = a["managed"].get("serverless", {})
    inst, st = val(sh["instance_monthly_usd"]), val(sh["storage_usd_per_gb_month"])
    get_p, gets = val(sh["get_usd_per_1000_requests"]), sh["gets_per_query"]["value"]
    sg, ru, rpq = val(sv.get("storage_usd_per_gb_month")), val(sv.get("usd_per_million_read_units")), sv.get("read_units_per_query", {}).get("value")
    if not _need(inst, st, get_p, gets, sg, ru, rpq):
        return None
    fixed_sh = inst + gb * st
    fixed_sv = gb * sg
    var_sh = (gets / 1000.0) * get_p
    var_sv = (rpq / 1e6) * ru
    if var_sv <= var_sh:
        return None
    q_breakeven = (fixed_sh - fixed_sv) / (var_sv - var_sh)
    return round(q_breakeven)


def build(a):
    if a["bytes_per_vector"].get("value") is None or a["queries_per_month"].get("value") is None:
        raise SystemExit("bytes_per_vector and queries_per_month are null in the pricing file. Measure bytes/vector "
                         "(eval.bench_latency synth -> bytes_per_vector_on_disk) and state a query volume first.")
    rows = []
    for n in a["vector_counts"]:
        gb, sh_ec2, sh_s3, expl = self_hosted(n, a)
        mg = managed(n, a, gb)
        pod = managed_pod(n, a)
        sv = managed_serverless(n, a, gb)
        be_q = calc_break_even_queries(n, a, gb)
        diff = None if sh_ec2 is None or mg["usd"] is None else round(mg["usd"] - sh_ec2, 2)
        ratio = None if sh_ec2 is None or mg["usd"] is None or sh_ec2 == 0 else round(mg["usd"] / sh_ec2, 1)
        rows.append({
            "vectors": n,
            "stored_gb": round(gb, 3),
            "self_hosted_usd": sh_ec2,
            "self_hosted_ec2_usd": sh_ec2,
            "self_hosted_s3_only_usd": sh_s3,
            "self_hosted_breakdown": expl,
            "managed_usd": mg["usd"],
            "managed_model": mg["model"],
            "pinecone_pod_usd": pod["usd"],
            "pinecone_serverless_usd": sv["usd"],
            "difference_usd": diff,
            "ratio": ratio,
            "ratio_vs_pod": None if sh_ec2 is None or pod["usd"] is None or sh_ec2 == 0 else round(pod["usd"] / sh_ec2, 1),
            "break_even_queries_month": be_q,
        })
    return rows


def markdown(rows):
    f = lambda x: "needs source" if x is None else f"${x:,.2f}"
    lines = [
        "| Vectors | Stored GB | Self-Hosted (EC2+S3) $/mo | Self-Hosted (Pure S3) $/mo | Pinecone Pods $/mo | Pinecone Serverless $/mo | Pod Savings Ratio | Break-Even Queries/mo |",
        "|---|---|---|---|---|---|---|---|"
    ]
    for r in rows:
        be = f"{r['break_even_queries_month']:,}" if r.get('break_even_queries_month') else "N/A"
        lines.append(
            f"| {r['vectors']:,} | {r['stored_gb']} | {f(r['self_hosted_ec2_usd'])} | {f(r['self_hosted_s3_only_usd'])} | "
            f"{f(r['pinecone_pod_usd'])} | {f(r['pinecone_serverless_usd'])} | {r['ratio_vs_pod'] or '-'}x | {be} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    a = json.load(open(sys.argv[1]))
    rows = build(a)
    Path("results").mkdir(exist_ok=True)
    Path("results/cost_comparison.json").write_text(json.dumps({"assumptions": a, "rows": rows}, indent=2))
    print(markdown(rows))
