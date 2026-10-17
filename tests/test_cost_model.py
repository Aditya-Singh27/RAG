import copy
from eval.cost_model import build, markdown

# TEST FIXTURE ONLY: made-up round numbers to check arithmetic. Not prices.
A = {"vector_counts": [1000, 2000], "bytes_per_gb": 1000, "queries_per_month": {"value": 1000},
     "bytes_per_vector": {"value": 10},
     "self_hosted": {"instance_monthly_usd": {"value": 5.0}, "storage_usd_per_gb_month": {"value": 2.0},
                     "get_usd_per_1000_requests": {"value": 4.0}, "gets_per_query": {"value": 3}},
     "managed": {"tiers": [{"max_vectors": 1000, "monthly_usd": 50.0}],
                 "serverless": {"storage_usd_per_gb_month": {"value": 10.0}, "usd_per_million_read_units": {"value": 1e6},
                                "read_units_per_query": {"value": 2}}}}


def test_arithmetic():
    r = build(A)
    # n=1000: gb = 1000*10/1000 = 10 ; reads = 1000*3/1000*4 = 12 ; total = 5 + 10*2 + 12 = 37
    assert r[0]["stored_gb"] == 10 and r[0]["self_hosted_usd"] == 37.0 and r[0]["managed_usd"] == 50.0
    assert r[0]["difference_usd"] == 13.0 and r[0]["ratio"] == 1.4
    # n=2000 falls through tiers to serverless: 20gb*10 + 1000*2/1e6*1e6 = 200 + 2000
    assert r[1]["managed_usd"] == 2200.0 and r[1]["managed_model"] == "serverless"


def test_missing_prices_are_not_invented():
    a = copy.deepcopy(A); a["self_hosted"]["instance_monthly_usd"]["value"] = None; a["managed"]["tiers"][0]["monthly_usd"] = None
    r = build(a)
    assert r[0]["self_hosted_usd"] is None and r[0]["difference_usd"] is None
    assert "needs source" in markdown(r)
