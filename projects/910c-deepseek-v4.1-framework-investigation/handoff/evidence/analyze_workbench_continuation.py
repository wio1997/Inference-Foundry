#!/usr/bin/env python3
"""Recompute conditional gates from archived same-window inputs; no device work."""
import argparse
import json
import statistics
from pathlib import Path


def summary(values):
    return dict(n=len(values), min=min(values), median=statistics.median(values), max=max(values))


def pair_order(offsets, real_indices=(0, 1), limit=100, block=1, dim=512):
    """Small CPU branch witness for CopyInKv, not a native equivalence test.

    Offsets are elements, stride is bytes for BF16. This witness covers only
    the paired/single-copy choice; it does not model full pipeline packing.
    """
    valid = [o >= 0 and 0 <= i < limit for o, i in zip(offsets, real_indices)]
    stride = (abs(offsets[0] - offsets[1]) - block * dim) * 2
    fallback = stride < 0 or stride >= 2**31 - 1 or any(i + block >= limit for i in real_indices)
    order = [i for i in range(2) if valid[i]]
    if not fallback:
        order.sort(key=lambda i: offsets[i])
    return dict(branch="single-copy fallback" if fallback else "paired physical-stride copy", order=order)


def analyze(data):
    result = dict(scope=data["scope"], new_NPU_measurements=0, router={}, warnings=[
        "Router results hold GEMM, outside-stage readiness, host submission and expert work fixed.",
        "Same-window local-period sums are conditional prioritization budgets, not service speedup predictions.",
        "QLI uses an optimistic same-cost-per-MAC scenario; its two source cohorts do not identify scoring fraction.",
        "DP and copy-order witnesses are static/CPU branch checks, not graph or NPU acceptance.",
    ])
    for scope, rows in data["router"].items():
        cases = []
        for x in rows:
            groups = x["groups"]
            free = groups["packet_and_gating"]["native_union_us"] + groups["shared_rms_boundary"]["native_union_us"]
            required = 0.01 * x["local_period_us"]
            gemm = groups["router_gemm"]["native_union_us"]
            cases.append(dict(rank=x["rank"], launch_TASK_ROW=x["launch_TASK_ROW"],
                free_us=free, required_us=required, free_percent=100*free/x["local_period_us"],
                gemm_us=gemm, required_additional_GEMM_fraction=(required-free)/gemm))
        result["router"][scope] = dict(
            non_GEMM_free_percent=summary([x["free_percent"] for x in cases]),
            required_additional_GEMM_fraction=summary([x["required_additional_GEMM_fraction"] for x in cases]),
            all_windows_below_one_percent=all(x["free_percent"] < 1 for x in cases), windows=cases)
    q=data["qli"]; old=q["old_four_calls_ms"]; fraction=q["ideal_removed_score_MAC_fraction"]
    threshold=q["batch_ms"]*.01/q["reference_critical_waves"]
    result["qli"] = dict(required_net_ms_per_four_calls=threshold,
        minimum_scoring_fraction_at_equal_speed_zero_overhead=threshold/(fraction*old),
        max_new_overhead_ms_if_whole_old_stage_is_scoring=old*fraction-threshold,
        minimum_relative_rate_if_whole_old_stage_is_scoring=(1-fraction)*old/(old-threshold),
        formula="gain = f*c*T - (1-f)*c*T*(1/rho-1) - overhead; need gain>=threshold",
        note="c is exposed score time / old total time; rho is compact/dense rate. c=1 is deliberately favorable.")
    dp=data["DP_counterexample"]
    assert dp[0]["target_bucket_tokens"] == dp[1]["target_bucket_tokens"]
    assert dp[0]["draft_query_tokens"] != dp[1]["draft_query_tokens"]
    result["DP"] = dict(target_padded_count_alone_is_sufficient=False, counterexample=dp,
        required_packet_fields=["epoch", "per-DP request count", "scheduled K/query width", "live/dummy/skip role", "agreed fallback", "target graph mode separately"])
    result["SMLA_pair_witnesses"] = [dict(offsets=[4096,1024],**pair_order([4096,1024])),
        dict(offsets=[1024,1024],**pair_order([1024,1024])),
        dict(offsets=[4096,-1],real_indices=[99,100],**pair_order([4096,-1],(99,100)))]
    assert result["SMLA_pair_witnesses"][0]["order"] == [1,0]
    assert result["SMLA_pair_witnesses"][1]["order"] == [0,1]
    assert result["SMLA_pair_witnesses"][2]["order"] == [0]
    return result


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--inputs",type=Path,required=True);ap.add_argument("--output",type=Path)
    args=ap.parse_args(); result=analyze(json.loads(args.inputs.read_text()))
    content=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if args.output: args.output.write_text(content)
    else: print(content,end="")
