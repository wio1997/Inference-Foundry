#!/usr/bin/env python3
"""Conditional retained-row expert footprint relaxation, not attainable bytes."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"evidence/20260927_loop077_bound/run375/capture"
PRIOR=json.loads((ROOT/"evidence/20260927_loop077_bound/run380/analysis.json").read_text())
WEIGHT_BYTES=12582912  # current packed GMM1+GMM2 footprint per local expert, Run380
rows=[]
for cohort in range(1,6):
    objs=[json.loads((BASE/f"rank{rank}_cohort{cohort}.json").read_text()) for rank in range(8)]
    assert [o["rank"] for o in objs]==list(range(8))
    assert all(o["accepted_counts_by_cycle_slot"]==objs[0]["accepted_counts_by_cycle_slot"] for o in objs)
    counts=objs[0]["accepted_counts_by_cycle_slot"]
    remaining=objs[0]["remaining"]
    produced=[0]*12
    useful=[]
    for accepted in counts:
        kept=[min(a,max(0,remaining[i]-produced[i])) for i,a in enumerate(accepted)]
        produced=[x+y for x,y in zip(produced,kept)]
        useful.append(sum(kept))
    assert produced==remaining
    for cycle in (64,65):
        u=useful[cycle]
        rank_route=[next(r["counts"] for r in o["route_rows"] if r["cycle"]==cycle) for o in objs]
        assert all(len(x)==43 and all(len(y)==32 for y in x) for x in rank_route)
        layer_bounds=[]
        for layer in range(43):
            expert_counts=[count for rank in range(8) for count in rank_route[rank][layer]]
            assert sum(expert_counts)==576
            cap=sorted((min(c,u) for c in expert_counts if c),reverse=True)
            cumulative=0
            k=0
            for val in cap:
                if cumulative>=6*u: break
                cumulative+=val
                k+=1
            assert cumulative>=6*u
            active=sum(bool(c) for c in expert_counts)
            layer_bounds.append({"lower_unique_experts":k,"upper_unique_experts":min(active,6*u),"current_unique_experts":active})
        low=sum(x["lower_unique_experts"] for x in layer_bounds)
        high=sum(x["upper_unique_experts"] for x in layer_bounds)
        current=sum(x["current_unique_experts"] for x in layer_bounds)
        prior_rows=[x for x in PRIOR["rows"] if x["cohort"]==cohort and x["cycle"]==cycle]
        assert len(prior_rows)==8
        assert abs(sum(x["active_packed_weight_GB"] for x in prior_rows)-current*WEIGHT_BYTES/1e9)<1e-6
        rows.append({"cohort":cohort,"cycle":cycle,"useful_Runtime_clipped_rows":u,"selected_routed_slots_relaxation":6*u,
                     "TP8_packed_weight_current_GB":current*WEIGHT_BYTES/1e9,
                     "TP8_retained_row_unique_packed_weight_footprint_relaxation_GB":[low*WEIGHT_BYTES/1e9,high*WEIGHT_BYTES/1e9],
                     "sum_expert_layer_pairs":{"lower":low,"upper":high,"current":current},
                     "layer_bounds":layer_bounds})
assert [[r["useful_Runtime_clipped_rows"] for r in rows if r["cohort"]==c] for c in range(1,6)]==PRIOR["summary"]["useful_clipped_tokens_cycle64_65_by_cohort"]
result={"status":"conditional_retained_row_expert_union_relaxation_not_compulsory_traffic",
        "weight_bytes_per_expert_layer":WEIGHT_BYTES,"rows":rows,
        "limits":["Assumes U Runtime-clipped useful outputs correspond to U Target rows worth of six expert routes per layer, with unchanged current candidate expert multisets. Target rejection outcome is unknown before completing earlier candidate forward passes; this is a foreknowledge resource relaxation, not an implementable schedule.",
                  "Aggregate per-expert counts do not identify which expert IDs belong to retained rows; lower uses top capped counts min(c_e,U), upper uses min(active,6U). Bounds are not simultaneously feasible at all layers in general, only per-layer independent relaxations.",
                  "Packed unique weight footprint is not compulsory physical HBM bytes: cache/persistence, scale/metadata and alternative routing or layout unresolved. This TP8 sum cannot be divided by bandwidth to make Product latency without rank, overlap and capacity DAG.",
                  "Run375 adds a selected-cycle device route stack, and only ten early cycles are represented; U is Runtime clipped, not a complete external output ledger."]}
out=ROOT/"evidence/20260927_loop077_bound/run397/relaxation.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"status":result["status"],"range_GB":[[r["TP8_retained_row_unique_packed_weight_footprint_relaxation_GB"] for r in rows]]}))
