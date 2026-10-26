#!/usr/bin/env python3
"""Audit the 8-rank live group_list snapshots of Run121."""
import json
import statistics
from pathlib import Path
ROOT = Path("/data/wio/Inference_Foundry/evidence/20260925_loop039_gmm/run121")
data = {}
for p in (ROOT / "counts").glob("rank*_cycle*.json"):
    d = json.loads(p.read_text())
    key = d["rank"], d["cycle"]
    assert key not in data, key
    data[key] = d
assert set(data) == {(r, c) for r in range(8) for c in (64, 65)}
assert all(len(d["rows"]) == 86 for d in data.values())
stable, live = [], []
for o in range(86):
    changed = [
        data[r,64]["rows"][o]["counts"] != data[r,65]["rows"][o]["counts"]
        for r in range(8)
    ]
    assert all(changed) or not any(changed), (o, changed)
    (live if all(changed) else stable).append(o)
assert stable == list(range(43)), stable
assert live == list(range(43, 86)), live
per_layer = []
per_rank = []
for cycle in (64,65):
    for rank in range(8):
        rows = [data[rank,cycle]["rows"][o]["counts"] for o in live]
        assert all(len(row)==32 and all(isinstance(x,int) and 0<=x<=96 for x in row) for row in rows)
        per_rank.append({
            "cycle": cycle, "rank": rank,
            "active_experts_per_layer": [sum(x>0 for x in row) for row in rows],
            "tokens_per_layer": [sum(row) for row in rows],
        })
    for layer, ordinal in enumerate(live):
        rows = [data[rank,cycle]["rows"][ordinal]["counts"] for rank in range(8)]
        assert sum(map(sum,rows)) == 576, (cycle,layer)
        per_layer.append({
            "cycle": cycle, "layer": layer,
            "active_experts_global": sum(x>0 for row in rows for x in row),
            "active_experts_per_rank": [sum(x>0 for x in row) for row in rows],
            "tokens_per_rank": [sum(row) for row in rows],
            "max_tokens_per_expert": max(x for row in rows for x in row),
        })
act=[a for rec in per_rank for a in rec["active_experts_per_layer"]]
global_act=[rec["active_experts_global"] for rec in per_layer]
rank_tokens=[t for rec in per_rank for t in rec["tokens_per_layer"]]
result={
 "run":"run121", "contract":"12 requests, concurrency 12, exact max_tokens=1024, 8x910B3 DP1TP8 DSpark7",
 "status":"valid_live_counts",
 "files":len(data), "refs_per_rank":86,
 "static_ordinals":stable, "live_ordinals":live,
 "cross_rank_tokens_per_layer":576,
 "samples_rank_layer":len(act), "samples_global_layer":len(global_act),
 "active_experts_per_rank_layer":{
   "min":min(act),"median":statistics.median(act),"mean":statistics.mean(act),"max":max(act),
   "p10":sorted(act)[int(len(act)*.1)],"p90":sorted(act)[int(len(act)*.9)]},
 "active_experts_global_layer":{
   "min":min(global_act),"median":statistics.median(global_act),
   "mean":statistics.mean(global_act),"max":max(global_act)},
 "tokens_per_rank_layer":{
   "min":min(rank_tokens),"median":statistics.median(rank_tokens),
   "mean":statistics.mean(rank_tokens),"max":max(rank_tokens)},
 "layers":per_layer,
}
(ROOT/"counts_summary.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="layers"},indent=2))
