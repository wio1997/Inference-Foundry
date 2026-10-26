import json
from pathlib import Path
root=Path("/data/wio/Inference_Foundry/evidence/20260924_loop035_diagnostic/run88")
rows=[json.loads(p.read_text()) for p in sorted((root/"runtime").glob("rank*.json"))]
assert len(rows)==8
r=rows[0]
a,b,c=r["argmax_a"],r["argmax_b"],r["argmax_c"]
ab={(i,j) for i in range(12) for j in range(8) if a[i][j]!=b[i][j]}
ac={(i,j) for i in range(12) for j in range(8) if a[i][j]!=c[i][j]}
def count(x):return sum(v["mismatched_elements"] for v in x["mismatches"])
summary={
 "run":"run88",
 "scope":"Stock continuous cycle128 strict source-backed 67-view page and 76-metadata snapshot; Stock A/Product B/Stock C",
 "rank_records":len(rows),
 "all_rank_gates_pass":all(x["pass"] for x in rows),
 "all_restores_exact":all(all(z["exact"] for z in x["restores"]) for x in rows),
 "all_rank_metrics_identical":all(
  (x["argmax_equal_ab"],x["argmax_equal_ac"],x["accepted_equal_ab"],x["accepted_equal_ac"],x["counts_a"],x["counts_b"],x["counts_c"])
  ==(r["argmax_equal_ab"],r["argmax_equal_ac"],r["accepted_equal_ab"],r["accepted_equal_ac"],r["counts_a"],r["counts_b"],r["counts_c"])
  for x in rows),
 "cache_views":len([v for v in r["strict_coverage"]["captured_names"] if v.startswith("kv_cache")]),
 "snapshot_entries":r["strict_coverage"]["captured_rows"],
 "metadata_tensors":r["metadata_tensor_count"],
 "operator_checks_rank0":json.loads((root/"pages"/"rank0.json").read_text())["rows"][0]["operator_checks"],
 "rank0_argmax_equal_ab":r["argmax_equal_ab"],
 "rank0_argmax_equal_ac":r["argmax_equal_ac"],
 "rank0_accepted_equal_ab":r["accepted_equal_ab"],
 "rank0_accepted_equal_ac":r["accepted_equal_ac"],
 "rank0_count_mismatch_slots_ab":[i for i,(x,y) in enumerate(zip(r["counts_a"],r["counts_b"])) if x!=y],
 "rank0_count_mismatch_slots_ac":[i for i,(x,y) in enumerate(zip(r["counts_a"],r["counts_c"])) if x!=y],
 "rank0_ab_mismatch_positions":sorted(map(list,ab)),
 "rank0_ac_mismatch_positions":sorted(map(list,ac)),
 "rank0_ab_only_positions":sorted(map(list,ab-ac)),
 "rank0_ac_only_positions":sorted(map(list,ac-ab)),
 "rank0_post_write_mismatched_elements_ab":count(r["post_write_ab"]),
 "rank0_post_write_mismatched_elements_ac":count(r["post_write_ac"]),
 "interpretation":"Strict restores pass. Product B and Stock C have equal argmax mismatch counts versus Stock A, but B-only first-token/count split at slot2 merits same-state repeated replay. Cache value post-write mismatches are large in both AB and Stock AC and cannot be attributed to Product without a reference noise floor.",
}
(root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k!="operator_checks_rank0"},indent=2))
