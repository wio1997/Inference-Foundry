import json
from pathlib import Path
root=Path("/data/wio/Inference_Foundry/evidence/20260924_loop035_diagnostic/run91")
rows=[json.loads(p.read_text()) for p in sorted((root/"runtime").glob("rank*.json"))]
assert len(rows)==8
r=rows[0]; q=r["repeat"]
assert q is not None
assert all(x["pass"] and all(z["exact"] for z in x["repeat"]["restores"]) for x in rows)
def mismatches(a,b):
    return {(i,j) for i in range(12) for j in range(8) if a[i][j]!=b[i][j]}
a,b,c,b2,c2=r["argmax_a"],r["argmax_b"],r["argmax_c"],q["argmax_b2"],q["argmax_c2"]
stable_product=[]
reference_variable=[]
product_variable=[]
for i in range(12):
  for j in range(8):
    rv={a[i][j],c[i][j],c2[i][j]}
    pv={b[i][j],b2[i][j]}
    if len(rv)>1:reference_variable.append([i,j])
    if len(pv)>1:product_variable.append([i,j])
    if len(rv)==len(pv)==1 and rv!=pv:stable_product.append([i,j])
def top2(label,i,j):
    if label=="B2":
      ids=q["top2_b2_ids"]; scores=q["top2_b2_scores"]
    elif label=="C2":
      ids=q["top2_c2_ids"]; scores=q["top2_c2_scores"]
    else:
      ids=r[f"top2_{label.lower()}_ids"];scores=r[f"top2_{label.lower()}_scores"]
    return {"ids":ids[i*8+j],"scores":scores[i*8+j]}
selected=sorted({tuple(z) for z in stable_product+reference_variable+product_variable+[[2,0]]})
summary={
 "run":"run91",
 "scope":"strict Stock continuous cycle256 A/Product B/Stock C/Product B2/Stock C2 same-state replay",
 "rank_records":len(rows),
 "all_rank_gates_pass":all(x["pass"] for x in rows),
 "all_restores_exact":all(all(z["exact"] for z in x["restores"]+x["repeat"]["restores"]) for x in rows),
 "cache_views":len([z for z in r["strict_coverage"]["captured_names"] if z.startswith("kv_cache")]),
 "metadata_tensors":r["metadata_tensor_count"],
 "rank0_argmax_equal":{
    "AB":r["argmax_equal_ab"],"AC":r["argmax_equal_ac"],
    "BB2":q["argmax_equal_bb2"],"CC2":q["argmax_equal_cc2"]
 },
 "rank0_accepted_equal":{
    "AB":r["accepted_equal_ab"],"AC":r["accepted_equal_ac"],
    "BB2":q["accepted_equal_bb2"],"CC2":q["accepted_equal_cc2"],
    "AB2":q["accepted_equal_ab2"],"AC2":q["accepted_equal_ac2"]
 },
 "rank0_count_vectors":{k:r["counts_"+k] if k in "abc" else q["counts_"+k] for k in ("a","b","c","b2","c2")},
 "rank0_stable_product_only_argmax_positions":stable_product,
 "rank0_reference_variable_positions":reference_variable,
 "rank0_product_variable_positions":product_variable,
 "rank0_selected_top2":{f"{i},{j}":{k:top2(k,i,j) for k in ("A","B","C","B2","C2")} for i,j in selected},
 "all_rank_metrics_identical":all(
   (x["argmax_equal_ab"],x["argmax_equal_ac"],x["repeat"]["argmax_equal_bb2"],x["repeat"]["argmax_equal_cc2"],x["counts_a"],x["counts_b"],x["counts_c"],x["repeat"]["counts_b2"],x["repeat"]["counts_c2"])
   ==(r["argmax_equal_ab"],r["argmax_equal_ac"],q["argmax_equal_bb2"],q["argmax_equal_cc2"],r["counts_a"],r["counts_b"],r["counts_c"],q["counts_b2"],q["counts_c2"])
   for x in rows
 ),
}
max_margins = {}
for label, other, scores in (("B", b, r["top2_b_scores"]),
                              ("C", c, r["top2_c_scores"]),
                              ("B2", b2, q["top2_b2_scores"]),
                              ("C2", c2, q["top2_c2_scores"])):
  changed = mismatches(a, other)
  max_margins[label] = {
    "stock_a": max((r["top2_a_scores"][i*8+j][0] - r["top2_a_scores"][i*8+j][1]
                     for i,j in changed), default=0),
    "comparison": max((scores[i*8+j][0] - scores[i*8+j][1]
                       for i,j in changed), default=0),
  }
summary["rank0_max_top2_margin_at_argmax_disagreements"] = max_margins
summary["interpretation"] = ("Strict five-way replay at continuous Stock cycle256. "
 "Evaluate stable Product-only positions against Stock self-replay and top-2 "
 "margins before assigning semantic divergence.")
(root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
