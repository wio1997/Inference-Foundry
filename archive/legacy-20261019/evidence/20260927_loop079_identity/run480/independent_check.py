from pathlib import Path
import json,sys,hashlib,statistics,math,re
sys.path.insert(0,"scripts")
from loop079_logits_join_validate_run472 import validate,final_admit
base=Path("evidence/20260927_loop079_identity");root=base/"run472/b_clean2";out=base/"run480"
result={"validator":validate(root),"final_admit":final_admit(root)}
rows=[json.loads(f.read_text()) for f in sorted((root/"capture").glob("rank*_cohort*.json"))]
prior=[json.loads(f.read_text()) for f in sorted((base/"run439/b_retry2/capture").glob("rank*_cohort*.json"))]
reduced=json.loads((base/"run477/intervals.json").read_text())
for rel,h in reduced["input_sha256"].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h
edges=list(rows[0]["elapsed"])+["P_C1_ms"]
def value(d,e):return sum(d["elapsed"].values()) if e=="P_C1_ms" else d["elapsed"][e]
def stats(ds,e):
 vals=[value(d,e)*1000 for d in ds];q=statistics.quantiles(vals,n=4,method="inclusive")
 return dict(n=len(vals),min_us=min(vals),q1_us=q[0],median_us=statistics.median(vals),q3_us=q[2],max_us=max(vals),mad_us=statistics.median(abs(v-statistics.median(vals)) for v in vals))
result["intervals_us"]={e:stats(rows,e) for e in edges};result["old_queue_drained_intervals_us"]={e:stats(prior,e) for e in edges}
for e in edges:
 s=result["intervals_us"][e];p=reduced["all_slices"][e]
 for a,b in [("min_us","min_ms"),("median_us","median_ms"),("max_us","max_ms")]:assert abs(s[a]/1000-p[b])<1e-12
result["by_rank"]={r:{e:stats([d for d in rows if d["rank"]==r],e) for e in edges} for r in range(8)}
result["by_cohort"]={c:{e:stats([d for d in rows if d["cohort"]==c],e) for e in edges} for c in range(1,6)}
cohorts=[]
for c in range(1,6):
 ds=[d for d in rows if d["cohort"]==c];d=ds[0];rs=[json.loads((root/f"runtime/rank{r}_cohort{c}.json").read_text()) for r in range(8)]
 for item,r in zip(ds,rs):
  assert item["rank"]==r["rank"]
  assert item["stream"]==["npu:"+str(r["rank"]),dict(stream_id=0,device_index=r["rank"],device_type=20)]
  assert r["handoff_before_model_forward"] and r["target_graph_requested"] and r["model_runner_cycles_after_handoff"]==0
  assert r["pass"] and r["host_mirror_exact"]
  sums=[sum(row[i] for row in item["accepted_counts"]) for i in range(12)]
  assert sums==r["staged_output_counts"] and sum(sums)-12288==r["overshoot_tokens"]
  assert all(v>=1024 for v in sums) and all(v>0 for v in item["accepted_counts"][64])
  assert item["lineage"]["native_output"]["storage"]!=item["lineage"]["layout_return"]["storage"]
  assert item["lineage"]["native_output"]["storage_bytes"]==item["lineage"]["layout_return"]["storage_bytes"]==24821760
  shard=item["algorithm_shard"];assert shard["org_vocab_start"]==r["rank"]*16160 and shard["org_vocab_end"]==(r["rank"]+1)*16160
  assert list(item["records"])==["P","J","G","C0","C1"]
  h=[item["records"][x]["host_ns"] for x in item["records"]];assert h==sorted(h)
  assert h[0]<item["native_entry_ns"]<item["native_return_ns"]<h[1]
 cohorts.append(dict(cohort=c,cycles=d["cycles"],useful_tokens=12288,staged_tokens=sum(rs[0]["staged_output_counts"]),overshoot=rs[0]["overshoot_tokens"],useful_per_cycle=12288/d["cycles"],cycle64_tokens=sum(d["accepted_counts"][64]),runtime_wall_range_s=[min(r["wall_seconds"] for r in rs),max(r["wall_seconds"] for r in rs)]))
result["cohorts"]=cohorts
result["requests"]={}
for name in ["warmup48","bench"]:
 d=json.loads((root/(name+".json")).read_text());result["requests"][name]=dict(summary=d["summary"],input_token_counts=sorted(set(r["input_tokens"] for r in d["requests"])))
log=Path("logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP079-RUN472-CLEAN-B2.log").read_text()
post=[s for s in log.splitlines() if "POST /v1/chat/completions" in s];assert len(post)==60 and all("200 OK" in s for s in post)
errors=[s for s in log.splitlines() if re.search(r"\bERROR\b|Traceback|OUT_OF_SCOPE|RuntimeError|AssertionError",s)]
assert not errors
warnings=[s for s in log.splitlines() if "WARNING" in s]
(out/"server_warning_lines.txt").write_text("\n".join(warnings)+"\n")
result["server_log"]=dict(sha256=hashlib.sha256(log.encode()).hexdigest(),posts_200=len(post),error_matches=errors,warning_lines=len(warnings))
assert (root/"source_before.sha256").read_bytes()==(root/"source_after.sha256").read_bytes()
result["provenance_hashes"]={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(root.glob("*.json"))+list(root.glob("*.txt"))+list(root.glob("*.sha256"))+list((root/"capture").glob("*.json"))+list((root/"runtime").glob("*.json"))}
(out/"independent_checks.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k in ("intervals_us","cohorts","server_log","requests")},indent=2))
