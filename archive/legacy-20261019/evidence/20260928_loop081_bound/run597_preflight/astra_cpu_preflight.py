import ast,copy,hashlib,json,sys
from pathlib import Path
root=Path("/data/wio/Inference_Foundry");out=root/"evidence/20260928_loop081_bound/run597_preflight"
helper=root/"scripts/loop081_basis_capture_run597.py";tree=ast.parse(helper.read_text());sub=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("_prefix","_check_basis")],type_ignores=[]);ns={};exec(compile(sub,str(helper),"exec"),ns)
sys.path.insert(0,str(root/"scripts"));import loop081_host_event_basis_run596 as p596;import loop081_basis_patch_run597 as patch
manifest,old,new=patch.prepared();checked=json.loads((root/"evidence/20260928_loop081_bound/run597_patch_check.json").read_text())
records=[];mut_results={}
for cohort in range(5,9):
 trace=json.loads((p596.OLD/"trace"/f"trace_rank0_cohort{cohort}.json").read_text());runtime=json.loads((p596.OLD/"runtime"/f"rank0_cohort{cohort}.json").read_text());events=[dict(slot=e["slot"],next_cycle=e["cycle"],anchor=e["anchor"]) for e in p596.extract_host_events(trace)];stop={e["slot"]:e["next_cycle"] for e in events};N=len(trace)
 selected=sorted(({0,64,128,192,256,300}|{min(e["next_cycle"] for e in events)})&set(range(N)))
 rec=dict(cycles=N,initial=dict(position=trace[0]["num_computed_before"],last_token=trace[0]["last_token_before"],draft=trace[0]["draft_before"]),token_history=[r["accepted"] for r in trace],count_history=[[r["counts"][s] if t<stop.get(s,N) else 0 for s in range(12)] for t,r in enumerate(trace)],draft_history=[r["next_draft"] for r in trace],host_park_events=events,branch_history=[dict(cycle=t,scheduled_target=False,schedule_mode="off") for t in range(N)],target_witnesses=[],dspark_witnesses=[],generated_output_counts=runtime["generated_output_counts"],staged_output_counts=runtime["staged_output_counts"])
 for t in selected:
  row=trace[t];ids=sum(row["target_input_ids"],[]);pos=sum(row["target_positions"],[]);seq=[v+8 for v in row["num_computed_before"]]
  rec["target_witnesses"].append(dict(cycle=t,input_ids=ids,positions=pos,seq_lens=seq))
  rec["dspark_witnesses"].append(dict(cycle=t,token_indices=list(range(96)),sample_indices=[s*8+row["counts"][s]-1 for s in range(12)],num_rejected=[8-v for v in row["counts"]],query_start_loc=list(range(0,97,8)),seq_lens=seq,target_token_ids=ids,target_positions=pos))
 ans=ns["_check_basis"](rec);records.append(dict(cohort=cohort,**ans))
 if cohort==5:
  variants=[]
  for field in ("sample_indices","num_rejected","query_start_loc","seq_lens"):
   bad=copy.deepcopy(rec);bad["dspark_witnesses"][0][field]=[-999];variants.append(("corrupt_dspark_"+field,bad))
  bad=copy.deepcopy(rec);bad["target_witnesses"].append(copy.deepcopy(bad["target_witnesses"][0]));variants.append(("duplicate_target_witness",bad))
  bad=copy.deepcopy(rec);bad["target_witnesses"]=bad["target_witnesses"][:1];bad["dspark_witnesses"]=bad["dspark_witnesses"][:1];variants.append(("omit_all_checkpoints_after_cycle0",bad))
  for name,bad in variants:
   try:ns["_check_basis"](bad);mut_results[name]="ADMITTED"
   except Exception as e:mut_results[name]=type(e).__name__
result=dict(scope="CPU-only source/preflight audit; DSpark checkpoint arrays synthesized, not captured evidence",source_patch_compile=True,manifest_matches_check=manifest=={k:v for k,v in checked.items() if k!="action"},a0_target_basis_regression=records,negative_admission_gaps=mut_results,helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),patch_sha256=hashlib.sha256(Path(patch.__file__).read_bytes()).hexdigest(),patch_check_sha256=hashlib.sha256((root/"evidence/20260928_loop081_bound/run597_patch_check.json").read_bytes()).hexdigest())
(out/"astra_preflight_checks.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
