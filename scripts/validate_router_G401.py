"""Offline audit of actual native router receipts and pinned source; no NPU load."""
from pathlib import Path
import hashlib,itertools,json
R=Path(__file__).resolve().parents[1];b=R/'evidence/native_router_G401';a=json.loads((b/'acceptance.json').read_text())
assert a['accepted_component'] and not a['full_model_PD_quality_performance_accepted']
s=json.loads((b/'state.json').read_text());assert s['phase']=='stopped_released' and s['cleanup']['all8_process_lists_clear']
expected={(n,k,z,q,rows,mode) for n,k in [(384,6),(128,3)] for z in [0,1] for q in [False,True] for rows in [0,1,16,96,8192] for mode in ['structured','extreme','ties']}
cases=graphs=replays=zero=0;errors=[]
for rank in range(8):
 row=json.loads((b/f'evidence/rank{rank}.json').read_text());assert row['rank']==rank and row['phase']=='passed' and row['device']=='Ascend910B1' and row['native_source_pins_verified']
 assert {(c['experts'],c['routed_topk'],c['shared_slots'],c['apply_scale'],c['rows'],c['mode']) for c in row['cases']}==expected
 for c in row['cases']:
  assert c['phase']=='passed';cases+=1;graphs+=c['graphs'];replays+=c['changed_replays'];errors.append(c['max_abs_error'])
  if c['rows']==0:zero+=1;assert c['graphs']==c['changed_replays']==0
  else:assert c['graphs']==1 and c['changed_replays']==3 and c['CPU_FP32_exact_expert_IDs'] and len(set(c['hashes']))==3
assert (cases,graphs,replays,zero)==(960,768,2304,192)
assert max(errors)==a['max_abs_weight_error_FP64']
m=json.loads((b/'lowering_manifest.json').read_text());assert hashlib.sha256((b/'route_kernel.py').read_bytes()).hexdigest()==m['derived_kernel_sha256']
c=R/'code/full_model_candidates/sg_pd_g396';assert (c/'route_kernel.py').read_bytes()==(b/'route_kernel.py').read_bytes()
assert json.loads((c/'native_route_sha256.json').read_text())==json.loads((b/'native_source_sha256.json').read_text())
for name,h in json.loads((b/'native_source_sha256.json').read_text()).items():assert hashlib.sha256((R/'code/fixed_router_G385'/name).read_bytes()).hexdigest()==h
print(json.dumps({'offline_actual_ranks':8,'cases':cases,'graphs':graphs,'changed_replays':replays,'zero_eager':zero,'max_weight_error_FP64':max(errors),'full_model_PD_accepted':False}))
