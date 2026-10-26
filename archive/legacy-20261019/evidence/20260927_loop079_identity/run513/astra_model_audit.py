#!/usr/bin/env python3
"""CPU-only independent V3.23 byte/gate review; writes only Run513."""
import copy, hashlib, importlib, json, pathlib, sys, tempfile
sys.dont_write_bytecode=True
ROOT=pathlib.Path('/data/wio/Inference_Foundry');OUT=ROOT/'evidence/20260927_loop079_identity/run513';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts'))
m=importlib.import_module('extreme_bound_calibration_v3_23')
p=importlib.import_module('extreme_bound_calibration_v3_22')
e=importlib.import_module('extreme_bound_calibration_v3_16_rev2')
d=importlib.import_module('extreme_bound_calibration_v3_15')
inputs={};checks=[]
def read(path):
 path=pathlib.Path(path);b=path.read_bytes();inputs[str(path)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return b
def ok(name):checks.append({'name':name,'pass':True})
def reject(name,fn):
 try:fn()
 except (ValueError,AssertionError,KeyError):ok(name);return
 raise AssertionError('unexpected acceptance '+name)
for path,digest in {**p.HASHES,**m.HASHES}.items():assert hashlib.sha256(read(ROOT/path)).hexdigest()==digest
ok('all V3.22 and V3.23 pinned evidence hashes')
for name in ['extreme_bound_calibration_v3_23.py','extreme_bound_calibration_v3_22.py','extreme_bound_calibration_v3_16_rev2.py','extreme_bound_calibration_v3_15.py']:read(ROOT/'scripts'/name)
model=m.build();raw=(json.dumps(model,ensure_ascii=False,indent=2)+'\n').encode()
target=ROOT/'evidence/20260927_loop079_identity/run512/bound_calibration_v3_23.json'
assert raw==read(target);(OUT/'rebuild.json').write_bytes(raw);ok('byte-identical final V3.23 rebuild')
prior=json.loads(read(ROOT/m.PRIOR));assert model['current']==prior['current'] and model['current']['accepted_formal_tps']==571.681
assert model['contract']==prior['contract'];ok('unchanged frozen contract and formal Current')
nodes=model['proof_dag']['nodes'];resolved=d.validate_dag(nodes)
assert len(nodes)==len(resolved)==19 and resolved==model['proof_dag']['certified'] and not any(resolved.values())
assert set(nodes)==set(prior['proof_dag']['nodes']);ok('independent 19-node DAG recomputation all false')
endpoint_paths=[]
def walk(x,path=''):
 if isinstance(x,dict):
  for key,val in x.items():
   new=path+'.'+key
   if key in e.NUMERIC_ENDPOINT_KEYS:assert val is None;endpoint_paths.append(new)
   walk(val,new)
 elif isinstance(x,list):
  for i,val in enumerate(x):walk(val,path+f'[{i}]')
walk(model);e.require_null_endpoints(model);ok('all known recursive numeric endpoint fields null')
sched=model['bound_ladder']['scheduling_execution']['run508_selected_MLA_update']
assert sched['attention_key_count_each']==170 and sched['attn_params_handles_events_count_each']==0 and sched['selected_zip_iterations_each']==sched['source_inferred_graph_task_updates_each']==sched['source_inferred_event_records_each']==0
assert all(sched[k] is None for k in ['stream_context_cost_ms','other_private_work','four_output_typed_last_writers','terminal_notify_to_caller_R1','necessary_path_floor_ms','removable_product_ms'])
assert 'Run508 acquisition only' in sched['bound_effect'];ok('Run508 scoped zero-loop fields preserve all broader unknowns')
hardware=model['bound_ladder']['hardware_resource']['rate_boundary_certificate'];assert hardware['status']=='uncertified' and hardware['exact_board_C_plus'] is hardware['boundary_work_B'] is None
formula=model['bound_ladder']['product_e2e']['strict_outer_ceiling_sufficient_gate']['formula']
assert 'max(0,W_minus-B)/C_plus' in formula and 'W_minus > B' in formula and 'positive certified T_star' in formula
# Exact synthetic rational boundary sanity cases; no real work/capacity values supplied.
from fractions import Fraction
for W,B,C,expected in [(8,3,2,Fraction(5,2)),(3,3,2,0),(2,3,2,0),(8,0,2,4)]:assert Fraction(max(0,W-B),C)==expected
ok('conditional Resource boundary formula and positive-floor restriction')
assert 'run510/' not in raw.decode();ok('Run510 source-only evidence intentionally absent')
# Diagnostic type separation is preserved from V3.22; only explicitly reviewed fields change.
assert model['bound_ladder']['scheduling_execution']['run502_graph_dependency']==prior['bound_ladder']['scheduling_execution']['run502_graph_dependency']
assert model['bound_ladder']['algorithm_resource']==prior['bound_ladder']['algorithm_resource'];ok('historical trajectory and W-minus candidate not promoted')
# Hash rejection for every new pinned input, without touching real inputs.
original_hashes=m.HASHES.copy()
for path in original_hashes:
 m.HASHES={**original_hashes,path:'0'*64};reject('reject hash mismatch '+path,m.build)
m.HASHES=original_hashes
# Semantic negatives use disposable relocated evidence and explicitly synthetic hashes.
original_root=m.ROOT;original_build_prior=m.build_prior
with tempfile.TemporaryDirectory(prefix='SYNTHETIC_ONLY_',dir=OUT) as tmp:
 tmp=pathlib.Path(tmp)
 for path in original_hashes:
  dest=tmp/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/path).read_bytes())
 m.ROOT=tmp;m.build_prior=lambda:copy.deepcopy(prior)
 assert m.build()==model
 cases=[
 ('zero attention keys',m.UPDATE,lambda x:x['summaries'][0].update(attention_keys=0)),
 ('positive zip',m.UPDATE,lambda x:x['summaries'][0].update(zip_iterations=1)),
 ('nonempty events',m.UPDATE,lambda x:x['summaries'][0]['counts'].update(events=1)),
 ('missing row',m.UPDATE,lambda x:x['summaries'].pop()),
 ('wrong run UUID',m.UPDATE,lambda x:x.update(run_id='SYNTHETIC_WRONG')),
 ('missing rank scope',m.UPDATE,lambda x:x.update(ranks=7)),
 ('invalid final admission',m.FINAL,lambda x:x.update(valid=False)),
 ]
 for name,path,mut in cases:
  f=tmp/path;before=f.read_bytes();x=json.loads(before);mut(x);f.write_text(json.dumps(x));m.HASHES={**original_hashes,path:hashlib.sha256(f.read_bytes()).hexdigest()}
  reject('semantic reject '+name,m.build);f.write_bytes(before)
 m.HASHES=original_hashes
 path=m.CLEANUP;f=tmp/path;before=f.read_bytes();f.write_text(before.decode().replace('final_gate_exit=0','final_gate_exit=1'));m.HASHES={**original_hashes,path:hashlib.sha256(f.read_bytes()).hexdigest()};reject('semantic reject cleanup final gate',m.build)
 m.HASHES=original_hashes;m.ROOT=original_root;m.build_prior=original_build_prior
for key in sorted(e.NUMERIC_ENDPOINT_KEYS):reject('recursive reject endpoint '+key,lambda key=key:e.require_null_endpoints({'nested':[{'key':{key:1.0}}]}))
bad=copy.deepcopy(nodes);bad['matching_true_C_plus']['contract']='wrong';reject('DAG scope mismatch',lambda:d.validate_dag(bad))
bad=copy.deepcopy(nodes);bad['matching_true_C_plus']['requires_all']=['strict_outer_tps_ceiling'];reject('DAG cycle',lambda:d.validate_dag(bad))
# The following consistency checks capture initial feedback or its repaired state.
gate=model['bound_ladder']['product_e2e']['strict_outer_ceiling_sufficient_gate']
consistency={
 'formula_source_includes_Run509':'run509' in str(gate.get('source')),
 'capacity_or_scope_node_states_interval_boundary':any(('boundary' in s.lower() or 'cumulative' in s.lower() or 'W_minus > B' in s) for name in ['matching_true_C_plus','strict_scope_units_join'] for s in nodes[name]['missing']),
 'product_scope_checks_state_interval_boundary':any(('boundary' in s.lower() or 'cumulative' in s.lower() or 'W_minus > B' in s) for s in model['certificate_graph']['product_e2e']['strict_outer_ceiling']['scope_checks']),
}
for path,entry in inputs.items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==entry['sha256'],path
result={'verdict':'PASS' if all(consistency.values()) else 'PASS_NUMERIC_SCOPE_WITH_LEDGER_ALIGNMENT_FINDING','model_sha256':hashlib.sha256(raw).hexdigest(),'checks':checks,'check_count':len(checks),'proof_nodes':19,'certified_nodes':0,'formal_current_tok_s':571.681,'finite_endpoints':0,'null_endpoint_paths':endpoint_paths,'ledger_alignment':consistency,'input_count':len(inputs),'limitations':['Human-reviewed obligation ledger, not evidence-truth theorem prover','Named-field null guard is not an arbitrary numeric-field schema validator','No service, device, TaskCtl or model edits']}
(OUT/'audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'input_sha256_manifest.json').write_text(json.dumps({'inputs':inputs,'rechecked_unchanged':True},indent=2)+'\n')
print(json.dumps(result,indent=2))
