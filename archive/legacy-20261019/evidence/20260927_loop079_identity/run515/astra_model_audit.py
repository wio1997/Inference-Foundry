#!/usr/bin/env python3
"""Independent CPU-only V3.24 audit. Writes only Run515 evidence."""
import copy,hashlib,importlib,json,pathlib,subprocess,sys,tempfile
sys.dont_write_bytecode=True
ROOT=pathlib.Path('/data/wio/Inference_Foundry'); OUT=ROOT/'evidence/20260927_loop079_identity/run515';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts'))
m=importlib.import_module('extreme_bound_calibration_v3_24')
p=importlib.import_module('extreme_bound_calibration_v3_23')
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
script=ROOT/'scripts/extreme_bound_calibration_v3_24.py';target=ROOT/'evidence/20260927_loop079_identity/run514/bound_calibration_v3_24.json'
assert hashlib.sha256(read(script)).hexdigest()=='decd4150396617ae7ec07c53bee4af1fdd00694e021fc988e96cda9e2afd4402'
assert hashlib.sha256(read(target)).hexdigest()=='71956b009bb160cf562d7135f9a963262e2088ab30e474596869f6bcaa3f985a'
for path,digest in {**p.HASHES,**m.HASHES}.items():assert hashlib.sha256(read(ROOT/path)).hexdigest()==digest
for name in ['extreme_bound_calibration_v3_23.py','extreme_bound_calibration_v3_16_rev2.py','extreme_bound_calibration_v3_15.py']:read(ROOT/'scripts'/name)
ok('frozen script/model and all direct V3.23/V3.24 pins match')
model=m.build();raw=(json.dumps(model,ensure_ascii=False,indent=2)+'\n').encode();assert raw==read(target)
cmd=[sys.executable,'-B',str(script),'--output',str(OUT/'rebuild.json')]
cp=subprocess.run(cmd,cwd='/tmp',capture_output=True,text=True);assert cp.returncode==0,cp.stderr
assert (OUT/'rebuild.json').read_bytes()==raw;ok('absolute-path CLI and build produce byte-identical model')
prior=json.loads(read(ROOT/m.PRIOR));assert model['contract']==prior['contract'] and model['current']==prior['current'] and model['current']['accepted_formal_tps']==571.681
ok('unchanged formal scope and Current571.681')
nodes=model['proof_dag']['nodes'];resolved=d.validate_dag(nodes);assert len(nodes)==19 and resolved==model['proof_dag']['certified'] and all(v is False for v in resolved.values())
assert set(nodes)==set(prior['proof_dag']['nodes']);ok('independently recomputed 19 proof nodes all false')
endpoint_paths=[]
def walk(x,path=''):
 if isinstance(x,dict):
  for k,v in x.items():
   q=path+'.'+k
   if k in e.NUMERIC_ENDPOINT_KEYS:assert v is None;endpoint_paths.append(q)
   walk(v,q)
 elif isinstance(x,list):
  for i,v in enumerate(x):walk(v,path+f'[{i}]')
walk(model);e.require_null_endpoints(model);ok('all recognized endpoint occurrences null')
s=model['bound_ladder']['scheduling_execution']['run510_terminal_and_MIX_source']
assert s['status']=='version_matched_general_contract_only' and s['placeholder_label_consumes_argument_bytes'] is False
assert s['run502_reduce_mean_aux_candidate_raw_word']==3 and s['run502_reduce_mean_aux_candidate_raw_byte_offset']==24
for k in ['run502_reduce_mean_aux_typed_y_role','run508_loaded_runtime_branch_and_build_join','run508_terminal_notify_object_join','run508_four_output_last_writers','run508_caller_R1_completion_join','necessary_path_floor_ms']:assert s[k] is None
assert 'Stars/non-AICPU branches' in s['release_mechanism'] and 'synthetic' in s['exporter_timing'] and 'no native/run-specific completion' in s['bound_effect']
assert not any(('run508' in k and ('byte' in k or 'word' in k)) for k in s)
ok('Run502 byte24 remains candidate; no Run508 ABI/identity transfer')
for k in ['run502_graph_dependency','run508_selected_MLA_update']:assert model['bound_ladder']['scheduling_execution'][k]==prior['bound_ladder']['scheduling_execution'][k]
for k in ['algorithm_resource','hardware_resource','product_e2e']:assert model['bound_ladder'][k]==prior['bound_ladder'][k]
ok('historical observations, Resource gates and Product formula unchanged')
changes=[]
def diff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:changes.append(path+'.'+k)
   else:diff(a[k],b[k],path+'.'+k)
 elif a!=b:changes.append(path)
diff(prior,model)
assert set(changes)=={'.model_revision','.bound_ladder.scheduling_execution.run510_terminal_and_MIX_source','.certificate_graph.scheduling_execution.run510_terminal_and_MIX_source','.proof_dag.nodes.typed_necessary_path.evidence','.proof_dag.nodes.typed_necessary_path.missing','.next_measurement.priority','.next_measurement.specific_gate','.input_paths'}
ok('only eight intended ledger locations changed')
# Expand pinned Run510 manifests; leaf integrity is checked here, not inferred from manifest text.
src=json.loads(read(ROOT/m.SOURCE_MANIFEST));base=ROOT/'evidence/20260927_loop079_identity/run510'
for item in src['files']:assert hashlib.sha256(read(base/item['archive'])).hexdigest()==item['sha256']
old=json.loads(read(base/'review_input_hashes.json'))
for path,h in old.items():assert hashlib.sha256(read(path)).hexdigest()==h
assert len(src['files'])==22 and len(old)==32;ok('22 release source files and 32 Run510 underlying inputs still match')
original_hashes=m.HASHES.copy();original_root=m.ROOT;original_build=m.build_prior
for path in original_hashes:
 m.HASHES={**original_hashes,path:'0'*64};reject('reject hash mismatch '+path,m.build)
m.HASHES=original_hashes
with tempfile.TemporaryDirectory(prefix='SYNTHETIC_ONLY_',dir=OUT) as tmp:
 tmp=pathlib.Path(tmp)
 for path in original_hashes:
  q=tmp/path;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/path).read_bytes())
 m.ROOT=tmp;m.build_prior=lambda:copy.deepcopy(prior)
 assert m.build()==model;ok('relocated synthetic control baseline matches')
 review=tmp/m.REVIEW;before=review.read_bytes()
 for label,anchor in [('admission','**PASS for the source findings below; NOT a complete Run502 typed-last-writer or Scheduling certificate.**'),('native conditional','**Keep conditional for Run502:**'),('ABI exclusion','not a certified decoder')]:
  review.write_bytes(before.replace(anchor.encode(),b'SYNTHETIC_REMOVED'))
  m.HASHES={**original_hashes,m.REVIEW:hashlib.sha256(review.read_bytes()).hexdigest()}
  reject('semantic reject missing '+label,m.build)
 review.write_bytes(before);m.HASHES=original_hashes
 for label,mut in [('Current',lambda x:x['current'].update(accepted_formal_tps=999)),('prior revision',lambda x:x.update(model_revision='BAD')),('certified proof',lambda x:x['proof_dag']['certified'].update(typed_necessary_path=True)),('numeric endpoint',lambda x:x['bound_ladder']['product_e2e'].update(finite_tps_upper_bound=999))]:
  altered=copy.deepcopy(prior);mut(altered);f=tmp/m.PRIOR;f.write_text(json.dumps(altered,ensure_ascii=False,indent=2)+'\n');m.HASHES={**original_hashes,m.PRIOR:hashlib.sha256(f.read_bytes()).hexdigest()};m.build_prior=lambda:copy.deepcopy(altered)
  reject('semantic reject prior '+label,m.build)
 m.ROOT=original_root;m.HASHES=original_hashes;m.build_prior=original_build
for key in sorted(e.NUMERIC_ENDPOINT_KEYS):reject('reject nested endpoint '+key,lambda key=key:e.require_null_endpoints({'nested':[{'deeper':{key:1}}]}))
bad=copy.deepcopy(nodes);bad['typed_necessary_path']['contract']='WRONG';reject('reject DAG contract mismatch',lambda:d.validate_dag(bad))
bad=copy.deepcopy(nodes);bad['typed_necessary_path']['requires_all']=['scheduling_relaxed_latency_floor'];reject('reject DAG cycle',lambda:d.validate_dag(bad))
assert m.build()==model;ok('production in-memory module state restored')
for module_name,module in list(sys.modules.items()):
 if module_name.startswith('extreme_bound_calibration_') and getattr(module,'__file__',None):read(module.__file__)
read(ROOT/'evidence/20260927_loop079_identity/run513/astra_v3_23_review.md')
for path,item in inputs.items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==item['sha256']
result={'verdict':'PASS','check_count':len(checks),'checks':checks,'script_sha256':inputs[str(script)]['sha256'],'model_sha256':hashlib.sha256(raw).hexdigest(),'formal_current_tok_s':571.681,'proof_nodes':19,'certified_nodes':0,'finite_endpoints':0,'null_endpoint_paths':endpoint_paths,'changed_paths':changes,'input_count':len(inputs),'cli':{'command':cmd,'cwd':'/tmp','exit':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr},'limitations':['Human-reviewed obligation ledger, not evidence-truth prover','Named endpoint guard is not arbitrary numeric schema','Version-matched source is not installed build/branch proof','No service/NPU/shared model/script/TaskCtl edits']}
(OUT/'audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'input_sha256_manifest.json').write_text(json.dumps({'inputs':inputs,'rechecked_unchanged':True},indent=2)+'\n')
print(json.dumps({'verdict':result['verdict'],'checks':len(checks),'inputs':len(inputs),'endpoints':len(endpoint_paths),'changed_paths':changes},indent=2))
