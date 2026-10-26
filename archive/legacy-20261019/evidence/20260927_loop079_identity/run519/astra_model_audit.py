#!/usr/bin/env python3
"""Independent V3.25 CPU audit; only Run519 artifacts are written."""
import copy,hashlib,importlib,json,pathlib,subprocess,sys,tempfile
sys.dont_write_bytecode=True
R=pathlib.Path('/data/wio/Inference_Foundry');O=R/'evidence/20260927_loop079_identity/run519';O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'scripts'))
m=importlib.import_module('extreme_bound_calibration_v3_25');p=importlib.import_module('extreme_bound_calibration_v3_24');e=importlib.import_module('extreme_bound_calibration_v3_16_rev2');d=importlib.import_module('extreme_bound_calibration_v3_15')
inputs={};checks=[]
def read(p):
 p=pathlib.Path(p);b=p.read_bytes();inputs[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return b
def ok(s):checks.append({'check':s,'pass':True})
def reject(s,f):
 try:f()
 except (ValueError,KeyError,AssertionError):ok(s);return
 raise AssertionError('unexpected acceptance '+s)
S=R/'scripts/extreme_bound_calibration_v3_25.py';T=R/'evidence/20260927_loop079_identity/run518/bound_calibration_v3_25.json'
assert hashlib.sha256(read(S)).hexdigest()==sys.argv[1];assert hashlib.sha256(read(T)).hexdigest()==sys.argv[2]
for path,h in {**p.HASHES,**m.HASHES}.items():assert hashlib.sha256(read(R/path)).hexdigest()==h
ok('supplied final script/output hashes and V3.24/V3.25 direct evidence pins')
model=m.build();raw=(json.dumps(model,ensure_ascii=False,indent=2)+'\n').encode();assert raw==T.read_bytes()
cp=subprocess.run([sys.executable,'-B',str(S),'--output',str(O/'rebuild.json')],cwd='/tmp',capture_output=True,text=True);assert cp.returncode==0 and (O/'rebuild.json').read_bytes()==raw
ok('build and absolute-path CLI byte-identical rebuild')
prior=json.loads(read(R/m.PRIOR));inventory=json.loads(read(R/m.INVENTORY));assert model['model_revision']=='V3.25' and model['current']==prior['current'] and model['current']['accepted_formal_tps']==571.681 and model['contract']==prior['contract']
ok('entire Current and frozen contract unchanged')
assert len(inventory['inputs'])==132
for path,h in inventory['inputs'].items():assert hashlib.sha256(read(R/path)).hexdigest()==h
for path,h in json.loads(read(R/m.MANIFEST)).items():assert hashlib.sha256(read(path)).hexdigest()==h['sha256']
for path,h in json.loads(read(R/m.LOCATIONS))['inputs'].items():assert hashlib.sha256(read(path)).hexdigest()==h['sha256']
ok('132 formal inputs and expanded152-input/15-source manifests verified')
nodes=model['proof_dag']['nodes'];resolved=d.validate_dag(nodes);assert len(nodes)==19 and resolved==model['proof_dag']['certified'] and all(v is False for v in resolved.values())
ok('independent DAG recomputation: all19 nodes false')
endpoint_paths=[]
def walk(x,path=''):
 if isinstance(x,dict):
  for k,v in x.items():
   q=path+'.'+k
   if k in e.NUMERIC_ENDPOINT_KEYS:assert v is None;endpoint_paths.append(q)
   walk(v,q)
 elif isinstance(x,list):
  for i,v in enumerate(x):walk(v,path+f'[{i}]')
walk(model);e.require_null_endpoints(model);ok('all recognized endpoint fields null')
a=model['bound_ladder']['algorithm_resource']['run516_formal_saved_witness_inventory']
assert a['status']=='accepted_saved_artifacts_insufficient' and 'inspected accepted Run99 files only' in a['scope'] and 'no claim that required work did not occur' in a['scope']
assert a['positive_in_window_W_minus_operations'] is None and all(a[k] is False for k in ['saved_client_to_Runtime_request_id_join','saved_same_generation_first_Target_token_to_external_raw_token_join','saved_scheduler_pre_append_G','saved_fresh_typed_wo_a_group_witness'])
assert inventory['request_id_join_from_saved_client_to_runtime'] is False and inventory['positive_formal_window_W_minus_certified'] is False
ok('saved insufficiency only; no universal work absence or numeric numerator')
for key in ['hardware_resource','scheduling_execution','product_e2e']:assert model['bound_ladder'][key]==prior['bound_ladder'][key]
assert model['certificate_graph']['hardware_resource']==prior['certificate_graph']['hardware_resource'] and model['certificate_graph']['scheduling_execution']==prior['certificate_graph']['scheduling_execution']
ok('hardware/scheduling/Product gates and Run502/508 source scopes unchanged')
changes=[]
def diff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:changes.append(path+'.'+k)
   else:diff(a[k],b[k],path+'.'+k)
 elif a!=b:changes.append(path)
diff(prior,model)
assert set(changes)=={'.model_revision','.bound_ladder.algorithm_resource.run516_formal_saved_witness_inventory','.certificate_graph.algorithm_resource.run516_formal_saved_witness_inventory','.proof_dag.nodes.positive_unavoidable_W_minus.evidence','.proof_dag.nodes.positive_unavoidable_W_minus.missing','.next_measurement.priority','.next_measurement.specific_gate','.input_paths'}
ok('only eight intended ledger locations changed')
priority=model['next_measurement']['priority']
assert 'matching exact-board C-plus with a cumulative interval-service guarantee, or with a proved boundary B and W-minus > B' in priority
assert 'add mixed all8 DAG when the claimed endpoint requires it' in priority
assert 'q<R, G+q<1024' in model['next_measurement']['specific_gate']
ok('next action preserves C-plus in both cases, positive boundary margin, and endpoint-scoped DAG')
orig_hash=m.HASHES.copy();orig_root=m.ROOT;orig_build=m.build_prior
for path in orig_hash:
 m.HASHES={**orig_hash,path:'0'*64};reject('reject new pinned hash mismatch '+path,m.build)
m.HASHES=orig_hash
with tempfile.TemporaryDirectory(prefix='SYNTHETIC_ONLY_',dir=O) as tmp:
 tmp=pathlib.Path(tmp)
 for path in orig_hash:
  q=tmp/path;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/path).read_bytes())
 m.ROOT=tmp;m.build_prior=lambda:copy.deepcopy(prior);assert m.build()==model
 f=tmp/m.INVENTORY;before=f.read_bytes()
 for label,mut in [('invalid',lambda x:x.update(valid=False)),('repeat count',lambda x:x.update(formal_client_repeats=2)),('request count',lambda x:x.update(requests_per_repeat=12)),('rank files',lambda x:x.update(runtime_rank_cohort_files=40)),('missing input',lambda x:x['inputs'].pop(next(iter(x['inputs'])))),('positive W certificate',lambda x:x.update(positive_formal_window_W_minus_certified=True)),('present witness alias',lambda x:x['witness_fields_present'].update(client_request_id=True))]:
  x=json.loads(before);mut(x);f.write_text(json.dumps(x));m.HASHES={**orig_hash,m.INVENTORY:hashlib.sha256(f.read_bytes()).hexdigest()};reject('semantic inventory reject '+label,m.build);f.write_bytes(before)
 m.HASHES=orig_hash;f=tmp/m.REVIEW;before=f.read_bytes()
 for anchor in ['**PASS for the revised frozen Run516 inventory and its limited conclusion:**','**not universal impossibility']:
  f.write_bytes(before.replace(anchor.encode(),b'SYNTHETIC_REMOVED'));m.HASHES={**orig_hash,m.REVIEW:hashlib.sha256(f.read_bytes()).hexdigest()};reject('semantic review qualification reject '+anchor,m.build);f.write_bytes(before)
 m.HASHES=orig_hash
 for label,mut in [('Current',lambda x:x['current'].update(accepted_formal_tps=999)),('revision',lambda x:x.update(model_revision='WRONG')),('proof promotion',lambda x:x['proof_dag']['certified'].update(positive_unavoidable_W_minus=True)),('finite endpoint',lambda x:x['bound_ladder']['product_e2e'].update(finite_tps_upper_bound=999))]:
  x=copy.deepcopy(prior);mut(x);f=tmp/m.PRIOR;f.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');m.HASHES={**orig_hash,m.PRIOR:hashlib.sha256(f.read_bytes()).hexdigest()};m.build_prior=lambda:copy.deepcopy(x);reject('semantic prior reject '+label,m.build)
 m.HASHES=orig_hash;m.ROOT=orig_root;m.build_prior=orig_build
for key in sorted(e.NUMERIC_ENDPOINT_KEYS):reject('nested endpoint reject '+key,lambda key=key:e.require_null_endpoints({'nested':[{key:123}]}))
bad=copy.deepcopy(nodes);bad['positive_unavoidable_W_minus']['contract']='WRONG';reject('DAG wrong contract',lambda:d.validate_dag(bad))
bad=copy.deepcopy(nodes);bad['positive_unavoidable_W_minus']['requires_all']=['strict_outer_tps_ceiling'];reject('DAG cycle',lambda:d.validate_dag(bad))
assert m.build()==model
for name,mod in list(sys.modules.items()):
 if name.startswith('extreme_bound_calibration_') and getattr(mod,'__file__',None):read(mod.__file__)
for path,h in inputs.items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==h['sha256']
result={'verdict':'PASS','script_sha256':inputs[str(S)]['sha256'],'model_sha256':hashlib.sha256(raw).hexdigest(),'checks':checks,'check_count':len(checks),'input_count':len(inputs),'proof_nodes':19,'certified_nodes':0,'formal_current_tok_s':571.681,'finite_endpoints':0,'null_endpoint_paths':endpoint_paths,'changed_paths':changes,'next_measurement':model['next_measurement'],'limits':['frozen human-reviewed ledger; not evidence truth prover','named-field endpoint guards and exact evidence pins','no service/NPU/model/TaskCtl edits']}
(O/'audit_results.json').write_text(json.dumps(result,indent=2)+'\n');(O/'input_sha256_manifest.json').write_text(json.dumps({'inputs':inputs,'rechecked_unchanged':True},indent=2,sort_keys=True)+'\n');print(json.dumps({k:result[k] for k in ['verdict','check_count','input_count','script_sha256','model_sha256','next_measurement']},indent=2))
