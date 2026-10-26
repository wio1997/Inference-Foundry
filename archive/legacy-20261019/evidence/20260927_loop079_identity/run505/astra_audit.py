import sys
sys.dont_write_bytecode = True
import copy, hashlib, importlib, json, os, subprocess, tempfile
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
OUT = ROOT / 'evidence/20260927_loop079_identity/run505'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts'))
import extreme_bound_calibration_v3_22 as m
from extreme_bound_calibration_v3_15 import validate_dag
from extreme_bound_calibration_v3_16_rev2 import NUMERIC_ENDPOINT_KEYS, require_null_endpoints
sha = lambda b: hashlib.sha256(b).hexdigest()
script_before = (ROOT/'scripts/extreme_bound_calibration_v3_22.py').read_bytes()
model_path = ROOT/'evidence/20260927_loop079_identity/run504/bound_calibration_v3_22.json'
model_before = model_path.read_bytes()
checks=[]
def check(name, ok, detail=None):
    checks.append({'test':name,'passed':bool(ok),'detail':detail})
def serialized(x): return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
model=m.build()
check('byte_identical_rebuild', serialized(model)==model_before)
check('formal_Current_unchanged', model['current']['accepted_formal_tps']==571.681)
resolved=validate_dag(model['proof_dag']['nodes'])
check('independent_proof_DAG_all19_false',len(resolved)==19 and not any(resolved.values()) and resolved==model['proof_dag']['certified'])
manifest=json.loads((ROOT/m.MANIFEST).read_text())
bad=[]
for entry in manifest:
    p=Path(entry['path'])
    if not p.exists():
        # The container-mounted backend source is archived verbatim in Run503.
        p=ROOT/'evidence/20260927_loop079_identity/run503/mla_v1.py'
    raw=p.read_bytes()
    if len(raw)!=entry['bytes'] or sha(raw)!=entry['sha256']:bad.append(entry['path'])
check('Run503_all145_manifest_entries',len(manifest)==145 and not bad,bad)
bl=model['bound_ladder']
check('new_resource_witness_conditional',bl['algorithm_resource']['first_position_work_witness']['status']=='conditional_scope_reduction_only' and bl['algorithm_resource']['first_position_work_witness']['compulsory_work_subset_operations'] is None)
d=bl['scheduling_execution']['run502_graph_dependency']
keys=['actual_update_iteration_count','external_event_handle_generation_join','four_output_typed_last_writers','terminal_notify_to_caller_R1','necessary_path_floor_ms','removable_product_ms']
check('new_scheduling_unresolved_values_null',all(d[k] is None for k in keys))
for field in ['memcpy_bytes','compulsory_traffic_bytes','exact_board_aggregate_capacity_upper']:
    check('hardware_null_'+field,bl['hardware_resource']['run502_graph_task_census'][field] is None)
original_hashes=dict(m.HASHES)
with tempfile.TemporaryDirectory(prefix='astra_run505_') as tmp:
    fixture=Path(tmp)
    for name in m.HASHES:
        p=fixture/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    m.ROOT=fixture
    check('minimal_eight_input_rebuild',serialized(m.build())==model_before)
    for name in m.HASHES:
        p=fixture/name;old=p.read_bytes();p.write_bytes(old+b' ')
        try:m.build();rejected=False
        except ValueError as e:rejected='hash mismatch' in str(e)
        check('hash_corruption_'+name,rejected);p.write_bytes(old)
    def negative(label,name,mutate):
        p=fixture/name;old=p.read_bytes();x=json.loads(old);mutate(x);p.write_bytes(serialized(x));m.HASHES[name]=sha(p.read_bytes())
        try:m.build();rejected=False;error=None
        except (ValueError,KeyError,TypeError) as e:rejected=True;error=str(e)
        check('semantic_'+label,rejected,error)
        p.write_bytes(old);m.HASHES[name]=original_hashes[name]
    negative('Current_changed',m.PRIOR,lambda x:x['current'].__setitem__('accepted_formal_tps',572))
    negative('certified_true',m.PRIOR,lambda x:x['proof_dag']['certified'].__setitem__('typed_necessary_path',True))
    negative('first_not_pass',m.FIRST,lambda x:x.__setitem__('source_identity','fail'))
    negative('first_patterns_missing',m.FIRST,lambda x:x.__setitem__('equality_patterns',127))
    negative('final_invalid',m.FINAL,lambda x:x.__setitem__('valid',False))
    negative('validation_invalid',m.VALID,lambda x:x.__setitem__('valid',False))
    negative('slices39',m.VALID,lambda x:x.__setitem__('slices',39))
    negative('ranks7',m.DUMP,lambda x:x.__setitem__('ranks',7))
    negative('meta_join_false',m.DUMP,lambda x:x.__setitem__('meta_join',False))
    negative('wrong_task_count',m.META,lambda x:x.__setitem__('node_count',5411))
    negative('wrong_stream_census',m.META,lambda x:x['stream_task_counts'].__setitem__('0',1039))
    negative('wrong_backend',m.META,lambda x:x['graph_update_backend']['impl']['source'].__setitem__('qualname','AscendDSAImpl'))
    negative('before_branch',m.META,lambda x:x['graph_update_backend']['selected_actual_update'].__setitem__('branch','before'))
    negative('wrong_private_stream',m.META,lambda x:x['graph_update_backend']['selected_actual_update']['configured_update_stream'].__setitem__('stream_id',0))
    negative('no_selected_return',m.META,lambda x:x['graph_update_backend']['selected_actual_update'].__setitem__('return_count',0))
    for key in sorted(NUMERIC_ENDPOINT_KEYS):
        negative('endpoint_'+key,m.PRIOR,lambda x,k=key:x.__setitem__(k,1.0))
    # Characterize the imported guard's limits without changing the frozen build.
    uncovered=[]
    for key in ['necessary_path_floor_ms','removable_product_ms','compulsory_work_subset_operations','exact_board_aggregate_capacity_upper']:
        try:require_null_endpoints({key:1.0});uncovered.append(key)
        except ValueError:pass
    checks.append({'test':'generic_guard_new_key_coverage','passed':None,'detail':{'not_rejected':uncovered,'scope':'The fixed generator explicitly overwrites these new fields with None; this is not an arbitrary future-model validator.'}})
    mods=['extreme_bound_calibration_v3_22.py','extreme_bound_calibration_v3_16_rev2.py','extreme_bound_calibration_v3_15.py']
    (fixture/'scripts').mkdir()
    for name in mods:(fixture/'scripts'/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    r=subprocess.run([sys.executable,str(fixture/'scripts'/mods[0]),'--output',str(fixture/'rebuilt.json')],capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    check('portable_CLI_reproduction',r.returncode==0 and (fixture/'rebuilt.json').read_bytes()==model_before,{'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
m.ROOT=ROOT
check('reviewed_script_and_model_unchanged',script_before==(ROOT/'scripts/extreme_bound_calibration_v3_22.py').read_bytes() and model_before==model_path.read_bytes())
result={'checks':checks,'all_expected':all(c['passed'] is not False for c in checks),'script_sha256':sha(script_before),'model_sha256':sha(model_before),'direct_input_sha256':dict(m.HASHES),'imported_module_sha256':{n:sha((ROOT/'scripts'/n).read_bytes()) for n in ['extreme_bound_calibration_v3_16_rev2.py','extreme_bound_calibration_v3_15.py']}}
(OUT/'cpu_review_results.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'reviewed_generator.py').write_bytes(script_before)
print(json.dumps(result,indent=2))
