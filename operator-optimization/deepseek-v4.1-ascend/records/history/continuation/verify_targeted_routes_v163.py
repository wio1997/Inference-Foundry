"""Independent new-route/source/scope review; no target execution or old suites."""
import ast,json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import N,O,T as TASK,read,write
from analyze_evidence_batches_v155 import pin
PRE='--preflight' in sys.argv;gates=[]
def check(ok,label):
    assert ok,label
    gates.append(label)
def valid(row):
    b=Path(row['path']).read_bytes()
    assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
manifest=O/('checkpoint-manifest-v161.json' if PRE else 'checkpoint-manifest-v163.json')
sealed=read(manifest)
for r in sealed['artifacts']:valid(r)
check(True,'Predecessor/current sealed authority pins match')
inputs=read(O/'targeted-routes-inputs-v163.json')['inputs']
for r in inputs:valid(r)
check(True,'All new immutable sources and CPU-result pins match')
root=N/'targeted-evidence-v163';cache=N/'profile-entrypoints-cache-v163'
sources=read(O/'graph-entrypoint-sources-v163.json')
for r in sources['files']:valid(dict(path=r['local_path'],bytes=r['bytes'],sha256=r['sha256']))
check(sources['no_runtime_calls'] and len(sources['files'])==3,'Three exact own-container source paths read without importing/running target modules')
factory=(N/'profile-entrypoints-cache-v162/profiler_factory.py').read_text();tree=ast.parse(factory)
call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch_npu.profiler.profile')
kw={k.arg:ast.unparse(k.value) for k in call.keywords}
check('record_shapes' not in kw and kw['profile_memory']=='profiler_config.torch_profiler_with_memory',
 'Existing record_shapes forwarding gap and separate memory requirement independently checked')
source=(cache/'compilation/updatable_graph.py').read_text();tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UpdatableGraph')
methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
resource=methods['get_capture_resource'];calls=[n for n in ast.walk(resource) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='factory']
check(len(calls)==2 and 'if use_max_workspace:' in ast.get_source_segment(source,resource),
 'Exact source permits distinct initial and max-candidate factory invocations')
update=ast.get_source_segment(source,methods['update'])
check('ws_buffer: dict[Hashable, Any] = {}' in update and 'task.operation == torch_npu._npu_paged_attention' in update and 'task.kwargs["workspace"] = ws_buffer[ws_key]' in update,
 'PA sharing is conditional and cache-local to one update call;not generalized to the configured model')
check('runtime_kwargs = {**self.kwargs, **params}' in source and 'self.event.record(update_stream)' in source and 'weak_ref_tensors(self.capture_resources)' in source,
 'Provider override,event-record and weak-reference boundaries do not attest complete epochs/retirement')
contract=read(O/'capture-workspace-epoch-contract-v163.json')
check(contract['PA_workspace_cache']['actual_DeepSeek_V41_branch_selected'] is None and not contract['physical_storage_or_time_or_MFU_bound_proved'],
 'No generic PA reachability,physical bound or independence promotion')

runtime=ast.parse((root/'targeted_runtime.py').read_text())
pin_node=next(n for n in runtime.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='PINNED' for x in n.targets))
pins=ast.literal_eval(pin_node.value)
for name,path in [('vllm_ascend.compilation.updatable_graph',cache/'compilation/updatable_graph.py'),
 ('vllm_ascend.compilation.acl_graph',cache/'compilation/acl_graph.py'),
 ('vllm_ascend.profiler.torch_npu_profiler',N/'profile-entrypoints-cache-v162/profiler_factory.py')]:
    assert pins[name]==pin(path)['sha256']
check(True,'Runtime adapter literals match independently cached exact module hashes')
obs=next(n for n in runtime.body if isinstance(n,ast.ClassDef) and n.name=='TargetedObserver')
init=next(n for n in obs.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
check(ast.literal_eval(init.args.defaults[-1]) is False,'Torch dispatch observer defaults off')
for file in ['evidence_core.py','targeted_runtime.py','test_targeted_cpu.py']:
    compile((root/file).read_bytes(),str(root/file),'exec')
check(True,'New adapter/core/CPU-fixture modules compile under Python3.12 without importing Torch/CANN')
for n in ast.walk(runtime):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
        assert n.func.attr not in ['item','cpu','numpy','tolist','synchronize','empty_cache','memcpy','string_at']
check(True,'Supplementary runtime contains no Tensor-content transfer,device sync or pointer dereference calls')
cpu=read(root/'cpu-contract-result.json')
check(cpu['status']=='PASS_CPU_CONTRACTS_ONLY' and cpu['gate_count']==35 and not cpu['real_Torch_dispatch_tested'] and not cpu['real_CANN_tested'],
 '35 new CPU contracts passed;exact-source fixtures are not claimed as real runtime validation')
check(any('Failed native begin' in x for x in cpu['gates']) and any('Failed native end' in x for x in cpu['gates']) and any('Original exception object' in x for x in cpu['gates']),
 'Failure/cleanup/epoch and exactly-once contracts explicitly checked')
docs=read(O/'primary-query-epoch-docs-v163.json')
for r in docs['sources']:valid(r)
doc=(root/'primary-docs/last-task-explicit-stream-920beta2.html').read_text()
check('用户显式创建' in doc and 'Stream' in doc,'Official task query document limits the named task domain to an explicit stream')
doc=(root/'primary-docs/inference-mode-210.html').read_text()
check('Inference tensors do not track version counter.' in doc,'Official PyTorch2.10 document supports preserving missing inference version as unknown')

sys.path.insert(0,str(root))
from evidence_core import join_native_candidates
own=read(O/'joint-profiler-capability-observation-v161.json')['native_host_packet']
own=dict(own,task_id_status='NOT_QUERIED',stream_id_status='NOT_QUERIED')
check(join_native_candidates(own,[])['status']=='UNKNOWN','Actual saved own packet remains unjoined;new code does not backfill its sentinels')
ready=read(O/'targeted-observer-readiness-v163.json')
check(not ready['default_TorchDispatchMode_enabled'] and not ready['actual_Torch_NPU_dispatch_or_capture_compatibility_tested'] and not ready['native_JOIN_API_witness_producer_implemented'] and not ready['whole_combined_model_collector_ready'],
 'Real dispatch/capture/native-witness gaps remain explicit')
check(ready['does_not_block_independently_ready_builtin_profile_acquisition'],
 'Independent builtin acquisition preparation is not blocked by unrelated native gaps')
summary=read(O/'targeted-routes-summary-v163.json')
check(summary['module_inventory']==26 and summary['semantic_rows']==358 and not summary['overall_complete'] and not summary['inventory_exhaustiveness_proved'] and summary['whole_modules_closed']==0,
 'Original full scope and all-module completion/physical ceilings remain open')
if not PRE:
    cp=read(O/'REVIEW-CHECKPOINT-v163.json');ob=read(Path(cp['coverage_obligations']))
    check({r['id'] for r in ob['obligations']}=={f'G{i:02d}' for i in range(15)},'All15 full-scope obligations preserved')
    index=read(TASK/'index.json');loop=read(TASK/'loops/loop-005/state.json')
    check(index['active_loop']=='loop-005' and loop['status']=='EVALUATING' and index['status']=='ACTIVE',
     'Correct derived171 Loop active/evaluating;no frozen old Loop rewritten')
result={'status':'PASS','checkpoint':163,'preflight':PRE,'gate_count':len(gates),'gates':gates,
 'sealed_manifest':pin(manifest),'sealed_manifest_files':len(sealed['artifacts']),
 'input_pins_verified':len(inputs),'verification_scope':'New source/adapter contract integrity and failure/UNKNOWN boundaries;not whole-model/native/mathematical completion.',
 'old_unchanged_operator_case_suites_reexecuted':False,'new_device_work':False,
 'whole_modules_closed':0,'overall_complete':False,'observed_utc':datetime.now(timezone.utc).isoformat(),'verifier':pin(Path(__file__))}
write(O/('targeted-routes-preflight-v163.json' if PRE else 'targeted-routes-postcheck-result-v163p.json'),result)
if not PRE:write(N/'CURRENT-POSTCHECK.json',result)
print(json.dumps({'status':'PASS','preflight':PRE,'gate_count':len(gates),'sealed_pins':len(sealed['artifacts']),'overall_complete':False}))
