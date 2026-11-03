"""Audit the real factory's missing flag forwarding and save a non-executable patch.

This prepares reviewable source changes only; no install, import of the target,
device call, full-model run or assertion of whole-collector readiness.
"""
import ast,difflib,json
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
cache=N/'profile-entrypoints-cache-v162'
source=cache/'profiler_factory.py';text=source.read_text();tree=ast.parse(text)
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch_npu.profiler.profile']
assert len(calls)==1
keywords={k.arg:ast.unparse(k.value) for k in calls[0].keywords}
assert 'record_shapes' not in keywords
assert keywords['with_stack']=='False'
assert keywords['profile_memory']=='profiler_config.torch_profiler_with_memory'
assert keywords['with_modules']=='profiler_config.torch_profiler_with_stack'
config=(cache/'profiler_config.py').read_text()
assert 'torch_profiler_record_shapes: bool = False' in config
assert 'torch_profiler_with_memory: bool = False' in config
old='            with_stack=False,\n'
assert text.count(old)==1
new=text.replace(old,'            with_stack=True,\n            record_shapes=True,\n',1)
newtree=ast.parse(new)
assert len(newtree.body)==len(tree.body)
patch=''.join(difflib.unified_diff(text.splitlines(True),new.splitlines(True),
 fromfile='validated-image/torch_npu_profiler.py',tofile='temporary-diagnostic-only/torch_npu_profiler.py'))
p=N/'profile-entrypoints-cache-v162/builtin-field-route-only.patch';assert not p.exists();p.write_text(patch)
audit={
 'status':'READONLY_ENTRYPOINT_AUDIT_AND_FIELD_ROUTE_PATCH_PREPARED_NOT_INSTALLED',
 'source_read_receipt':pin(O/'profile-entrypoints-readonly-v162.json'),
 'profiler_factory':pin(source),'source_keyword_expressions':keywords,
 'config_has_record_shapes_flag_but_Ascend_factory_does_not_forward_it':True,
 'one_CLI_record_shapes_flag_alone_does_not_fix_this_factory':True,
 'field_route_patch':pin(p),
 'required_config_in_addition_to_patch':{'torch_profiler_with_memory':True},
 'two_fixed_patch_flags_match_successful_own_case':{'with_stack':True,'record_shapes':True},
 'no_claim_every_runtime_path_or_CANN_record_op_args_validated':True,
 'ordinary_service_factory_unchanged':True,
 'startup_capture':{
  'saved_runner_capture_model_delegates_to_GPUModelRunner':True,
  'regular_HTTP_start_profile_after_health_cannot_recreate_prior_CPU_producers':True,
  'needed':'A separate bounded observation point before graph creation,with explicit creation/update/replay identity and no nested incompatible profiler sessions.',
  'implemented_and_runtime_validated':False},
 'native_task_bridge':{
  'exact_SDK_declarations_already_bound':True,
  'own_probe_QUERY_IDS':0,
  'IDs_from_probe_are_not_a_failed_getter_or_actual_task0':True,
  'headers_alone_prove_call_error_state_or_all_bypass_coverage':False,
  'join_needs':['same process/context generation','actual device namespace bridge',
   'actual task/stream/correlation and launch-order/epoch association','capture-instance/update-to-replay bridge for captured tasks'],
  'nearest_timestamp_or_same_numeric_modelId_join_allowed':False},
 'full_case':{
  'same_frozen_client_sha256':pin(cache/'client.py')['sha256'],
  'same_serve_sha256':pin(cache/'serve.sh')['sha256'],
  'target_case':'c8-input32k-output1k-prefix90-171',
  'profile_samples_are_not_unprofiled_baseline_or_speedup_proof':True,
  'one_case_cohort_should_be_shared_across_modules_and_compatible_evidence_routes':True},
 'unnecessary_work_retired':['No repeated CANN header survey','No duplicate all-tensor logger',
  'No blanket per-pair reruns','No reexecution of the consumed69B probe'],
 'full_combined_model_collector_ready':False,
 'missing_implementation':['Explicit native task/capture correlation bridge',
  'Targeted offset/extent/epoch/value/reader supplements only for conclusion-changing cuts',
  'Startup graph-creation/update observation and globally enforced budgets/rollback'],
 'new_model_or_device_execution':False,'new_probe_authorization':False,
 'whole_modules_closed':0,'overall_complete':False}
write(O/'profile-field-route-readiness-v162.json',audit)
paths=[Path(__file__),N/'read_profile_entrypoints171_v162.py',O/'profile-entrypoints-readonly-v162.json',
 O/'profile-field-route-readiness-v162.json',p]
paths.extend(x for x in cache.iterdir() if x.is_file())
unique=sorted(set(paths))
manifest={'status':'PENDING_EVIDENCE_DELTA_NOT_NEW_CHECKPOINT_AUTHORITY',
 'parent_checkpoint':pin(O/'REVIEW-CHECKPOINT-v161.json'),
 'artifacts':[pin(x) for x in unique],
 'does_not_modify_sealed161_authority':True,'new_runtime_work':False,'overall_complete':False}
write(O/'pending-entrypoint-evidence-manifest-v162.json',manifest)
write(N/'NEXT-EVIDENCE-PENDING.json',{'current_checkpoint':161,
 'pending_manifest':pin(O/'pending-entrypoint-evidence-manifest-v162.json'),
 'next_action':'Implement only missing shared correlation/capture/reader facts,or explicitly narrow a future builtin-only diagnostic. Current partial patch is not full-model execution readiness.',
 'new_device_window_authorized':False,'overall_complete':False})
print(json.dumps({'status':audit['status'],'factory_missing_forwarding':'record_shapes',
 'new_native_work':False,'full_collector_ready':False,'pending_artifacts':len(unique)}))
