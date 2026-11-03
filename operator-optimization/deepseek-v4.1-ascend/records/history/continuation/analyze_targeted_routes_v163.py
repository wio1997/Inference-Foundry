"""Reduce exact source and new CPU contracts into scoped operator evidence routes."""
import ast,json,sys
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
assert not (O/'REVIEW-CHECKPOINT-v163.json').exists()
cache=N/'profile-entrypoints-cache-v163';root=N/'targeted-evidence-v163'
g=ast.parse((cache/'compilation/updatable_graph.py').read_text())
classes={n.name:n for n in g.body if isinstance(n,ast.ClassDef)}
methods={n.name:n for n in classes['UpdatableGraph'].body if isinstance(n,ast.FunctionDef)}
def segment(node):return ast.get_source_segment((cache/'compilation/updatable_graph.py').read_text(),node)
contract={
 'status':'EXACT_INSTALLED_SOURCE_CONDITIONAL_CONTRACT_NOT_RUNTIME_BRANCH_ATTESTATION',
 'source_receipt':pin(O/'graph-entrypoint-sources-v163.json'),
 'affected_modules':['M04','M10','M21','M22','M24'],
 'capture_resources':{
  'source_range':[methods['get_capture_resource'].lineno,methods['get_capture_resource'].end_lineno],
  'first_missing_key_with_use_max_workspace_can_call_factory_twice':True,
  'same_existing_key_without_max_can_return_same_cached_Tensor':True,
  'max_candidate_can_replace_cached_Tensor':True,
  'capture_end_rewrites_resource_references_with_weak_ref_tensors':True,
  'implication':'Do not sum all factory candidates as simultaneously live workspaces or infer retirement from a weak Python reference. Actual roles/overlap/lifetimes need same-run evidence.'},
 'resolved_update_parameters':{
  'bind_overlay':'runtime_kwargs = {**self.kwargs, **params}',
  'provider_resolved_once_per_provider_then_selected_by_provider_index':True,
  'implication':'Captured initial kwargs alone are insufficient for every replay epoch; source permits provider overrides. The observer must not invoke resolve/factory/operation a second time.'},
 'PA_workspace_cache':{
  'scope_guard':'Only tasks whose operation == torch_npu._npu_paged_attention inside an actually selected UpdatableGraph.update path.',
  'fresh_ws_buffer_each_update_call':True,'same_key_assigns_same_workspace_Tensor_in_that_call':True,
  'key_fields':['context_lens.data_ptr','context_lens.shape','query shape/dtype','key_cache shape/dtype',
   'value_cache shape/dtype','block_table shape/dtype','out shape/dtype','num_kv_heads','num_heads','scale_value'],
  'does_not_encode_all_parameter_values_pointers_or_storage_generations':True,
  'actual_DeepSeek_V41_branch_selected':None,
  'implication':'This is a conditional alias/rebinding source contract,not independent buffers or a new fusion credit. Do not spend per-pair work on this generic PA route before the actual backend/guard is recorded.'},
 'completion':{
  'source_order':'graph_task_update_begin -> operation -> graph_task_update_end -> event.record',
  'source_host_return_or_event_record_proves_GPU_reader_completion':False,
  'weak_resource_reference_is_all_reader_retirement':False},
 'current_config_scope':{
  'raw_checkpoint_has_compress_ratios_under_text_config':True,
  'raw_nested_config_proves_every_live_hf_config_attribute':False,
  'must_observe':['runner.use_compress','runner.use_sparse','actual attn_backend MRO',
   'wrapper runtime/forward mode','enable_enpu/use_eagle','same graph Python instance/capture state'],
  'guard_snapshot_is_prepared_not_collected':True},
 'new_historical_native_promotions':0,'new_independence_proofs':0,
 'physical_storage_or_time_or_MFU_bound_proved':False,'whole_modules_closed':0,'overall_complete':False}
write(O/'capture-workspace-epoch-contract-v163.json',contract)
cpu=read(root/'cpu-contract-result.json')
readiness={
 'status':'BOUNDED_TARGETED_ADAPTERS_IMPLEMENTED_CPU_CONTRACT_CHECKED_NOT_DEPLOYED',
 'sources':{p.name:pin(p) for p in [root/'evidence_core.py',root/'targeted_runtime.py',root/'test_targeted_cpu.py']},
 'CPU_contract_result':pin(root/'cpu-contract-result.json'),'CPU_contract_gates':cpu['gate_count'],
 'implemented':['Supplementary storage offset/extent/version metadata without Tensor contents',
  'Exact source guards and actual runner/backend guard snapshot entry',
  'Python graph identity and separate begin/end success/failure epoch states',
  'Original-operation/factory/provider single-call forwarding with inherited descriptor restore',
  'Bounded sink,explicit drop/footer and no allocation-generation/reader-retirement invention',
  'No-time-nearest native join predicate that requires independently verified identity/witness facts'],
 'limits':{'records_per_process':65536,'bytes_per_process':67108864,'host_sequence_limit':64,
  'optional_dispatch_selector':'constant_pad_nd OR smallBOOL input/output <=4096 elements'},
 'default_TorchDispatchMode_enabled':False,
 'reason_for_dispatch_default_off':'Real Torch/Dynamo/compiled/capture compatibility has not been measured. A fake CPU mode is not that evidence.',
 'actual_Torch_NPU_dispatch_or_capture_compatibility_tested':False,
 'native_JOIN_API_witness_producer_implemented':False,
 'native_join_status_flags_are_not_self_authenticating_proof':True,
 'private_CANN_error_state_or_all_bypass_semantics_proved':False,
 'whole_combined_model_collector_ready':False,
 'does_not_block_independently_ready_builtin_profile_acquisition':True,
 'next_readyable_unit':'One bounded shared171 builtin profile cohort plus graph/guard scopes with dispatch/native hooks disabled;prepare global limits,case/phase and restoration. Retain all native/supplement/capture gaps where not observed.',
 'new_model_or_device_execution':False,'installed_into_target':False,
 'original1352_recovered':0,'whole_modules_closed':0,'overall_complete':False}
write(O/'targeted-observer-readiness-v163.json',readiness)
write(O/'targeted-local-environment-correction-v163.json',{
 'event':'Initial CPU fixture used /usr/bin/python3 3.9.6;exact target source uses dataclass(slots=True) and failed at import before test execution.',
 'error_type':'TypeError','correction':'Use existing /Users/wio/.local/bin/python3.12;no install or source compatibility rewrite.',
 'failed_intermediate_source_SHA_not_retained':True,
 'final_contract_run':pin(root/'cpu-contract-result.json'),
 'new_CANN_or_device_work':False,'not_a_native_or_model_failure':True})
summary={'version':163,'status':'PROGRESS',
 'previous_goal_turn':'PROGRESS:v161 real own probe and restore;v162 precise factory forwarding audit;active Goal resumed.',
 'new_evidence':'Exact171 graph sources,conditional workspace/rebinding contract,and bounded targeted observer CPU contracts.',
 'CPU_contract_gates':cpu['gate_count'],'real_dispatch_or_CANN_execution':False,
 'default_dispatch_mode_disabled':True,'shared_builtin_profile_route_not_blocked_by_all_native_gaps':True,
 'new_device_work':False,'new_pair_clauses':0,'new_independence_proofs':0,
 'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,
 'module_inventory':26,'semantic_rows':358,'model_candidates':35,'framework_candidates':7,
 'inventory_exhaustiveness_proved':False,'physical_MFU_or_lowest_established':False,
 'whole_modules_closed':0,'overall_complete':False}
write(O/'targeted-routes-summary-v163.json',summary)
inputs={Path(r['path']) for r in read(O/'pending-entrypoint-evidence-manifest-v162.json')['artifacts']}
inputs.add(O/'pending-entrypoint-evidence-manifest-v162.json')
inputs.update(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
inputs.update(p for p in cache.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
inputs.update(O/name for name in ['graph-entrypoint-sources-v163.json','capture-workspace-epoch-contract-v163.json',
 'targeted-observer-readiness-v163.json','primary-query-epoch-docs-v163.json',
 'targeted-local-environment-correction-v163.json','joint-profiler-capability-observation-v161.json'])
inputs.add(Path(__file__))
write(O/'targeted-routes-inputs-v163.json',{'inputs':[pin(p) for p in sorted(inputs)],
 'summary_is_generated_output_not_immutable_acquisition_input':True,
 'no_new_runtime_authorization':True,'overall_complete':False})
print(json.dumps({'status':'TARGETED_OPERATOR_EVIDENCE_PREPARED','CPU_gates':cpu['gate_count'],
 'inputs':len(inputs),'no_new_runtime_work':True,'overall_complete':False}))
