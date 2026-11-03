"""Prepare a single evidence delta from the completed v160/v161 acquisitions.

No runtime calls, profile replays or predecessor case-suite reexecution.
"""
import json,struct
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
from decode_memset_metadata_v111 import decode_record

assert not (O/'REVIEW-CHECKPOINT-v161.json').exists()
M=Path('/Users/wio/work/171-vllm-migration-20261102')
native=read(O/'joint-profiler-capability-observation-v161.json')
packet=bytes.fromhex(native['native_host_packet']['args_hex'])
old=read(O/'native-store-own-scalar-traces-v128.json')
fields=decode_record(packet[24:120])
same=fields==old['decoded_fields']
assert same and native['native_host_packet']['num_blocks']==48
conditional={
 'status':'EXISTING_SCALAR_PROOF_REUSABLE_UNDER_EXPLICIT_DEVICE_OBJECT_AND_RELOCATION_PREMISES',
 'current_packet':pin(O/'joint-profiler-capability-observation-v161.json'),
 'reused_proof':pin(O/'native-store-own-scalar-traces-v128.json'),
 'reused_limitations':pin(O/'native-store-fusion-consequences-v128.json'),
 'inline_record_offset_from_host_packet':24,'inline_record_bytes':96,
 'placeholder_pairs':native['native_host_packet']['placeholders'],
 'host_pre_relocation_pointer_slot_at_offset8':int.from_bytes(packet[8:16],'little'),
 'zero_slot_does_not_prove_NULL_or_no_data':True,
 'decoded_record':fields,'record_equals_v128_own_record':same,
 'num_blocks_equals_v128':True,
 'conditional_result':{'output_site_blocks':[0],'site_PC':1144,
  'source_matched_store_request_bytes':32,'launched_blocks':48,
  'conditional_first_pointer_range':[native['raw_packet_word0'],native['raw_packet_word0']+32]},
 'premises':['The executed device code and scalar semantics match the pinned v128 object/model.',
  'The host placeholder [8,24] supplies this inline record as the second device argument.',
  'The first host word supplies the model destination without a role-changing runtime transformation.'],
 'no_new_synthetic_or_48_block_suite_run':True,
 'loaded_device_instruction_bytes_attested':False,
 'actual_HBM_or_store_transaction_bytes_measured':False,
 'actual_kernel_GPU_time_measured':False,
 'private_allocation_membership_is_not_workspace_role_or_all_reader_proof':True,
 'original_pending_operands_recovered':0,'new_independence_proofs':0,
 'whole_modules_closed':0,'overall_complete':False}
write(O/'joint-native-existing-proof-reuse-v161.json',conditional)

cases=[
 {'id':'c8-input32k-output1k-prefix90-171','shape':[8,32768],
  'dtype':'token IDs; internal W8A8/BF16/FP32','layout':'TP8 DP2 EP16',
  'group':'frozen-end-to-end-171','reason':'用户迁移至171继续原整体目标；保存独立硬件身份，未运行完整Case。',
  'attributes':{'output_tokens_per_request':1024,'prefix_hit_target':0.9,
   'hardware':'172.16.10.171 Ascend910C/A3 logical16','CANN':'9.2.0',
   'vllm_commit':'ced6857afa0ea7b2e3f0846a62e1394e90f15607',
   'ascend_commit':'8fde6d4e335fe300e2b949aab0e31659cce0811d',
   'full_case_on_171_status':'NOT_MEASURED','driver_firmware_variation_accepted_by_user':True,
   'semantics':'Preserve frozen numerical/cache/RNG/collective/alias/state contracts; no implementation changes.'}},
 {'id':'bringup-one-chat-171','shape':[1,40],'dtype':'token IDs; model mixed precision',
  'layout':'TP8 DP2 EP16','group':'runtime-bringup-171',
  'reason':'单次基础生成单独登记，不能替代完整负载性能或模型精度验收。',
  'attributes':{'max_output_tokens':128,'actual_output_tokens':26,'answer':'2',
   'prompt':'请直接回答：1加1等于几？','temperature':0,
   'scope':'One health/model-registration/basic-generation check only; not full accuracy or performance.'}},
 {'id':'provider-pad20-view-offset1-171','shape':[20],'dtype':'BOOL',
  'layout':'strided contiguous view, storage_offset=1','group':'bounded-provider-capability-171',
  'reason':'本次单独授权的联合最小探针；原模型历史任务不回填。',
  'attributes':{'device':0,'input_owner_shape':[21],'storage_offset':1,
   'outputs':[[24],[24]],'application_NPU_tensor_bytes':69,
   'runtime_context_workspace_overhead_separate':True,'CPU_profiler_sessions':2,
   'profile_memory_control':False,'profile_memory_observed':True,
   'record_shapes':True,'with_stack':True,'NPU_profiler_activity':False,
   'model_loads_in_probe':0,'graph_capture':False,'graph_replay':False,
   'native_attempts':1,'native_record_limit':2,'packet_limit_bytes':4096,
   'timeout_seconds':45,'diagnostic_limit_bytes':8388608}}]
write(O/'case-registrations-v161.json',{'cases':cases,'old_cases_preserved':True})

write(O/'local-control-analysis-corrections-v161.json',{
 'status':'CORRECTED_LOCALLY_NO_NATIVE_RESUBMISSION',
 'events':[
  {'stage':'Ordinary bringup controller template','error':'Global placeholder substitution rewrote embedded names, causing remote Python SyntaxError before execution.',
   'correction':'Substitute controller names before embedding source; successful worker source separately pinned.'},
  {'stage':'Publish first runtime receipt template','error':'Replacing RUN rewrote RUNTIME tokens, causing Python SyntaxError before execution.',
   'correction':'Prefix variable injection; successful publisher pinned.'},
  {'stage':'Offline raw TLV decoder','error':'Counter.update(fields) treated string field values as counts and raised TypeError.',
   'correction':'Count fields.keys(); re-decode the same saved bytes, no native rerun.'}],
 'failed_template_exact_bytes_not_retained':True,
 'these_errors_are_not_native_or_model_failures':True,'native_attempts':1,
 'all_device_and_model_executions_reported_separately':True})

summary={'version':161,'status':'PROGRESS',
 'previous_goal_turn':'v159 source/header contract; completed v160 bounded content/provider and v161 single authorized runtime evidence',
 'checkpoint_consolidates_evidence_versions':[160,161],
 'no_separate_v160_authority_created':True,
 'target':'172.16.10.171','target_root':'/data/tiankuan/wio',
 'ordinary_basic_generation_pass':True,'full_fixed_case_on_171_measured':False,
 'six_duplicate_current_payload_pairs_equal':True,'parameter_payload_bytes_read':1258618880,
 'mapped_CANN_backing_ELF_objects':2,'provider_mapping_records':40,
 'SDK_ABI_CPU_static_checks_pass':True,
 'native_probe_attempts':1,'native_probe_correctness':'PASS_OWN_PAD_PAIR_ONLY',
 'native_host_packets':1,'raw_host_packet_bytes':152,
 'diagnostic_files':25,'diagnostic_file_bytes':108031,
 'builtin_detail_gate_runtime_observed_in_this_case':True,
 'view_offset_missing_from_builtin_records':True,
 'real_HostArgs_interposition_observed_in_this_case':True,
 'all_provider_bypass_or_error_paths_verified':False,
 'service_restored_health_http':200,'probe_authorization_consumed':True,
 'new_device_work':True,'new_device_work_scope':'Ordinary model startup/basic <REDACTED_CREDENTIAL>, then one bounded own Pad pair and authorized model restore.',
 'new_pair_clauses':0,'new_independence_proofs':0,
 'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,
 'model_candidates':35,'framework_candidates':7,'module_inventory':26,'semantic_rows':358,
 'inventory_exhaustiveness_proved':False,'physical_MFU_or_lowest_established':False,
 'whole_modules_closed':0,'overall_complete':False}
write(O/'joint-runtime-summary-v161.json',summary)

paths=set()
paths.update(p for p in O.glob('*v160.json'))
paths.update(p for p in O.glob('*v161*.json') if not any(x in p.name for x in ['inputs','preflight','postcheck','CHECKPOINT','manifest']))
paths.update(p for p in N.glob('*v160.py'))
paths.update(N/name for name in ['native_hostargs_pad171_v161.py','native_hostargs_controller171_v161.py',
 'prepare_native_hostargs171_v161.py','launch_native_hostargs171_v161.py','collect_native_hostargs171_v161.py',
 'analyze_joint_profiler171_v161.py','host-adapter-CPU166-v127c.json'])
paths.update(p for p in (N/'provider-abi-cache-v160').iterdir() if p.is_file())
paths.update(Path(r['local_path']) for r in read(O/'joint-hostargs-probe-file-index-v161.json')['files'])
paths.update(M/name for name in ['RUNTIME-READY-171-20261102.json','bringup_worker.py',
 'launch_bringup.py','publish_runtime_ready.py','source-fixed-version-files.json',
 'source-CANN-version-files.json','postflight-target.json'])
paths.update(O/name for name in ['TorchNPU-input-field-producer-contract-v159.json',
 'checkpoint-all-header-index-v159.json','native-store-own-scalar-traces-v128.json',
 'native-store-fusion-consequences-v128.json','host-adapter-collector-readiness-v127.json'])
paths.update([N/'decode_memset_metadata_v111.py',Path(__file__)])
write(O/'joint-runtime-inputs-v161.json',{'inputs':[pin(p) for p in sorted(paths)],
 'input_scope':'Completed snapshots only; no mutable remote status/pid files used as live authority.',
 'old_unchanged_operator_suites_reexecuted':False,'overall_complete':False})
print(json.dumps({'prepared':'v161','input_pins':len(paths),'new_cases':len(cases),
 'existing_record_equal':same,'no_native_execution_in_this_script':True}))
