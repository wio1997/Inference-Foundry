"""Analyze saved metadata and selected source contracts without importing the model."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
from collections import Counter

from finalize_origin_coverage_v9 import N, O, read, write


def selected(relative, cls, name):
    p = N / 'original-source-cache-v53' / relative
    text = p.read_text(); tree = ast.parse(text)
    body = tree.body
    if cls:
        body = next(x.body for x in body if isinstance(x, ast.ClassDef) and x.name == cls)
    node = next(x for x in body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                qualified_name=(cls + '.' if cls else '') + name, first_line=node.lineno,
                last_line=node.end_lineno, source=text.splitlines()[node.lineno-1:node.end_lineno]), node


def main():
    stamp = datetime.now(timezone.utc).isoformat()
    status = O / 'original-profile-access-status-v53.json'
    archive = O / 'original-profile-access-before-xinyufeng-v53.json'
    if not archive.exists():
        assert not read(status)['original_profile_access_restored']
        archive.write_bytes(status.read_bytes())
    identity = read(N / 'original166-xinyufeng-recheck-v53.json')
    assert identity['application_status'] == 'REMOTE_ROOT_IDENTITY_RETURNED'
    assert identity['remote_result']['hostname'] == 'bm-jn-zs-zone1-910C-64G-10-105'
    core = read(O / 'original-profile-core-cache-index-v53.json')
    source = read(O / 'original-source-cache-index-v53.json')
    assert core['all_four_match_historical_hashes'] and source['all_six_historical_controls_match']
    write(status, dict(generated_utc=stamp, latest_user_instruction='Try the previously supplied xinyufeng account on exact10.55.0.166.',
        successful_receipt=str(N / 'original166-xinyufeng-recheck-v53.json'),
        historical_failed_account_diagnosis=str(archive), account='xinyufeng@root@10.55.0.166',
        hostname=identity['remote_result']['hostname'], root_identity_verified=True,
        original_profile_access_restored=True, four_core_indices_match_historical_hashes=True,
        original_profile_core_cache=str(O / 'original-profile-core-cache-index-v53.json'),
        all_original_files_integrity_verified=False, original_case_not_replaced_by_193=True,
        no_model_torch_device_service_health_process_work=True, pending_information=None, overall_complete=False))
    receipt = read(N / 'saved-metadata-config166-v53.json')
    assert receipt['returncode'] == 0 and receipt['application_json_valid']
    data = json.loads(receipt['stdout']); cfg = data['capture_configurations']; dbs = data['databases']
    assert len(cfg) == len(dbs) == 32
    signatures = Counter((r['common_config']['record_shapes'], r['common_config']['with_stack'],
                          r['common_config']['profile_memory'], r['experimental_config']['record_op_args']) for r in cfg)
    assert signatures == {(True, True, False, False): 32}
    assert {(r['cann_version'], r['torch_npu_version']) for r in cfg} == {('9.2.0', '2.10.0.post7')}
    assert Counter(r['rank_id'] for r in cfg) == {i: 2 for i in range(16)}
    assert all(r['source_stat_unchanged'] for r in dbs)
    metadata_tables = Counter(t['table'] for r in dbs for t in r['metadata'] if t['present'])
    write(O / 'saved-capture-config-metadata-census-v53.json', dict(generated_utc=stamp,
        source_receipt=str(N / 'saved-metadata-config166-v53.json'),
        databases=32, capture_configurations=32, ranks=16, captures_per_rank=2,
        cann_version='9.2.0', torch_npu_version='2.10.0.post7',
        record_shapes_true=32, with_stack_true=32, profile_memory_false=32, record_op_args_false=32,
        metadata_table_presence=dict(metadata_tables), raw_database_stat_unchanged=32,
        meta_data_values_read=False,
        answered='The missing full32configuration census is complete. Saved collection did not enable profile_memory or record_op_args in any of32configurations.',
        implications=['Existing stacks/shapes remain useful for bounded attribution and point-source selection.',
            'These flags do not establish storage alias/version/ownership. Inspect other saved value/metadata surfaces only for a concrete question.',
            'Session/rank metadata does not itself bind16phase-unbound graphs; their prior zero-sidecar-candidate finding is retained.',
            'NPU_INFO reports Ascend910B as a saved tool label; this is not independent physical SKU/peak attestation.'],
        new_capture=False, whole_modules_closed=0, overall_complete=False))
    specs = [
        ('vllm/vllm/distributed/eplb/eplb_state.py', 'EplbState', 'build_initial_global_physical_to_logical_map'),
        ('vllm/vllm/model_executor/utils.py', None, 'set_weight_attrs'),
        ('vllm-ascend/vllm_ascend/quantization/method_adapters.py', 'AscendFusedMoEMethod', 'process_weights_after_loading'),
        ('vllm/vllm/outputs.py', 'RequestOutput', 'add'),
        ('vllm/vllm/v1/engine/output_processor.py', 'OutputProcessor', '_update_streaming_request_state'),
        ('vllm/vllm/entrypoints/openai/completion/serving.py', 'OpenAIServingCompletion', 'completion_stream_generator'),
    ]
    parts = [selected(*s)[0] for s in specs]
    write(O / 'restored-source-point-contracts-v53.json', dict(generated_utc=stamp, parts=parts,
        identity_scope='Read20files from exact existing166stage container by docker cp archive, without container execution. Six historical file controls match. Other files establish current stage bytes, not historical loaded Python object identity.',
        previously_reference_files_now_current_stage_hash_matched=['method_adapters.py','modelslim_config.py','vllm_adaptor.py','eplb_updator.py','eplb_device_transfer_loader.py','output_processor.py','async_llm.py','logprobs.py'],
        no_new_import_device_or_container_process=True, overall_complete=False))
    write(O / 'F21-initial-map-and-attribute-helper-contract-v53.json', dict(generated_utc=stamp,
        source='restored-source-point-contracts-v53.json',
        initial_map_domain='E>0,R>=0,integer physical p in[0,E+R)',
        initial_map_formula='P0[p]=p for p<E;otherwise (p-E)%E. R=0 gives identity.',
        source_map_is='Original logical experts followed by round-robin redundant copies; it does not consume the static deployment placement file.',
        set_weight_attrs=['None attrs returns.', 'Each attribute must be absent or an assertion is raised.',
            'For weight_loader only, current_platform.use_sync_weight_loader() may replace the supplied callable with make_synced_weight_loader.',
            'setattr then captures the resulting callable on the tensor; rebinding the layer callback later does not rewrite an existing tensor attribute.'],
        adapter_postload='AscendFusedMoEMethod.process_weights_after_loading delegates to quant_method when present; no independent placement reconciliation in this inspected adapter method.',
        answered=['Missing P0 constructor body is now known from original166stage bytes.',
            'Missing helper assignment/sync-wrapper behavior is now known.',
            'Five previously local-reference callback/EPLB dependencies match current166stage bytes.'],
        remaining=['Actual selected platform hook/config and parameter callback objects are not attested by static source.',
            'Actual static deployment placement, loader map, full execution route and postload payload generations must commute if static placement is pursued.',
            'Current stage bytes do not retroactively attest historical loaded objects or all backend hooks.'],
        target_changed=False, implementation=False, new_gain=None, whole_module_closed=False, overall_complete=False))
    add = selected('vllm/vllm/outputs.py', 'RequestOutput', 'add')[1]
    writes = sorted({ast.unparse(t) for x in ast.walk(add) if isinstance(x, (ast.Assign, ast.AugAssign))
                     for t in (x.targets if isinstance(x, ast.Assign) else [x.target])})
    assert 'self.finished' in writes and 'completion.finish_reason' in writes
    assert not any(k in str(writes) for k in ['sampling_mask','routed_experts','spec_decode_metrics','prompt_logprobs','metrics'])
    # Independent text-match specification: minimize completion position, then configured list order.
    def first_end(text, stops):
        matches = [(start+len(s), j, start, s) for j,s in enumerate(stops)
                   for start in range(len(text)+1) if text.startswith(s,start)]
        end,j,start,s = min(matches)
        return dict(stop=s,end=end,list_index=j,visible_excluding_stop=text[:start])
    witnesses = [dict(text='AB', stops=['B','AB'], result=first_end('AB',['B','AB'])),
                 dict(text='AB', stops=['AB','B'], result=first_end('AB',['AB','B'])),
                 dict(text='abc', stops=['bc','ab'], result=first_end('abc',['bc','ab']))]
    assert [w['result']['visible_excluding_stop'] for w in witnesses] == ['A','','']
    write(O / 'M25-output-merge-and-terminal-contract-v53.json', dict(generated_utc=stamp,
        source=['M25-output-lifecycle-source-v53.json','M25-output-terminal-dependency-source-v53.json','restored-source-point-contracts-v53.json'],
        existing_mechanisms=['Native DecodeStream is already primed with prompt IDs.',
            'Collector buffers one pending result; DELTA aggregates, other RequestOutput modes replace matched completion snapshots.',
            'Async handler already chunks output batches and yields between chunks; generate already uses get_nowait before awaiting.',
            'FINAL_ONLY suppresses unfinished output. stream_interval condition preserves first/final paths; sent_tokens_offset update is DELTA-only in inspected source.'],
        add_assignment_targets=writes,
        add_semantics=dict(request_fields='finished OR;kv_transfer_params and ec_transfer_params take next values.',
            matched_DELTA='Append text and token IDs;append truthy next logprobs with existing nonNone assertion;take next cumulative logprob,finish_reason,stop_reason.',
            matched_nonDELTA='Replace the completion object.', unmatched_index='Append the next completion object.',
            fields_not_explicitly_refreshed_by_add=['prompt_logprobs','metrics','num_cached_tokens','num_cache_creation_tokens','sampling_mask','routed_experts','spec_decode_metrics'],
            scope='No explicit refresh in this method is not a claim that all values are immutable or lost in the current run; ownership,mode,flags and external mutation matter.'),
        stop_semantics='check_stop_strings uses earliest match end;ties preserve configured list order. update decodes all IDs in each supplied chunk before checking stop and may truncate text without trimming token history.',
        independent_stop_specification_witnesses=witnesses,
        pre_stop_fusion_counterexample=dict(kind='Independent abstract string/token-state witness,not native tokenizer/model test.',
            decoder={0:'E',1:'N',2:'D',3:'x'}, stop=['END'], min_tokens=0, include_stop=False,
            chunks=[[0,1,2],[3]], separate='First chunk stops and finishes request with empty visible text and3token IDs;later output ignored after state removal.',
            fused='One ENDx chunk stops with the same empty visible text but4token IDs because stop check follows the entire chunk.',
            implication='Text equality alone is insufficient for moving aggregation ahead of semantic detokenization/stop/finish.'),
        finish_metadata_counterexample=dict(kind='Conditional source-level pending-DELTA witness.',
            pending='Existing completion index0 has sampling_mask/routed_experts/spec_decode_metrics=None.',
            next='Final completion index0 supplies a nonNone finish-only field.',
            aggregate_result='add updates finish_reason/stop_reason but does not assign those3fields in the matched aggregate branch.',
            scope='Requires actual enabled fields and pending merge without shared-object update;not a currentC8failure claim. NonDELTA replacement has different behavior.'),
        lifetime_constraints=['Stop string detection follows decoder update,then logprobs update,then emission;reordering must preserve this state boundary.',
            'DELTA prompt_logprobs is a one-shot nonempty pop;empty text/IDs can still carry finish or metadata.',
            'Parent n>1 aggregation may change request finished independently of child finish.',
            'streaming_input originates from resumable input,not SSE stream flag;final input marker,queued updates and identity of STREAM_FINISHED matter.',
            'Cancel/GeneratorExit abort,errors have distinct abort behavior,finally closes input-stream task. Offloaded late results need request generation/owner checks.',
            'Collector put/Event mutation is not an arbitrary-thread safety contract.'],
        API_stream_observers=['Actual selected completion generator serializes per completion;echo/token-ID/logprob/text-offset branches constrain visible chunks.',
            'Initial empty chunked-prefill suppression already exists;do not generalize it to arbitrary terminal/metadata output.',
            'Continuous usage,optional final usage metrics,terminal fingerprint,error messages and [DONE] are separate observable outputs.',
            'SSE source analysis does not prove actual socket flush,client latency or every API path.'],
        semantic_source_review_complete_for_selected_functions=True, original26module_scope_complete=False,
        implementation=False, new_model_native_test=False, performance_gain=None, whole_module_closed=False, overall_complete=False))
    conditions = [
        ('H05-PREP','Per-request decoder preparation and owner publication','Backend must expose genuinely preparable work;preserve prompt priming,special tokens,UTF8 recovery and publication-before-first-use. Long wall profile does not establish backend cause/GIL behavior.'),
        ('H05-SPLIT','Per-request serial offload with ordered owner publication','Serial stream mutation;request generation and cancel/abort guard;publish structured results on owner event loop;bound work-in-flight and prove backend thread safety.'),
        ('H05-EMIT','Emission policy or serialization batching after semantic processing','Reuse existing merge/chunk/yield baseline;preserve DELTA/CUMULATIVE/FINAL_ONLY,first/final/interval behavior,all fields,one-shot prompt logprobs,parent andresumable-input lifecycle;explicit latency budget.'),
        ('H05-STOP','Optional compiled or shared immutable stop matcher','Only if stop matching is relevant;preserve earliest end/list-order ties,min_tokens eligibility,overlap window,holdback and UTF8 semantics;do not fuse chunks across stop/finish boundaries.'),
    ]
    write(O / 'M25-conditional-output-reorganizations-v53.json', dict(generated_utc=stamp,
        existing_framework_candidate='H05', variants=[dict(id=i,proposal=p,conditions=c,status='CONDITIONAL_ANALYSIS')for i,p,c in conditions],
        candidate_count_increased=False, already_implemented_mechanisms_not_new_gains=True,
        additive_gain_assumed=False, original_guard_values_attested=False,
        no_implementation_native_test_or_gain=True, whole_module_closed=False, overall_complete=False))
    families = [
        ('01','Engine output acquire / handler scheduling','Acquisition,existing chunk/yield and direct process handoff reviewed.','Producer-ready/currentEngineCore cost and complete transport path.'),
        ('02','Pending collector / merge / error precedence','DELTA aggregate vs snapshot replace and direct RequestOutput.add fields reviewed.','Actual optional fields/aliases,all object generations and error interleavings.'),
        ('03','Output modes / interval / first and final','Three modes and DELTA-only offset mutation reviewed.','Actual modes/intervals across all requests and all guards.'),
        ('04','Incremental decode / stop / UTF8','Historicaldetokenizer hash retained;stop ordering and chunk-boundary witness reviewed.','Backend actual cause/onCPU/thread contract and remaining tokenizer variants.'),
        ('05','Structured text / IDs / logprobs / finish metadata','Delta empty slice,prompt pop,finish mask/expert/spec fields and merge limitations reviewed.','Actual flags,objects and all consumers.'),
        ('06','Parent and n>1 output aggregation','Direct get_outputs call and parent-finished distinction localized.','ParentRequest body and all child/finish/abort cases.'),
        ('07','Resumable streaming input','Queue/apply/final marker helper reviewed;distinct from SSE.','STREAM_FINISHED construction,all request/update/cancel interleavings.'),
        ('08','Cancel / abort / input and engine errors','Selected generate/abort/error paths reviewed.','All EngineCore and API cancellation paths and outstanding offload behavior.'),
        ('09','Cleanup / mapping / request lifetime','State removal,parent mapping and input task close reviewed.','All ownership/destructor/request-ID-reuse and late-message cases.'),
        ('10','Completion API / JSON / SSE / usage / client','Selected generator echo/logprob/usage/error/DONE observers reviewed.','All alternateAPI paths,transport/socket/client timing and actual response options.'),
        ('11','Worker / EngineCore / frontend IPC and output readiness','Prior worker readiness evidence reused;frontend parent timing retained.','Current166EngineCore process,producer-ready/IPC causal cost and complete branch graph.'),
    ]
    evidence=['M25-output-merge-and-terminal-contract-v53.json','M25-conditional-output-reorganizations-v53.json','restored-source-point-contracts-v53.json']
    rows=[dict(id='M25-S'+i,branch=b,state='SCOPED_SOURCE_CONTRACTS_REVIEWED;FAMILY_PARTIAL',evidence=evidence,gap=g,
               scope='Selected saved-profile paths/currentstage source;'+f)for i,b,f,g in families]
    write(O / 'M25-semantic-coverage-frontier-v53.json', dict(generated_utc=stamp, module='M25',semantic_families=rows,
        universe_exhaustive=False, full_source_guard_inventory=False, all_consumers_closed=False,
        historical_frontend_Python_events=39544, frontend_PID_scope='1;separate from16workerPIDs and historical169EngineCore',
        saved_wall_timing_is_onCPU_or_gain=False, whole_module_closed=False, overall_complete=False))
    gate=read(O / 'PROFILE-CODE-QUESTION-GATE-v53.json')
    gate.update(no_remote_work=False, remote_scope='User explicitly requested exact166rechecks and xinyufenglogin;read only saved profile/config and exact existing container files.',
        latest_source_identity='Fourcoreindex hashes and sixhistoricalsource controls match;20currentstage files cached. Unpinned files are not proof of historical loadedobjects.',
        additional_point_questions=['Missing P0 constructor,attribute helper,postload adapter and output merge/API contracts unresolved by saved operator profile alone.'],
        original_profile_access_restored=True, new_capture=False)
    write(O / 'PROFILE-CODE-QUESTION-GATE-v53.json',gate)
    print(json.dumps(dict(metadata_configs=32,source_files=20,point_contracts=len(parts),M25_semantic_families=len(rows),
                         H05_variants=len(conditions),original_access_restored=True,overall_complete=False)))


if __name__ == '__main__':
    main()
