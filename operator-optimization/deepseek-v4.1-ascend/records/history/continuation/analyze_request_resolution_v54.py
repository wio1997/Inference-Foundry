"""Freeze parameter provenance and finite branch contracts, not runtime objects."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import ast,json,hashlib
from finalize_origin_coverage_v9 import N,O,read,write

B=N/'request-resolution-source-v54/vllm/vllm'


def method(path,cls,name):
    t=path.read_text();a=ast.parse(t);body=a.body
    if cls:body=next(c.body for c in body if isinstance(c,ast.ClassDef)and c.name==cls)
    f=next(f for f in body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))and f.name==name)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),name=(cls+'.'if cls else'')+name,
                first_line=f.lineno,last_line=f.end_lineno,source=t.splitlines()[f.lineno-1:f.end_lineno])


def main():
    stamp=datetime.now(timezone.utc).isoformat();inputs=read(O/'request-resolution-inputs-v54.json')
    assert inputs['all_source_files_read']and len(inputs['saved_files'])==4 and len(inputs['source_files'])==6
    protocol=B/'entrypoints/openai/completion/protocol.py';tree=ast.parse(protocol.read_text())
    cls=next(c for c in tree.body if isinstance(c,ast.ClassDef)and c.name=='CompletionRequest')
    fields={x.target.id:ast.unparse(x.value)for x in cls.body if isinstance(x,ast.AnnAssign)and isinstance(x.target,ast.Name)and x.value}
    expected={'n':'1','echo':'False','seed':'Field(None, ge=_INT64_MIN, le=_INT64_MAX)',
        'stop':'[]','logprobs':'None','prompt_logprobs':'None','watermarking':'True','min_tokens':'0','use_beam_search':'False'}
    assert all(fields[k]==v for k,v in expected.items())
    specs=[('entrypoints/openai/completion/protocol.py','CompletionRequest','to_sampling_params'),
        ('entrypoints/openai/completion/protocol.py','CompletionRequest','extract_structured_outputs'),
        ('config/model.py','ModelConfig','get_diff_sampling_param'),
        ('config/model.py','ModelConfig','try_get_generation_config'),
        ('sampling_params.py','SamplingParams','from_optional'),('sampling_params.py','SamplingParams','__post_init__'),
        ('sampling_params.py','SamplingParams','update_from_generation_config'),
        ('sampling_params.py','SamplingParams','_verify_greedy_sampling'),
        ('v1/engine/parallel_sampling.py','ParentRequest','__init__'),
        ('v1/engine/parallel_sampling.py','ParentRequest','_get_child_sampling_params'),
        ('v1/engine/parallel_sampling.py','ParentRequest','get_child_info'),
        ('v1/engine/parallel_sampling.py','ParentRequest','get_outputs'),
        ('v1/engine/parallel_sampling.py','ParentRequest','observe_finished_request')]
    parts=[method(B/p,c,f)for p,c,f in specs]
    old=N/'original-source-cache-v53/vllm/vllm'
    parts.append(method(old/'v1/engine/async_llm.py','AsyncLLM','add_request'))
    out=old/'outputs.py';a=ast.parse(out.read_text())
    sentinel=next(x for x in a.body if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='STREAM_FINISHED'for t in x.targets))
    parts.append(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),name='STREAM_FINISHED',first_line=sentinel.lineno,last_line=sentinel.end_lineno,source=out.read_text().splitlines()[sentinel.lineno-1:sentinel.end_lineno]))
    write(O/'request-resolution-selected-source-v54.json',dict(generated_utc=stamp,CompletionRequest_declared_defaults=fields,parts=parts,
        source_identity_scope='Six files from original166stage;serving control matches v53. Existing async/output files reused. Current stage bytes plus saved records are not historical loaded object snapshots.',
        no_import_model_or_device_work=True,overall_complete=False))
    rows=[]
    def add(field,value,origin,effect,status='SOURCE_RESOLVED_FOR_FROZEN_REQUEST'):
        rows.append(dict(field=field,value=value,origin=origin,analysis_effect=effect,status=status))
    add('endpoint','/v1/completions','HistoricalSHA-pinned official_profile_client.run_request','Completion protocol applies;chat-only paths are alternatives.')
    add('prompt','8 text strings','Pinned frozen request receipt v8 plus original client','Text tokenization remains in scope;token-ID HTTP branch is not this input.')
    add('temperature',0,'Client explicit literal','Does not depend on model default temperature.')
    add('top_p/top_k/min_p',[1.0,0,0.0],'SamplingParams.__post_init__ greedy normalization after validation','Model defaults do not preserve nontrivial top filters for zero temperature;pre-normalization validation still matters.')
    add('n',1,'Request omitted;CompletionRequest default1','Selected AsyncLLM path bypasses ParentRequest fanout;not a universal n>1 exclusion.')
    add('seed',None,'Request omitted;CompletionRequest defaultNone','Per-request seed differs from model engine seed0;no seeded-child claim.')
    add('logprobs/prompt_logprobs/logprob_token_ids',[None,None,None],'Protocol defaults,echoFalse,not model-default sampling keys','Source-selected request asks for no probability return;other internal probability/entropy consumers remain.')
    add('presence_penalty/frequency_penalty',[0.0,0.0],'Protocol scalar defaults','These per-request penalty parameters are neutral;does not settle repetition penalty or all processors.')
    add('repetition_penalty','MODEL_DEFAULT_DEPENDENT','Protocol None→ModelConfig.get_diff_sampling_param','Do not assume1.0 before model generation config/override provenance.',status='REMAINING_VALUE_GAP')
    add('watermarking',True,'Protocol default→from_optional passes through','A true field is not proof of an active watermark processor. Consumer/installation remains unresolved.',status='FIELD_KNOWN_CONSUMER_OPEN')
    add('stop/min_tokens/include_stop_str_in_output',[[],0,False],'Protocol defaults;generation sampler whitelist has no stop strings','Stop-string matcher variant is alternative-path analysis,not an evidenced active C8bottleneck.')
    add('ignore_eos',True,'Client explicit literal','Generation-config update can still populate all-stop-token IDs for other semantics;do not zero all stop metadata.')
    add('max_tokens',1024,'Formal client construction plus actual8x1024 validated usage','Keep model-budget clipping/get_max_tokens semantics;requested count was realized.')
    add('use_beam_search/echo',[False,False],'Protocol defaults','No beam branch or prompt echo for this source-selected request.')
    add('stream/output_kind',[True,'DELTA'],'Formal requests parallel(...,True) and to_sampling_params','Pending DELTA aggregation path remains relevant.')
    add('stream_options.include_usage',True,'Client conditional addition when streamTrue','Final usage chunk is required;do not drop empty choices chunks.')
    add('return_token_ids',None,'Protocol default','Explicit token-ID transport not requested;internal ID semantics remain observable elsewhere.')
    add('stream_interval','SERVER_DEFAULT_DEPENDENT','Request omitted/None;AsyncLLM outputprocessor receives scheduler interval','No assumption that all interval logic is inactive.',status='REMAINING_VALUE_GAP')
    add('cache_salt/session_id',[None,None],'Protocol defaults','No explicit request salt/session;other cache hash namespace and generations remain necessary.')
    add('skip_reading_prefix_cache',False,'SamplingParams defaultNone→prompt_logprobs is notNone(False)','Initial source-selected eligibility permits prefix-cache reads;does not prove hits or later object mutation.')
    add('enable_return_routed_experts',False,'Saved effective engine config service.log103','Current run does not request routed expert output;do not generalize to alternative configs.',status='SAVED_EFFECTIVE_ENGINE_CONFIG')
    add('per_request_spec_decode_metrics','none','Saved effective engine config service.log103','Finish-only per-request spec metrics disabled;global speculative counters still exist.',status='SAVED_EFFECTIVE_ENGINE_CONFIG')
    log=next(Path(x['local_path'])for x in inputs['saved_files']if x['relative']=='runtime/logs/service.log').read_text().splitlines()
    effective=[(i+1,l)for i,l in enumerate(log)if 'Initializing a V1 LLM engine'in l]
    assert len(effective)==1 and 'enable_return_routed_experts=False'in effective[0][1]and "per_request_spec_decode_metrics='none'"in effective[0][1]
    write(O/'frozen-request-parameter-provenance-v54.json',dict(generated_utc=stamp,parameters=rows,
        saved_effective_engine_config_line=effective[0][0],profile_support='Validated formal8requests each32768prompt/1024output,plusnativegreedy positives;not per-object snapshots for every dummy/graph batch.',
        resolved_scope='Explicit client plus selected originalstage protocol/init source and saved engine config. All values apply to the declared source path,not arbitrary later mutations or alternative requests.',
        historical_loaded_object_identity_proved=False,all_processors_removed=False,new_performance_result=False,whole_module_closed=False,overall_complete=False))
    # Distinguish the singleton identity path from merging terminal state into a payload.
    write(O/'M25-parent-and-sentinel-contract-v54.json',dict(generated_utc=stamp,source='request-resolution-selected-source-v54.json',
        parent=dict(child_ids='index_parentID;each registered in child_requests.',
            seed_none='Shallow-copy SamplingParams once,setn1,cache same childparam reference for reuse.',
            seeded='Separate copy with seed+child index;do not replace by one common seed or reorder logicalchild indices.',
            streaming='Finished child is removed;already-finished-and-returned child suppressed. Other output is returned with finished=(no remainingchildren).',
            final_only='Store by completion.index;publish aggregation only after children set empty.',
            C8_scope='n1 direct path,so parent body is an analyzed alternative;all downstream request mutation/transport paths not globally excluded.'),
        sentinel=dict(definition='Module singleton RequestOutput(request_id="",prompt=None,prompt_token_ids=None,prompt_logprobs=None,outputs=[],finished=True) in vllm.outputs.',
            empty_buffer='Collector stores singleton;generate observes finishedTrue and identity check suppresses yielding it.',
            pending_payload='Collector.add merges singleton into pending payload,sets finishedTrue,keeps payload object;generate yields that payload because identity differs.',
            clone_witness='Cloning empty sentinel loses identity and can yield an extra empty RequestOutput.',
            discard_witness='Treating any terminal insertion as discard would drop an existing pending payload.',
            scope='Source state-machine witnesses,not a runtime interleaving/native/transport test. Preserve exception precedence and resumable cleanup.'),
        actual_C8_streaming_input_resumable=False,actual_resumable_exclusion_scope='Source-selected normal completion text input;all later processor/request mutation not historically snapshotted.',
        no_implementation_or_native_test=True,whole_module_closed=False,overall_complete=False))
    meta=read(O/'request-guard-saved-profile-v54.json');counts=Counter();ranks={}
    differing=[]
    for r in meta['databases']:
        assert len(r['rank_device_rows'])==1
        rank=r['rank_device_rows'][0][0];groups=next(m['value']for m in r['metadata']if m['name']=='parallel_group_info')
        for g in groups.values():
            assert g['global_ranks'][g['group_rank']]==rank
            counts[(g['group_name'],tuple(g['global_ranks']))]+=1
        ranks[rank]=ranks.get(rank,0)+1
        es=[m['value']for m in r['metadata']if m['name']=='ENV_VARIABLES']
        assert len(es)==2
        if es[0]!=es[1]:differing.append(r['database_index'])
    assert ranks=={i:2 for i in range(16)}and len(differing)==32
    write(O/'saved-logical-group-and-env-provenance-v54.json',dict(generated_utc=stamp,
        groups=[dict(group=k[0],global_ranks=list(k[1]),database_records=v)for k,v in counts.items()],
        all_group_rank_maps_agree_with_saved_rank_device_map=True,TP_membership_sets=2,EP_MC2_rank_size=16,
        physical_EP16_topology_or_MC2_workwait_proved=False,
        duplicate_env_rows_differ_in_databases=differing,
        env_implication='EachDBhas2ENV_VARIABLES rows with different loglevel/allocator values. Row order/last-wins is not established;do not turn duplicate rows or empty strings into actual worker env assertions.',
        processor_frames_observed=list(meta['recorded_frame_totals']),
        no_watermark_frames_is_disable_proof=False,no_native_event_storage_phase_or_peak_proof=True,whole_module_closed=False,overall_complete=False))
    definitions=[
        ('01','Endpoint / frozen request / DP header','Exact client route,body construction,formal versus seed request provenance reviewed.','All admission/header validation and error branches.'),
        ('02','Input representation / render / tokenizer','8text inputs and real tokenization profile retained;IDs/chat/beam branches separated.','All renderer/tokenizer variants and producer cache namespace/readiness.'),
        ('03','Length and token budgets','Requested1024 and realized8x1024;max-model limit remains.','get_max_tokens clipping/override/all prompt-length/error cases.'),
        ('04','Sampling scalar defaults and greedy normalization','temperature0,top filters normalized;validation precedes greedy normalization.','All alternative probability paths and per-step/dummy metadata guards.'),
        ('05','Penalties / constraints / custom processors','Presence/frequency neutral;repetition default dependent;watermark fieldTrue but consumer unproved.','Actual model generation defaults,watermark/processor installation and all consumers.'),
        ('06','Logprob / entropy / structured return','No per-request logprob return by source-selected fields;engine optional expert/spec metrics disabled.','Internal probability/entropy,sampling-mask and alternate output consumers.'),
        ('07','Seed / RNG / determinism','RequestseedNone distinguished from engineseed0;parent seeds index-aware.','All algorithm/RNG draw order and stochastic alternative guards.'),
        ('08','Stop / EOS / finish policy','No request stopstrings,min_tokens0,ignoreEOStrue;all-stop metadata not zeroed.','All limit/error/abort/repetition/structured finish paths and tokenizer variants.'),
        ('09','Multiplicity / beam / parent','n1 normal path;full selected ParentRequest methods reviewed.','Other fanout/admission/childcancel/duplicate completion interleavings.'),
        ('10','Stream / delta / usage / interval','FormalstreamTrue/DELTA/include_usageTrue;server interval stillunknown.','Actual server interval,all output metadata and transport consumers.'),
        ('11','Prefix-cache eligibility / salt / session','Requestsalt/sessionNone,initial prompt-logprob cache skipFalse.','H,extra namespace,alloc/recycle/generation and later request mutation.'),
        ('12','Lifecycle / failure / request mutation','AsyncLLM registration-before-await and request.params reread located.','Full inputprocessor mutations,resumable/cancel/engine/API error and process boundaries.'),
    ]
    evidence=['frozen-request-parameter-provenance-v54.json','request-resolution-selected-source-v54.json','request-guard-saved-profile-v54.json']
    families=[dict(id='M01-S'+i,branch=b,state='SCOPED_SOURCE_AND_PROFILE_PROVENANCE;FAMILY_PARTIAL',evidence=evidence,gap=g,scope=f)for i,b,f,g in definitions]
    write(O/'M01-semantic-coverage-frontier-v54.json',dict(generated_utc=stamp,semantic_families=families,
        universe_exhaustive=False,all_guards_consumers_closed=False,whole_module_closed=False,overall_complete=False))
    prev=read(O/'module-phase-branch-matrix-v53.json');m25=deepcopy(next(m for m in prev['modules']if m['id']=='M25')['branches'])
    for row in m25:
        if row['id']=='M25-S04':
            row['scope']+=' C8source-selected stopstrings are empty;stop-matcher variant is an alternative-path analysis.'
        if row['id']=='M25-S05':
            row['gap']='Actual sampling-mask guard/aliases/all consumers;C8saved config disables returned routed experts and per-requestspec metrics,while request logprob fields are source-resolvedNone.'
        if row['id']=='M25-S06':
            row.update(scope='ParentRequest selected methods now read;normalC8n1 path bypasses fanout.',gap='All child/admission/cancel/transport interleavings and historical request mutation remain. Parent body is no longer missing.')
        if row['id']=='M25-S07':
            row.update(scope='Resumable input helper plus singleton definition and pending-payload merge now reviewed;distinct from SSE.',gap='All resumable request/update/cancel/error interleavings;singleton construction is no longer missing.')
        if row['id']=='M25-S10':
            row['gap']='Other API paths,response options,network/socket/client timing and transport consumers;formalC8include_usageTrue nowknown.'
        if row['id']in['M25-S06','M25-S07','M25-S10']:
            row['evidence'].append('M25-parent-and-sentinel-contract-v54.json');row['evidence'].append('frozen-request-parameter-provenance-v54.json')
    write(O/'M25-semantic-coverage-refinement-v54.json',dict(generated_utc=stamp,semantic_families=m25,universe_exhaustive=False,whole_module_closed=False,overall_complete=False))
    write(O/'PROFILE-CODE-QUESTION-GATE-v54.json',dict(status='SAVED_REQUEST_AND_STACK_PROFILE_NEED_DEFAULT_AND_LIFECYCLE_CALLEES',
        previous_turn_classification='PROGRESS:v53original_access_full_saved_cache_and_output_contracts',
        profile_evidence=['request-guard-saved-profile-v54.json','frontend-function-contract-v14.json','module-phase-branch-matrix-v53.json#M01', 'M25-semantic-coverage-frontier-v53.json'],
        questions=['Resolve omitted request fields without inferring disabled paths from missing HTTPkeys or nativeframes.',
            'Resolve ParentRequest and STREAM_FINISHED direct consumer contracts left open byv53.'],
        allowed_source_files=str(O/'request-resolution-inputs-v54.json'),selected_source=str(O/'request-resolution-selected-source-v54.json'),
        remote_work='Existing saved capture files and known166stage docker-cp reads only;no container exec/start/model/Torch/NPU/newcapture/servicehealth/processquery.',
        no_repository_wide_or_SDK_audit=True,whole_module_closed=False,overall_complete=False))
    print(json.dumps(dict(parameter_provenance_rows=len(rows),selected_source_parts=len(parts),M01_families=len(families),M25_families=len(m25),all32_metadata_read=True,overall_complete=False)))


if __name__=='__main__':main()
