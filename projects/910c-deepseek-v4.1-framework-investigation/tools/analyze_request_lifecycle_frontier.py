#!/usr/bin/env python3
"""Original control fixtures and existing-cohort no-next-demand prices.

No NPU imports or new inference. Fixtures do not establish numerical acceptance.
"""
import ast
from hashlib import sha256
import gzip
import json
from pathlib import Path
from types import SimpleNamespace as S
import sys

sys.dont_write_bytecode=True
OUT=Path('audit/execution-search')
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
sys.path.insert(0,str(BASE/'tools'))
from price_vocab_selection_fusion import records, RECEIPT, BODY, SPANS, REPLAYS, SCHEDULE, ARRIVAL


def methods(path,names,cls=None):
    tree=ast.parse(path.read_text())
    nodes=tree.body if cls is None else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    chosen=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in chosen}==set(names)
    for n in chosen:n.decorator_list=[]
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*chosen],type_ignores=[])
    return compile(ast.fix_missing_locations(mod),str(path),'exec')


def main():
    pins=json.loads((OUT/'request-lifecycle-execution-identity.json').read_text())
    by={r['runtime_path'].removeprefix('/vllm-workspace/'):Path(r['local_path']) for r in pins['sources']}
    for r in pins['sources']:assert sha256(Path(r['local_path']).read_bytes()).hexdigest()==r['runtime_sha256']
    # Preserve original assign-ID body; deterministic UUID generator is a fixture.
    ids=iter(['11111111-uuid-fixture','22222222-uuid-fixture'])
    env=dict(envs=S(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False),random_uuid=lambda:next(ids))
    exec(methods(by['vllm/vllm/v1/engine/input_processor.py'],['assign_request_id'],'InputProcessor'),env)
    a,b=S(request_id='external-same',external_req_id=None),S(request_id='external-same',external_req_id=None)
    env['assign_request_id'](a);env['assign_request_id'](b)
    assert a.request_id!=b.request_id and a.external_req_id==b.external_req_id
    abortenv={}
    exec(methods(by['vllm/vllm/v1/engine/output_processor.py'],['abort_requests'],'OutputProcessor'),abortenv)
    def processor():
        return S(request_states={i:S(external_req_id='external-same',queue=None,lora_name=None) for i in [a.request_id,b.request_id]},
                 external_req_ids={'external-same':[a.request_id,b.request_id]},parent_requests={},
                 lora_states=S(request_finished=lambda *args:None),_update_admission_stats=lambda:None)
    p=processor();one=abortenv['abort_requests'](p,[a.request_id],True)
    assert one==[a.request_id] and b.request_id in p.request_states
    p=processor();all_ids=abortenv['abort_requests'](p,['external-same'],False)
    assert set(all_ids)=={a.request_id,b.request_id} and not p.request_states
    penv={}
    exec(methods(by['vllm/vllm/sampling_params.py'],['update_from_generation_config'],'SamplingParams'),penv)
    normalized=[]
    for ignore in [False,True]:
        # __post_init__ has already copied explicit stop IDs into all-stop IDs.
        p=S(ignore_eos=ignore,_eos_token_id=None,_all_stop_token_ids={7},stop_token_ids=[7])
        penv['update_from_generation_config'](p,{'eos_token_id':[2,3]},2)
        normalized.append(dict(ignore_eos=ignore,eos=p._eos_token_id,user_stop_ids=sorted(p.stop_token_ids),all_min_token_ids=sorted(p._all_stop_token_ids)))
    assert normalized[1]==dict(ignore_eos=True,eos=None,user_stop_ids=[7],all_min_token_ids=[2,3,7])
    # Same existing scheduler predicate, not a newly invented optimistic length.
    scheduler=by['vllm/vllm/v1/core/sched/scheduler.py'];tree=ast.parse(scheduler.read_text())
    cond=next(n.test for n in ast.walk(tree) if isinstance(n,ast.If) and n.lineno==617)
    mod=ast.parse('def guaranteed_after_schedule(request):\n return False\n')
    mod.body[0].body[0].value=cond;tenv={}
    exec(compile(ast.fix_missing_locations(mod),str(scheduler),'exec'),tenv)
    guaranteed=tenv['guaranteed_after_schedule']
    # P1000, max13; post schedule of6 inputs and6 placeholders.
    fixture=[]
    for committed in [8,12]:
        req=S(num_computed_tokens=1000+committed-1+6,num_output_placeholders=6,num_prompt_tokens=1000,max_tokens=13)
        fixture.append(dict(committed_before=committed,computed_after=req.num_computed_tokens,
                            placeholders_after=6,guard=guaranteed(req)))
    assert not fixture[0]['guard'] and fixture[1]['guard']
    # Exact cap trimming can be smaller than GPU valid sampled-token count.
    status=S(FINISHED_STOPPED='STOP',FINISHED_LENGTH_CAPPED='LENGTH',FINISHED_REPETITION='REPETITION')
    stopenv=dict(RequestStatus=status)
    exec(methods(by['vllm/vllm/v1/core/sched/utils.py'],['check_stop']),stopenv)
    stopenv['check_sequence_repetition']=lambda *args:False
    exec(methods(scheduler,['_update_request_with_output'],'Scheduler'),stopenv)
    req=S(pooling_params=None,sampling_params=S(eos_token_id=None,stop_token_ids=[],min_tokens=0,repetition_detection=None),
          output_token_ids=[10]*12,num_output_tokens=12,num_tokens=1012,max_tokens=13,status='RUNNING')
    def append(token):
        req.output_token_ids.append(token);req.num_output_tokens+=1;req.num_tokens+=1
    req.append_output_token_ids=append
    proposed=[21,22,23];visible,stopped=stopenv['_update_request_with_output'](S(max_model_len=34816),req,proposed)
    assert visible==[21] and stopped and req.status=='LENGTH'
    waves,hist,_,_,_=records();last={};first={}
    for wave,rs in sorted(waves.items()):
        for dp in [0,1]:
            for slot in rs[8*dp].get('scheduled',{}):
                first.setdefault(slot,(wave,dp));last[slot]=(wave,dp)
    assert set(last)==set(map(str,range(8)))
    terminal_waves=sorted({v[0] for v in last.values()})
    assert max(v[0] for v in first.values())<min(terminal_waves)
    spans=[]
    for wave in terminal_waves:
        local=[]
        for rank,r in sorted(waves[wave].items()):
            if 'target' not in r['calls']:continue
            aa=r['calls']['target']['end'];zz=r['calls']['draft']['end'];assert aa<zz
            local.append(dict(rank=rank,start_after_target_score_ns=aa,last_draft_score_ns=zz,ms=(zz-aa)/1e6))
        spans.append(dict(wave=wave,local=local,max_local_ms=max(r['ms'] for r in local)))
    endpoints=json.loads((BASE/'full-review/GRAPH-REQUEST-PREFIX-BUDGETS.json').read_text())['endpoint_ms']
    batch=max(endpoints['makespan']);total=sum(r['max_local_ms'] for r in spans)
    ordinary_path=BASE/'full-review/runs/ordinary-load-9144f2d04ace-result.json.gz'
    ordinary=json.loads(gzip.decompress(ordinary_path.read_bytes()))
    client=OUT/'request-lifecycle-ordinary-client.py'
    assert sha256(client.read_bytes()).hexdigest()==ordinary['contract']['files']['probes/client.py']
    ordinary_rows=[];versions={}
    for phase,groups in ordinary['generations'].items():
        for g in groups.values():
            for r in g['requests']:
                c=r['content_chunk_ns'];assert len(c)>1
                gap=(r['end_ns']-c[-2])/1e6
                ordinary_rows.append(dict(phase=phase,name=r['name'],input=r['input_tokens'],max_output=r['output_tokens'],
                    chunks=len(c),elapsed_ms=r['elapsed_ms'],TTFT_ms=r['ttft_ms'],
                    final_ordered_delivery_gap_upper_ms=gap,upper_elapsed_percent=100*gap/r['elapsed_ms'],
                    one_percent_endpoint_budget_ms=.01*r['elapsed_ms'],request_sha256=r['request_sha256'],output_sha256=r['output_sha256']))
                versions.setdefault(r['name'],[]).append(r)
    pairing=[dict(name=n,same_input=len({r['request_sha256'] for r in rr})==1,
                  same_output=len({r['output_sha256'] for r in rr})==1) for n,rr in versions.items()]
    assert len(ordinary_rows)==18 and all(r['same_input'] for r in pairing)
    native_sources=[Path(p) for p in [RECEIPT,BODY,SPANS,REPLAYS,SCHEDULE,ARRIVAL]]
    for p in [Path(RECEIPT),Path(BODY)]:
        pin=json.loads(p.read_text()).get('cache')
        if isinstance(pin,dict):
            q=Path(pin['path']);assert sha256(q.read_bytes()).hexdigest()==pin['sha256'];native_sources.append(q)
    native_sources.append(BASE/'tools/price_vocab_selection_fusion.py')
    out=dict(status='PARTIAL_REQUEST_STOP_ERROR_AND_NO_NEXT_DEMAND_REVIEW',whole_chain_complete=False,
       precision='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',source_identity='request-lifecycle-execution-identity.json',
       original_control_fixtures=dict(internal_ID_distinguishes_same_external=True,external_abort_targets_both=True,
           internal_abort_targets_one=True,normalization=normalized,
           terminal_predicate=fixture,cap_trim=dict(GPU_valid_count_fixture=3,visible_new_tokens=1,status=req.status),
           scope='Original function bodies/condition with inert UUID/allocator/parameter fixtures; no actual random-collision bound, live flag assertion, model/NPU or bit gate.'),
       current_protocol=dict(async_batch_queue_size=2,source_default_request_randomization=True,
           randomization_live_flag='Not inspected: source supports opt-out; designs require explicit generation regardless, so no decision-changing probe.',
           abort='Both eager abort queue and ordered input queue, scheduler idempotent. Process before old future update; tombstone/order and late-output dropping do not imply physical memory retirement.',
           async_counts='Bookkeeping returns empty valid CPU list and stores -1 placeholders. Device accepted count, visible clipped output count and CPU optimistic length are different.',
           old_target_guard='scheduler.py617-631 already blocks a guaranteed terminal extra target step; no blanket new40-layer credit.',
           no_next_demand_host_flag='Carry same post-schedule conservative terminal predicate in immutable per-request/step generation; no extra accepted-count D2H required for this subset. Actual frequency not recorded in archived scheduled-only packets.',
           RPC='Zero-copy input frames retain storage owners; mutable encoded output bytearray waits first-frame tracker before reuse. Object lifetime is not permission to mutate its bytes.',
           errors='Preprocessing error/MM retry are request-scoped; execution exception rethrows, EngineCore fatal sends ENGINE_CORE_DEAD then shutdown, client/output handler propagates. Do not turn failure into successful empty output.'),
       original_C8_terminal_gpu_facing_tail=dict(last_scheduled_by_slot=last,terminal_waves=spans,
           boundary='after original target head/verification last native task -> last draft score; excludes later host/D2H/socket EOF and earlier source planning',
           favorable_all_terminal_wave_tails_free_ms=total,baseline_batch_ms=batch,
           favorable_original_fixed_schedule_batch_percent=100*total/batch,TTFT_saved_ms=0,
           paid_conditional=[dict(rho=rho,h_per_terminal_wave_ms=h,net_ms=rho*total-h*len(spans),
                                  batch_percent=100*(rho*total-h*len(spans))/batch)
                             for rho in [0,.5,1] for h in [.1,.25]],
           qualification='Relaxed oracle erases all-rank tail on every any-slot terminal wave even if peers/rows live and even if host guard not eligible. Local-span max per phase, not sum16. Original work/arrivals/geometry/phase order fixed; conditional exposure, no measured speedup.',
           decision='STOP same independent original C8 GPU-facing terminal-tail1pct budget, not host/queued-delivery enlargement or six ordinary'),
       six_ordinary_endpoint_sensitivities=ordinary_rows,ordinary_pairing=pairing,
       ordinary_contract='Exact archived client source SHA pins max_tokens to six output caps126/13/300/20/126/13 and ignore_eosTrue/temperature0. Rows are each original arm separately. Last ordered-delivery gap is only an endpoint relaxation with prior visible prefix fixed, NOT local native tail duration or net saving. Four of six names have different output hashes across arms; never treat arms as before/after or candidate bit acceptance.',
       candidate=dict(name='No-next-demand terminal DSpark/provider/output/queued-input joint delivery',state='待确认',
           source_design='Use existing conservative host terminal bound for a subset; accepted-count extension needs authoritative device counts and ordinary parent time. Preserve zero draft placeholders and valid counts, all-DP/TP agreement and still-live EP service; _skip_drafting currently runs dummy bodies, so flag-only call does not remove them.',
           cache='Even terminal generated outputs do not prove context/cache producers dead. Retain published prefix-replay/cache writes or prove owner/range retirement; query body/head/Markov future-only values and context producers have different consumers.',
           queued='No40-layer credit for already suppressed target. Any accepted-based extra cancellation needs actual admitted future/row IDs, no real-peer bit/layout/collective change and preserved error/abort semantics.',
           ordinary_gap='Whole endpoint upper remains nontrivial on short caps, but native/CPU tail parent, accepted-count/host-guard eligibility and newly introduced fence/collective/lease costs missing; do not import C8 timing.',
           overlaps=['metadata-host immutable BatchFrame/early output','DSpark three-body original-geometry path','global-empty/inactive-DP EP-service protocol']),
       new_implementable_admission=[],new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[],
       inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [OUT/'request-lifecycle-execution-identity.json',ordinary_path,client,BASE/'full-review/GRAPH-REQUEST-PREFIX-BUDGETS.json',Path(__file__),*native_sources]])
    (OUT/'request-lifecycle-frontier.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(fixtures='PASS source-control only',terminal_waves=terminal_waves,C8_all_free_ms=total,
       C8_favorable_batch_percent=100*total/batch,ordinary_rows=len(ordinary_rows),ordinary_pairing=pairing,
       new_native_acceptance=False,whole_chain_complete=False)))


if __name__=='__main__':main()
