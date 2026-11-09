#!/usr/bin/env python3
"""Source/history qualification and paid net-delta sensitivity, not measured gain."""
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys

sys.dont_write_bytecode = True
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'))
from price_vocab_selection_fusion import records, ENDPOINTS, ORDINARY, GUARDS, SCHEDULE

def read(p): return json.loads(Path(p).read_bytes())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha(raw): return hashlib.sha256(raw).hexdigest()

def load_renderer(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def main():
    identity=read(OUT/'metadata-host-execution-identity.json')
    sources={s['runtime_path']:s for s in identity['sources']}
    for s in sources.values():assert digest(s['local_path'])==s['runtime_sha256']
    cp=sources['/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py']
    stock=Path(cp['local_path']).read_bytes()
    cp_path=BASE/'probes/cp_length_elision/patch.py'
    cpu_path=BASE/'probes/cp_local_plan/patch.py'
    cp_render=load_renderer('qualified_cp_renderer',cp_path)
    cpu_render=load_renderer('qualified_cpu_renderer',cpu_path)
    elided=cp_render.render(stock);cached=cpu_render.render(elided)
    old=read(BASE/'full-review/METADATA-PUBLICATION-FRONTIER.json')
    expected=next(s['admitted_source_sha256'] for s in old['source'] if s['runtime_path']==cp['runtime_path'])
    assert sha(cached)==expected,(sha(cached),expected)
    profile_path=BASE/'full-review/CPU-PLAN-RESIDUAL-ANALYSIS.json'
    profile=read(profile_path)
    mismatches=[]
    for p, expected_hash in profile['inputs'].items():
        actual=digest(p) if Path(p).is_file() else None
        if actual!=expected_hash:mismatches.append(dict(path=p,expected=expected_hash,actual=actual))
    assert not mismatches,mismatches
    body=profile['CPU_windows'];assert len(body)==128
    assert all(w['calls']==26 and w['CPU_plan_hits']==w['batch_hits']==25 for w in body)
    host_stages=['CPU_plan_lookup_or_build','qsl_pin_or_cached_copy_submission','length_branch',
                 'local_common_and_maxima','query_views_and_config','CPU_plan_store','return_replace']
    same_generation_joint=[sum(w['stage_ms'][k] for k in host_stages+['global_parent','local_parent']) for w in body]
    qualification=dict(status='HISTORICAL_RENDER_AND_PROFILE_INPUTS_QUALIFIED_WITH_SCOPE',
        current_stock_cp_sha256=sha(stock),old_cp_elision_sha256=sha(elided),old_cpu_plan_sha256=sha(cached),
        historical_expected_cp_sha256=expected,
        inputs=[dict(path=str(p),sha256=digest(p)) for p in [cp_path,cpu_path,BASE/'probes/cp_local_plan/wio_cp_cpu_plan.py',profile_path,
                    BASE/'full-review/CPU-PLAN-CANN-API-ATTRIBUTION.json',BASE/'full-review/CPU-PLAN-C8.json']],
        profile_input_hashes_checked=len(profile['inputs']),mismatches=mismatches,
        per_generation_host_enclosing_ms=dict(n=len(body),minimum=min(same_generation_joint),median=median(same_generation_joint),maximum=max(same_generation_joint)),
        retained_parent_pair=profile['CPU_summary']['grouped_enclosing_ms']['parent_pair'],
        retained_plan_view_result=profile['CPU_summary']['grouped_enclosing_ms']['shared_plan_views_and_result'],
        caveats=['Old CPU cohort uses CP elision plus CPU-plan and diagnostic stage recording; current pinned disk CP is stock.',
                 'B4/T24/actual4/causal matches one legal current shape, not the original entire C8 schedule or six ordinary loads.',
                 'Host elapsed contains instrumentation, Python, submission, waits and backpressure. It is an enclosing component budget, not removable compute or expected gain.',
                 '128 host generations and112 native periods differ; never pair by ordinal or add across ranks.',
                 'Historical C8 CPU-plan comparison is 10.363959%/9.560014% makespan vs its own CP+slot controls; text/selected metadata only, no every-primitive bitwise admission. Do not add it as a new joint-producer gain.'])
    (OUT/'metadata-host-history-qualification.json').write_text(json.dumps(qualification,ensure_ascii=False,indent=2)+'\n')

    waves,history,schedule,_,_=records()
    endpoints=read(ENDPOINTS)['endpoint_ms'];clients=schedule['client']
    assert [c['slot'] for c in clients]==list(range(8))
    counts={m:[0]*8 for m in endpoints};active_waves=[]
    for n,ranks in waves.items():
        entries=[r['entry'] for r in ranks.values() if r.get('scheduled')]
        if not entries:continue
        active_waves.append(n)
        for metric in endpoints:
            for slot,c in enumerate(clients):
                deadline=c['end_ns'] if metric=='makespan' else c['nonempty_chunk_ns'][0]
                if min(entries)<=deadline:counts[metric][slot]+=1
    def metric_price(metric, delta_ms):
        old=endpoints[metric];saved=[k*delta_ms for k in counts[metric]]
        gain=max(old)-max(v-d for v,d in zip(old,saved))
        return dict(saved_ms=gain,percent=100*gain/max(old),per_request_saved_ms=saved)
    thresholds={}
    for metric,values in endpoints.items():
        target=max(values)*.99
        thresholds[metric]=max(max(0,v-target)/n for v,n in zip(values,counts[metric]))
    ordinary=[];guards=read(GUARDS)['stages']
    for r in read(ORDINARY)['ordinary_requests']:
        worker,= [v for v in guards[r['stage']]['requests'] if v['name']==r['request']]
        n=1+r['draft_verifications']
        ordinary.append(dict(request=r['request'],target_iterations=n,draft_verifications=r['draft_verifications'],
            one_percent_response_required_net_ms_per_iteration={k:.01*worker[k]/n for k in ('stock_elapsed_ms','restored_stock_elapsed_ms')},
            one_percent_TTFT_required_net_ms_for_one_prefill={k:.01*worker[k] for k in ('stock_ttft_ms','restored_stock_ttft_ms')},
            response_sensitivity={str(d):{k:100*n*d/worker[k] for k in ('stock_elapsed_ms','restored_stock_elapsed_ms')} for d in (.1,.5,1,2,3)}))
    price=dict(status='PAID_NET_DELTA_SENSITIVITY_NOT_SPEEDUP',active_C8_waves=len(active_waves),
        counts_by_deadline=counts,C8_one_percent_required_net_ms_per_eligible_wave=thresholds,
        C8_sensitivity={str(d):{m:metric_price(m,d) for m in endpoints} for d in (.1,.5,1,2,3)},
        short_slot7_one_percent_required_net_ms_per_wave=.01*endpoints['TTFT_P90'][7]/counts['TTFT_P90'][7],
        first_output_only_reference=dict(short_slot7_one_percent_required_net_ms=.01*endpoints['TTFT_P90'][7],
            P90_one_percent_required_net_ms=.01*max(endpoints['TTFT_P90']),
            qualification='One actual first-output contribution per request; no credit from later waves merely started before the first client chunk. The multiwave sensitivity above may over-credit asynchronous work not causally before that chunk.'),
        ordinary_requests=ordinary,
        scope='Net delta is after all descriptor/guard/event/copy/update costs and lost host overlap. Give it once per real wave, fully exposed and serially propagated to each inclusive archived deadline. No credit for warmup or isolated dummy-only waves. This is a favorable external-schedule sensitivity, not a universal upper bound.',
        qualifications=['Current CPU DP allreduce duration is not identified by NPU-only C8 DB: no Python/PyTorch/OSRT tables. No Gloo wait is erased or assigned to CPU-plan.',
                        'Separate legacy residual timing and C8/ordinary request envelopes. Do not transfer9ms host elapsed directly into these delta inputs.',
                        'Real exposure, branch eligibility and host/device/peer readiness may reduce credit to zero. Ordinary iteration counts are observed, same delta across shapes is hypothetical.',
                        '1% is historical work-ranking reference, not user accuracy or performance acceptance.'],
        inputs=[dict(path=str(p),sha256=digest(p)) for p in [ENDPOINTS,ORDINARY,GUARDS,SCHEDULE]],new_NPU_work=0,new_profile_collection=0)
    budgets={}
    for fraction in (.25,.5,.75):
        values=[(w['stage_ms']['CPU_plan_lookup_or_build']+w['stage_ms']['qsl_pin_or_cached_copy_submission'])*25/26
                +fraction*(w['stage_ms']['local_common_and_maxima']+w['stage_ms']['query_views_and_config']) for w in body]
        budgets[str(fraction)]=dict(minimum=min(values),median=median(values),maximum=max(values))
    scenarios=[]
    for fraction in (.25,.5,.75):
        gross=budgets[str(fraction)]['median']
        for rho in (0,.5,1):
            # rho combines source-proved removability and critical-path exposure;
            # h is new nonoverlapped frame/guard/publication overhead. Both are hypotheses.
            h=.25;delta=rho*gross-h
            scenarios.append(dict(shared_common_fraction=fraction,removable_exposed_fraction=rho,
                new_nonoverlapped_cost_ms=h,conditional_net_ms_per_iteration=delta,
                C8={m:metric_price(m,delta) for m in endpoints}))
    price['conditional_local_net_model']=dict(
        formula='gross=(CPU_plan_lookup_or_build+qsl_submission)*25/26 + f*(local_common_and_maxima+query_views_and_config); net=rho*gross-h',
        retained=['all global/local parent builders','per-group return_replace','original first producer/copy','CPU_plan_store','all model/math/collectives outside these fields'],
        paired_old_B4_gross_ms_by_f=budgets,scenarios=scenarios,
        required_rho_for_C8_one_percent_makespan_with_h025={f:(thresholds['makespan']+.25)/r['median'] for f,r in budgets.items()},
        qualification='A conditional mechanism model using the qualified diagnostic B4 residual cohort. The25/26 removal, f, rho and h are unmeasured hypotheses; no transfer of these timings to current stock C8 or ordinary as observed data. Preserving parent builders avoids crediting their required group-specific work. Negative net cases remain regressions, not clipped to zero.')
    (OUT/'metadata-host-net-price.json').write_text(json.dumps(price,ensure_ascii=False,indent=2)+'\n')

    ranges={
      'worker/model_runner_v1.py':[[825,885],[2548,2620],[3250,3270],[3325,3432],[3449,3907],[6120,6150],[6487,6493]],
      'spec_decode/dspark_proposer.py':[[31,106],[226,357],[366,429]],
      'spec_decode/llm_base_proposer.py':[[1013,1114],[1782,1915],[1960,2035],[2389,2488],[2560,2650]],
      'attention/context_parallel/dsa_v41_cp.py':[[52,95],[112,185]],
      'attention/dsa_v41.py':[[879,927],[929,1010],[1325,1424]],
      'utils.py':[[1304,1361]],'ascend_forward_context.py':[[373,425]],
      'worker/device_metadata.py':[[46,155]]}
    frontier=dict(status='PARTIAL_METADATA_HOST_JOINT_PRODUCER_REVIEW',whole_chain_complete=False,
        precision='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',
        identity=dict(path=str(OUT/'metadata-host-execution-identity.json'),sha256=digest(OUT/'metadata-host-execution-identity.json'),sources=len(sources)),
        reviewed_ranges=[dict(runtime_path='/vllm-workspace/vllm-ascend/vllm_ascend/'+p,ranges=rs) for p,rs in ranges.items()],
        CPU_group_source=dict(runtime_path='/vllm-workspace/vllm/vllm/distributed/parallel_state.py',
            ranges=[[342,359],[492,522],[2052,2058],[2188,2204]],
            frozen_pairs=[[i,i+8] for i in range(8)],backend='CPU Gloo; eight two-rank DP groups, not one sixteen-rank allreduce'),
        actual_coordination=dict(target='_determine_batch_execution_and_padding -> CPU DP packed counts/modes all_reduce -> min graph mode / role-dependent padding -> redispatch',
            draft='Actual AscendDSparkProposer.set_inputs_first_pass returns B*num_query_per_req; current sample_from_anchor K5 means5B. GPU rejected counts alter positions/lengths, not padded query count.',
            skip='Dense/KV-consumer/hierarchy/MegaMoe guards are distinct; frozen target/draft MoE with no KV consumer does not take dense/consumer skip.',
            counterexample='Target bucket24 does not imply draft24: B4→20, B1→5; sample_from_anchor=False uses K+1 instead. Dummy min/reordered/K0/skip paths need matching collectives.'),
        candidates=[
          dict(name='Explicit target+draft DP packet, not target bucket reuse',status='待确认',
               protocol='At first DP coordination exchange epoch/role/target local count and mode, actual B/Q and future draft query count, active/dummy/skip eligibility. Keep target min-mode/padding and draft max-count rules distinct. Reuse only if all participants agree; otherwise all take the same original later collective exactly once.',
               gaps=['complete empty-DP/dummy/K0/input-fits/error/abort role sequence','source-known counts do not imply positions/slots/lengths known early','matched all-rank CPU collective overhead vs peer readiness; no numerical gain from erasing wait']),
          dict(name='Same-generation global/local BatchFrame plus group resource assembly',status='待确认，值得深入',
               protocol='One immutable local plan and one global plan per runner build dictionary; preserve canonical batch owner and device/CPU values. Share CP range/qsl/suffix/length views/maxima only across compatible same-generation calls. Keep each resource block table, slot mapping, compressor state and native task frontiers distinct; materialize original per-group results.',
               incremental_work='CP length elision and CPU plan already have historical coverage. New increment must remove remaining view/result/config/parent assembly or grouped publication; 25 duplicate H2D removals alone are not the entire parent gain.',
               gaps=['full scalar/tensor/alias observer contract including callback closures','device metadata last-read/reuse for V1 eager/full/dummy and empty query','exact native primitive bytes, graph stable addresses and descriptor assembly cost']),
          dict(name='Shape-only cached return object or metadata mini graph',status='已停止（该原样方案）',
               reason='B/graph bucket alone omits seq_lens/max lengths, query partition, actual/causal counts, physical tables/slots and per-invocation reset/publish side effects. V1 FULL still submits producers on the host with external events, not V2 captured producers. A correct graph update/explicit frame protocol remains separate and pending.'),
          dict(name='Device-only metadata/DMA micro fusion in old residual cohort',status='已停止同一独立1%局部预算',
               reason='Old112-period all-native-prefix-free fraction max0.8183775%; retain host/submission. This does not prune the larger host joint producer, other workloads, or claim a C8 universal bound.')],
        history_qualification='metadata-host-history-qualification.json',net_price='metadata-host-net-price.json',
        V1_mode_correction='Actual V1 constructs DeviceMetadataExecutor(capture_producers=False), builds/submits every invocation, FULL passes batch_descriptor for external events, and _model_forward finally calls release. TargetDeviceMetadata.finish/capture_producers=True describes a different path; do not borrow its all-task join as V1 proof.',
        new_NPU_work=0,service_requests=0,remote_writes=[],
        next_action='Trace every BatchFrame observer and V1 producer release/last-read before a concrete grouped metadata implementation. Keep the explicit DP packet role/empty/error and Gloo net-cost gaps; do not repeat CPU-plan or credit its historical gain twice. Then continue final-layer attention/FFN row-demand producer and remaining input/embedding/Engram paths.')
    (OUT/'metadata-host-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(source_pins=len(sources),old_cpu_source_reproduced=True,profile_inputs_checked=len(profile['inputs']),
        C8_waves=len(active_waves),counts=counts,one_percent_net_threshold_ms=thresholds,short_threshold_ms=price['short_slot7_one_percent_required_net_ms_per_wave'],
        residual_host_median_ms=median(same_generation_joint),new_NPU_work=0)))

if __name__=='__main__':main()
