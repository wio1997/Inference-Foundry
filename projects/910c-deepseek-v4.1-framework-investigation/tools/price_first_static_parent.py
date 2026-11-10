#!/usr/bin/env python3
"""Price entire source-bounded first-static prefixes on original C8 clocks."""
from pathlib import Path
from hashlib import sha256
from collections import defaultdict,Counter
from statistics import median
import json,sys,zlib
sys.dont_write_bytecode=True
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104');ROOT=Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from price_vocab_selection_fusion import records
from price_attention_send_epilogue import ENDPOINTS
from analyze_final_producer_entry import latest_dp_change
def main():
    receipt=json.loads((ROOT/'first-static-parent-receipt.json').read_text());assert not receipt['unqualified'] and receipt['windows']==3696
    math_receipt=json.loads((ROOT/'first-static-math-receipt.json').read_text());assert math_receipt['qualified']==3624 and not math_receipt['unqualified']
    p=Path(math_receipt['raw_path']);assert sha256(p.read_bytes()).hexdigest()==math_receipt['raw_sha256'];data=json.loads(zlib.decompress(p.read_bytes()))
    waves,hist,schedule,_,_=records();offsets=hist['clock_contract']['offset_upper_ns'];by={};population=Counter()
    for rank in data:
        previous_draft_end=0
        for w in rank['windows']:
            if w['active']:
                assert w['target_start_task'][1]>=previous_draft_end,('prior_draft_embedding_confusion',rank['rank'],w['wave'])
                scheduled=sum(waves[w['wave']][rank['rank']]['scheduled'].values());B=len(waves[w['wave']][rank['rank']]['scheduled'])
                assert w['target_start_task'][7]==f'"{scheduled},5120"',('target_embedding_shape',rank['rank'],w['wave'])
                shape=w['draft_start_task'][7].strip('"').split(',');assert len(shape)==2 and shape[1]=='5120';M=int(shape[0])
                assert M%5==0 and B<=M//5<=min(4,B+1),('draft_metadata_padding_population',rank['rank'],w['wave'],B,M)
                population[B,M,w['kind']]+=1
                by[w['wave'],rank['rank']]=w
            previous_draft_end=w['draft_score_end_ns']
    groups=[];costs={};local=defaultdict(list)
    for wave in waves:
        dpvalues=[]
        for dp in [0,1]:
            ranks=list(range(dp*8,dp*8+8))
            if not all((wave,k) in by for k in ranks):continue
            ws=[by[wave,k] for k in ranks];o=[[offsets[i][j] for j in ranks] for i in ranks]
            target=latest_dp_change([w['target_prefix_end_task'][1] for w in ws],[w['target_start_task'][1] for w in ws],o)
            draft=latest_dp_change([w['draft_prefix_end_task'][1] for w in ws],[w['draft_start_task'][1] for w in ws],o)
            target_math=latest_dp_change([w['target_pre_rope_end_task'][2] for w in ws],[w['target_first_HcPre'][1] for w in ws],o)
            draft_math=latest_dp_change([w['draft_pre_rope_end_task'][2] for w in ws],[w['first_HcPre_ns'] for w in ws],o)
            bounds=dict(target_first_static=target,draft_first_static=draft,joint_first_static=[target[i]+draft[i] for i in [0,1]],joint_HC_and_QKV=[target_math[i]+draft_math[i] for i in [0,1]])
            groups.append(dict(wave=wave,DP=dp,kind=ws[0]['kind'],bounds_ms=bounds,target_boundary_types=sorted({w['target_prefix_end_task'][5] for w in ws}),draft_boundary_types=sorted({w['draft_prefix_end_task'][5] for w in ws})))
            dpvalues.append(bounds)
            for k,v in bounds.items():local[k].append(v[1])
        if dpvalues:costs[wave]={k:max(d[k][1] for d in dpvalues) for k in dpvalues[0]}
    assert len(groups)==453 and len(costs)==227 and len(by)==3624
    endpoints=json.loads(Path(ENDPOINTS).read_text())['endpoint_ms'];client=schedule['client']
    def price(scope,rho=1.,h=0.):
        out={}
        for metric,base in endpoints.items():
            gains=[];counts=[]
            for c in client:
                deadline=c['end_ns'] if metric=='makespan' else c['nonempty_chunk_ns'][0]
                selected=[w for w in costs if min(v['entry'] for v in waves[w].values() if 'entry' in v)<=deadline]
                gains.append(sum(costs[w][scope]*rho-h for w in selected));counts.append(len(selected))
            saved=max(base)-max(a-b for a,b in zip(base,gains));out[metric]=dict(signed_saved_ms=saved,signed_percent=saved/max(base)*100,eligible_waves=counts,signed_per_request_ms=gains)
        return out
    gross={k:price(k) for k in next(iter(costs.values()))};scenarios=[dict(scope=k,removable_and_exposed_fraction=rho,new_ms_per_wave=h,result=price(k,rho,h)) for k in gross for rho in [0,.25,.5,1] for h in [0,.05,.1]]
    old_path=ROOT/'embedding-context-frontier.json';old=json.loads(old_path.read_text())
    qualified=next(x for x in old['archived_qualification'] if Path(x['path']).name=='EMBEDDING-ENGRAM-FUSION-ESTIMATES.json')
    assert qualified['input_pin_match']
    for pin in qualified['pins']:assert sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['expected']
    cache_upper={}
    for metric,v in gross['joint_HC_and_QKV'].items():
        named=max(old['archived_narrow_embedding']['gross_ms'][metric]);math_max=max(v['signed_per_request_ms']);base=max(endpoints[metric]);cache_upper[metric]=dict(named_embedding_favorable_endpoint_ms=named,first_math_max_request_credit_ms=math_max,combined_replay_upper_ms=named+math_max,combined_replay_upper_percent=100*(named+math_max)/base)
    out=dict(status='QUALIFIED_ORIGINAL_C8_WHOLE_FIRST_STATIC_PREFIX_GROSS_AND_PAID_SENSITIVITY',source_derivation='embedding-entry-primary.json',receipt='first-static-parent-receipt.json',math_receipt='first-static-math-receipt.json',active_rank_windows=len(by),TP8_groups=len(groups),active_waves=len(costs),groups=groups,gross=gross,scenarios=scenarios,draft_lookup_population=[dict(active_requests=B,query_embedding_rows=M,target_kind=kind,rank_windows=n) for (B,M,kind),n in sorted(population.items())],
        scope='Target: last current-step standard-table embedding before body through first SMLA or earlier HCPost; includes external bank, captured internal lookup, initial SP/HC, normalization, input gather and Q/KV production including required waits. Draft: own query embedding through its first SMLA or earlier HCPost, after dynamic context panel production. Does not include all40layer QKV, full first attention, three contextWKV/main or all three draft bodies.',
        fixed_reader_cache_upper=cache_upper,fixed_reader_cache_upper_proof='Within the same saved fixed-schedule endpoint replay only: max(base-named-math)>=max(base-named)-max(math), hence combined credit <= old named favorable endpoint credit + max new per-request math credit. Named embedding/SP/outer-bank scope excludes firstHC arithmetic; firstHC-through-Qb/KVnorm completion math spans are disjoint. Empty-query ends at KVnorm completion, not the later dynamic Oproj/HCPost wait. Does not credit the interstitial host/Engram/graph-entry gap to a static-word cache, and does not bound changed attention/resource/layout parents.',
        qualification='Existing immutable original C8 only, exact target lookup shape and observed draft padded-request population plus chronology prevent choosing prior-step draft lookup. Clock-bounded most favorable active DP per wave; joint sums disjoint target/draft upper intervals. Fullprefix gross includes required interstitial host/Engram/graph-entry work and cannot all be claimed by static-word caching. Separate fixed-reader cache upper uses matched old named component plus max request math envelope within replay only. Necessary work, cachemiss, primitive bits, replica/cache memory and maintenance relaxed. Signed rho/h are sensitivities, not speedup. Six ordinary new whole-static parent not yet priced; no C8 percent transfer.',
        same_budget_decision='Only fixed-reader static cache/replication/precomputation child may be stopped by its complete replay upper; full prefix host/Engram/input-frame/graph-entry and larger QKV/HC/context/layout parents are separate. Old1percent reference is prioritization, not user hard acceptance.',
        local_upper_ms={k:dict(n=len(v),min=min(v),median=median(v),max=max(v)) for k,v in local.items()},math_scope='Only first-HC through later of Qb or512KVnorm completion; excludes RoPE/cache write, attention, O projection and empty-query dynamic wait.',physical_DP_qualification='453 active-request TP8 groups of454 physically present DP groups; 8 synthetic rank windows with no scheduled actual request not priced by this reader. Does not claim complete all-physical-group upper.',new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],inputs=[dict(path=str(x),sha256=sha256(x.read_bytes()).hexdigest()) for x in [ROOT/'embedding-entry-primary.json',ROOT/'first-static-parent-receipt.json',ROOT/'first-static-math-receipt.json',old_path,p,Path(ENDPOINTS)]],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'first-static-parent-price.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(gross=gross,local_upper=out['local_upper_ms'],new_NPU_work=0)))
if __name__=='__main__':main()
