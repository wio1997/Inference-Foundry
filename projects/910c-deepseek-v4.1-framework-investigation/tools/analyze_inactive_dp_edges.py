#!/usr/bin/env python3
"""Bind finished profile edges and distinguish wait/arrival/recording cutoff.

All prices are conditional original-schedule sensitivities, not speedups.
"""
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import statistics
import struct
import sys
import zlib

sys.dont_write_bytecode=True
OUT=Path('audit/execution-search')
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
sys.path.insert(0,str(BASE/'tools'))
from price_vocab_selection_fusion import cached,BODY,records


def load(p):return json.loads(Path(p).read_bytes())
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def ns(v):return int(Decimal(str(v))*1000)
def start(e):return ns(e['ts'])
def end(e):return start(e)+ns(e.get('dur',0))
def stats(v):return dict(n=len(v),min=min(v),median=statistics.median(v),max=max(v))


def main():
    receipt=load(OUT/'inactive-dp-framework-edges-receipt.json')
    assert digest(receipt['path'])==receipt['sha256']
    raw=json.loads(zlib.decompress(Path(receipt['path']).read_bytes()))
    files={r['rank']:r for r in raw['files']};assert set(files)==set(range(16))
    old_summary=load('audit/framework-profile-171/framework-summary.json')
    inventory={r['file']:r for r in old_summary['files']}
    cycles={};gloo={};counts=[]
    for rank,r in sorted(files.items()):
        rel=r['path'].split('/framework-profile-20261109T192807Z/')[1]
        assert r['bytes']==inventory[rel]['bytes']
        sel=r['selected']
        tok=sorted([e for e in sel if e['name'] in ['_C_ascend::dispatch_ffn_combine','AscendCL@aclmdlRIExecuteAsync']],key=start)
        fc=sorted([e for e in sel if e['name']=='_C_ascend::dispatch_ffn_combine'],key=start)
        hp=sorted([e for e in sel if e['name']=='_C_ascend::npu_hc_pre_v3'],key=start)
        native=sorted([e for e in r['native'] if e['name']=='DispatchFFNCombine'],key=start)
        nhp=sorted([e for e in r['native'] if e['name']=='HcPre'],key=start)
        npost=sorted([e for e in r['native'] if e['name']=='HcPost'],key=start)
        assert len(hp)==2*len(fc) and len(nhp)==len(npost)==2*len(native)
        gi=sorted([e for e in sel if e['name']=='gloo:all_reduce'],key=start)
        gloo[rank]=gi;cs=[];i=0;fc_idx=0
        while i<len(tok):
            is_graph=tok[i]['name']=='AscendCL@aclmdlRIExecuteAsync'
            n=1 if is_graph else 40
            target=tok[i:i+n];draft=tok[i+n:i+n+3]
            assert len(target)==n and len(draft)==3
            assert all((e['name']=='AscendCL@aclmdlRIExecuteAsync')==is_graph for e in target)
            assert all(e['name']=='_C_ascend::dispatch_ffn_combine' for e in draft)
            first_cpu=start(target[0]) if is_graph else start(hp[fc_idx*2])
            if not is_graph:fc_idx+=40
            draft_cpu_start=start(hp[fc_idx*2]);fc_idx+=3
            j=len(cs);nf=native[j*43:(j+1)*43]
            assert len(nf)==43
            target_models={e['args']['Model Id'] for e in nf[:40]}
            assert (target_models!={4294967295})==is_graph
            assert len(target_models)==1
            assert {e['args']['Model Id'] for e in nf[40:]}=={4294967295}
            for k in range(j*43,(j+1)*43):
                assert start(nhp[2*k])<=start(nhp[2*k+1])<=start(native[k])
                assert start(npost[2*k])<=start(native[k])
                assert end(native[k])<=start(npost[2*k+1])
            assert end(gi[2*j])<=first_cpu
            assert end(target[-1])<=start(gi[2*j+1])
            assert end(gi[2*j+1])<=draft_cpu_start
            cs.append(dict(cycle=j,target_graph=is_graph,target_model=next(iter(target_models)),
                target_cpu_first_ns=first_cpu,target_cpu_last_ns=end(target[-1]),
                draft_cpu_first_ns=draft_cpu_start,draft_cpu_last_ns=end(draft[-1]),
                target_gloo=2*j,draft_gloo=2*j+1,
                first_native_HcPre_ns=start(nhp[2*j*43]),last_native_HcPost_ns=end(npost[2*(j+1)*43-1]),
                target_F21=40,draft_F21=3))
            i+=n+3
        assert len(gi)==2*len(cs) and fc_idx==len(fc) and len(native)==43*len(cs)
        cycles[rank]=cs
        counts.append(dict(rank=rank,cycles=len(cs),gloo=len(gi),native_DFC=len(native),
                           target_graph=sum(c['target_graph'] for c in cs),target_eager=sum(not c['target_graph'] for c in cs)))
    assert {r['cycles'] for r in counts if r['rank']<8}=={13}
    assert {r['cycles'] for r in counts if r['rank']>=8}=={15}
    pairs=[];phase_costs=[]
    for ordinal in range(26):
        phase=[]
        for tp in range(8):
            a,b=gloo[tp][ordinal],gloo[tp+8][ordinal]
            late=max(start(a),start(b));last=max(end(a),end(b))
            row=dict(ordinal=ordinal,kind='target_coord' if ordinal%2==0 else 'draft_coord',TP=tp,
                ranks=[tp,tp+8],backend_start_ns=[start(a),start(b)],backend_end_ns=[end(a),end(b)],
                arrival_skew_ms=abs(start(a)-start(b))/1e6,end_skew_ms=abs(end(a)-end(b))/1e6,
                latest_backend_start_to_last_end_ms=(last-late)/1e6,
                larger_local_wait_ms=max(ns(a['dur']),ns(b['dur']))/1e6)
            pairs.append(row);phase.append(row)
        phase_costs.append(dict(ordinal=ordinal,kind=phase[0]['kind'],
            earliest_backend_start_ns=min(min(r['backend_start_ns']) for r in phase),
            last_backend_end_ns=max(max(r['backend_end_ns']) for r in phase),
            max_TP_late_backend_span_ms=max(r['latest_backend_start_to_last_end_ms'] for r in phase),
            max_TP_local_wait_ms=max(r['larger_local_wait_ms'] for r in phase)))
    cr=load(OUT/'inactive-dp-framework-clients-receipt.json');assert digest(cr['path'])==cr['sha256']
    clients=json.loads(zlib.decompress(Path(cr['path']).read_bytes()))
    prof=next(r['data'] for r in clients if r['path'].endswith('/profiled.json'))
    ref=next(r['data'] for r in clients if r['path'].endswith('/reference.json'))
    req=sorted(prof['requests'],key=lambda r:r['index'])
    ttft=[(r['first_token_mono_ns']-r['start_mono_ns'])/1e6 for r in req]
    baseline_batch=(prof['end_mono_ns']-prof['start_mono_ns'])/1e6
    wall_offsets=[r[k+'_wall_ns']-r[k+'_mono_ns'] for r in req for k in ['start','end']]
    first_deadlines=[r['first_token_mono_ns']+(r['start_wall_ns']-r['start_mono_ns']) for r in req]
    last_first=max(first_deadlines)
    def price(cost_key,rho=1.,h=0.):
        gains=[];eligible=[]
        for deadline in first_deadlines:
            selected=[r for r in phase_costs if r['earliest_backend_start_ns']<=deadline]
            eligible.append(len(selected));gains.append(sum(rho*r[cost_key]-h for r in selected))
        saved=max(ttft)-max(a-b for a,b in zip(ttft,gains))
        return dict(TTFT_P90_saved_ms=saved,TTFT_P90_percent=saved/max(ttft)*100,
                    per_request_signed_ms=gains,eligible_phase_counts=eligible)
    suffix=[]
    for rank in range(8,16):
        for ordinal,e in enumerate(gloo[rank][26:],26):
            suffix.append(dict(rank=rank,ordinal=ordinal,kind='target_coord' if ordinal%2==0 else 'draft_coord',
                start_ns=start(e),end_ns=end(e),duration_ms=ns(e['dur'])/1e6,
                after_all_profiled_first_tokens=start(e)>last_first,
                DP0_partner_recorded=False))
    # Original observer-containing C8: full target40+draft EP participants.
    waves,hist,_,_,_=records();bodies=cached(BODY)['profiles'];offset=hist['clock_contract']['offset_upper_ns']
    assert not waves[3][8].get('scheduled') and waves[3][0].get('scheduled')
    draft_bound=load(BASE/'full-review/runs/dspark-decoder-f21-boundaries.json')
    f21=[];old_local=[]
    for rank,p in enumerate(bodies):
        b=p['bodies'][0];idx=set(b['F21_TASK_ROWS']);target=sorted([r for r in b['rows'] if r[0] in idx],key=lambda r:r[1])
        dr=[r for q in draft_bound['rows'] if q['rank']==rank for r in q['records'] if r[0]==416]
        assert len(target)==40 and len(dr)==3
        f21.append([[r[1],r[2]] for r in target]+[[r[5],r[6]] for r in dr])
        whole=(waves[3][rank]['calls']['draft']['end']-b['start_ns'])/1e6
        retained=sum((z-a)/1e6 for a,z in f21[-1])
        old_local.append(dict(rank=rank,local_HC_shape=b['HcPre_shape'],MoE_carrier=b['F21_carrier_rows'],
            whole_target_draft_ms=whole,original_43_DFC_ms=retained,complement_ms=whole-retained))
    def cross_bound(a,b):
        lo=max(min(a[i]-b[j]-offset[j][i] for j in range(8)) for i in range(8,16))
        hi=max(min(a[i]-b[j]+offset[i][j] for j in range(8)) for i in range(8,16))
        assert lo<=hi
        return [lo/1e6,hi/1e6]
    old_starts=[dict(layer=i,phase='target' if i<40 else 'draft',
        latest_idle_minus_latest_active_F21_start_ms=cross_bound(
            {r:f21[r][i][0] for r in range(16)},{r:f21[r][i][0] for r in range(16)})) for i in range(43)]
    tail=cross_bound({r:waves[3][r]['calls']['draft']['end'] for r in range(16)},
                     {r:waves[3][r]['calls']['draft']['end'] for r in range(16)})
    # Marker-derived representative target classification, under pinned text /
    # initialized-KV / graph-external Engram producer conditions only.
    front=load(OUT/'inactive-dp-frontend-edges.json');clock=load(OUT/'inactive-dp-frontend-clock.json')
    assert front['frontend']['sha256']==clock['sha256']
    base_ns=clock['metadata']['baseTimeNanoseconds']
    submits=[r for r in front['frontend']['events'] if 'add_request_async' in r['event']['name']]
    assert len(submits)==8
    ready=min(base_ns+start(r['event']) for r in submits)
    markers={}
    for q in front['rank_hash_markers']:
        rank=8*q['dp'];assert q['sha256']==files[rank]['sha256']
        h=[e for e in q['events'] if e.get('cat')=='cpu_op' and e['name']=='_engram_req_index_kernel']
        labels=[]
        for c in cycles[rank]:
            a=end(gloo[rank][c['target_gloo']]);z=start(gloo[rank][c['draft_gloo']])
            n=sum(a<=start(e)<z for e in h);assert n in [0,1]
            labels.append(dict(cycle=c['cycle'],hash_launch=n,
                interpretation='real target hash path' if n else 'consistent with dummy/no-request path'))
        assert sum(r['hash_launch'] for r in labels)==len(h)
        markers[rank]=labels
    empty_candidates=[i for i in range(13) if not markers[0][i]['hash_launch'] and not markers[8][i]['hash_launch']]
    startup=[]
    for i in empty_candidates:
        if i+1>=13:continue
        old_next=max(start(gloo[r][2*(i+1)]) for r in range(16))
        opened=min(start(gloo[r][2*i]) for r in range(16))
        startup.append(dict(cycle=i,first_request_submit_start_ns=ready,old_next_target_backend_start_ns=old_next,
            favourable_request_ready_clamped_ms=max(0,old_next-max(ready,opened))/1e6))
    total_startup=sum(r['favourable_request_ready_clamped_ms'] for r in startup)
    neg_zero=int.from_bytes(struct.pack('>f',-0.0)[:2],'big')
    add_zero=int.from_bytes(struct.pack('>f',-0.0+0.0)[:2],'big')
    out=dict(status='PARTIAL_WHOLE_INACTIVE_DP_AND_PROFILE_CAUSAL_EDGE_REVIEW',whole_chain_complete=False,
        precision='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',current_source_identity='inactive-dp-execution-identity.json',
        old_original_C8=dict(local_windows=old_local,all43_F21_arrival_bounds=old_starts,
            idle_ready_no_later_count=sum(r['latest_idle_minus_latest_active_F21_start_ms'][1]<0 for r in old_starts),
            latest_idle_minus_active_final_draft_end_ms=tail,
            contract='Archived observer-containing C8 first real wave only. HC3 idle side still pads MoE to416 and serves real peers. Complement is not savings. Earlier idle F21 entry and tail imply no original latest-entry-frontier credit for a pure local-prefix deletion with original active arrivals/EP service/resource placement fixed; this is not a bound on changed EP service, placement or ordinary loads.'),
        new_profile=dict(rank_counts=counts,cpu_native_cycles=cycles,paired_backend_edges=pairs,phase_costs=phase_costs,
            pair_end_skew_ms=stats([r['end_skew_ms'] for r in pairs]),
            phase_late_span_ms=stats([r['max_TP_late_backend_span_ms'] for r in phase_costs]),
            profiled_TTFT_P90_ms=max(ttft),profiled_batch_ms=baseline_batch,
            reference_batch_ms=(ref['end_mono_ns']-ref['start_mono_ns'])/1e6,
            client_wall_minus_monotonic_ns=[min(wall_offsets),max(wall_offsets)],
            all_profiled_first_tokens_before_ns=last_first,unmatched_DP1_suffix=suffix,
            gloo_only_recorded_TTFT_favourable=price('max_TP_late_backend_span_ms'),
            misleading_erase_all_local_wait_sensitivity=price('max_TP_local_wait_ms'),
            paid_recorded_TTFT_scenarios=[dict(rho=rho,added_ms_per_phase=h,result=price('max_TP_late_backend_span_ms',rho,h))
                for rho in [0,.5,1] for h in [.05,.1]],
            contract='Same new profiled C8 only. CPU ordered40+3 or graph+3 submissions match native Model-ID families and43 F21/86 HCPre/86 HCPost per cycle. Backend Sequence number=-1: pair by TP rank-group, preserved call order and completion agreement, not a recorded global sequence ID. 26 common calls/13cycles vs4 DP1-only calls; do not equate12 execute_model profiler steps to12 model cycles. Latest backend-start-to-last-end is an observed joint span under fixed upstream arrival, not pure network time or an implementation speedup. Prices optimistically expose all eligible common phases to each frontend endpoint; caller CPU/metadata and unrecorded tail are outside. Do not mix with original v168 pricing.'),
        cutoff=dict(log=raw['profile_log'],conclusion='DP0 synchronous automatic stop/export is logged19:36:45–46, DP1 stop19:36:46–47. The largest DP1 suffix waits have no recorded DP0 partners and occur after all first tokens. Strong indication of recording-boundary interference; exact fraction caused by stop vs other upstream work is not identifiable from these coarse log stamps. Never credit the whole~710ms Free/Gloo intersection as production gain.'),
        global_empty_joint_scope=dict(representative_hash_labels=markers,candidate_cycles=empty_candidates,
            frontend_submit_clock='Chrome ts*1000 + baseTimeNanoseconds; rejected raw host-monotonic bridge',
            earliest_frontend_submit_ns=ready,startup_bounds=startup,
            favourable_original_startup_ms=total_startup,
            favourable_profiled_P90_percent=total_startup/max(ttft)*100,
            favourable_profiled_batch_percent=total_startup/baseline_batch*100,
            paid_half_h01_ms=total_startup*.5-.1*len(startup),
            limits='Marker absence is source/config-conditional evidence on two DP representatives, not an explicit all16 real-work flag. Bound even erases next real preparation through its backend entry after frontend-ready clamp. Engine inbox acceptance, pending queue/cleanup, alias/event generations, lazy work and changed real numeric geometry must be retained/priced. No native admission. Joint BatchFrame/DP-real-work protocol overlaps existing metadata-host scope.'),
        bit_contract_witness=dict(recipe='all-false Engram mask => return hidden directly',
            BF16_negative_zero=hex(neg_zero),FP32_add_positive_zero_then_BF16=hex(add_zero),
            qualification='IEEE source-formula witness only, NOT a new NPU mismatch or a statement that frozen inputs contain this value. Original masked gate still performs FP32 add then BF16 cast. Generic algebraic identity is insufficient; preserve zero/sign/rounding or prove actual-domain equivalence.'),
        inputs=[dict(path=str(p),sha256=digest(p)) for p in [OUT/'inactive-dp-execution-identity.json',
            OUT/'inactive-dp-framework-edges-receipt.json',OUT/'inactive-dp-framework-clients-receipt.json',
            OUT/'inactive-dp-frontend-edges.json',OUT/'inactive-dp-frontend-clock.json',Path(BODY),
            BASE/'full-review/runs/dspark-decoder-f21-boundaries.json',Path(__file__)]],
        new_NPU_work=0,service_requests=0,new_profile_collection=0,remote_writes=[])
    (OUT/'inactive-dp-causal-frontier.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(old43early=out['old_original_C8']['idle_ready_no_later_count'],old_tail=tail,
        pair_end_skew=out['new_profile']['pair_end_skew_ms'],gloo_TTFT=out['new_profile']['gloo_only_recorded_TTFT_favourable'],
        startup=out['global_empty_joint_scope'],largest_suffix=max(r['duration_ms'] for r in suffix)),ensure_ascii=False))


if __name__=='__main__':main()
