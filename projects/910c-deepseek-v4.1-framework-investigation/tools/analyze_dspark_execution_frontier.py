#!/usr/bin/env python3
"""Bind actual DSpark score primitives and full-parent archived sensitivities."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zlib

BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'))
from price_vocab_selection_fusion import records,RECEIPT,cached,ORDINARY,GUARDS
from price_attention_send_epilogue import ENDPOINTS
from price_draft_movable_arrivals import latest_change

PINS={}
def read(p):
    p=Path(p);p=p if p.is_absolute() else BASE/p
    raw=p.read_bytes();PINS[str(p)]=hashlib.sha256(raw).hexdigest()
    return json.loads(zlib.decompress(raw) if p.suffix=='.zlib' else raw)

def stats(values):
    return dict(n=len(values),min=min(values),median=statistics.median(values),max=max(values))

def union(intervals):
    end=-1;total=0
    for a,b in sorted(intervals):
        total+=max(0,b-max(a,end));end=max(end,b)
    return total


def main():
    units=read('full-review/PROFILE-UNIT-COSTS.json')
    receipt=read('full-review/runs/publication-residual-profile-cpu-plan-2979e908408a-export-result.json')
    windows=read('full-review/CPU-PLAN-RESIDUAL-ANALYSIS.json')['native_windows']
    independent_clean=read('full-review/CLEAN-DSPARK-SCORE-VALUE.json')
    groups=defaultdict(list);clean=[]
    for pin in receipt['local_cache']:
        p=Path(pin['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==pin['sha256']
        data=read(p);regs=data['regimes'];rows=data['rows'];by={r[0]:r for r in rows}
        for w in windows:
            if w['rank']!=data['rank']:continue
            lo=by[w['target_wait_TASK_ROW']][2];hi=by[w['next_launch_TASK_ROW']][1]
            period_ns=w['period_ms']*1e6
            seq=sorted([r for r in rows if lo<=r[1]<hi and r[9]>=0],key=lambda r:(r[1],r[0]))
            head,=[r for r in seq if regs[r[9]]['opType']=='MatMulV2' and regs[r[9]]['inputShapes']=='"20,5120;16160,5120"']
            stream=[r for r in seq if r[3]==head[3] and r[1]>=head[1]]
            W2=[r for r in stream if regs[r[9]]['opType']=='MatMulV2' and regs[r[9]]['inputShapes']=='"4,256;129280,256"']
            selects=[r for r in stream if regs[r[9]]['opType']=='ArgMaxV2' and regs[r[9]]['inputShapes']=='"4,129280;"']
            assert len(W2)==len(selects)==5
            adds=[]
            for j,(a,b) in enumerate(zip(W2,selects)):
                add,=[r for r in stream if a[2]<=r[1]<b[1] and regs[r[9]]['opType']=='Add']
                assert regs[a[9]]['inputDataTypes']=='DT_BF16;DT_BF16'
                assert regs[add[9]]['inputDataTypes']=='DT_BF16;DT_BF16'
                assert regs[b[9]]['inputDataTypes']=='DT_BF16;INT64'
                assert regs[a[9]]['outputDataTypes']==regs[add[9]]['outputDataTypes']=='DT_BF16'
                assert j==4 or b[2]<=W2[j+1][1]
                adds.append(add)
            cast,publish=stream[stream.index(selects[-1])+1:stream.index(selects[-1])+3]
            assert regs[cast[9]]['opType']=='Cast' and regs[publish[9]]['opType']=='ViewCopy'
            retained=sum(r[2]-r[1] for r in [head,*W2]);whole=publish[2]-head[1]
            clean.append(dict(rank=data['rank'],period_us=period_ns/1000,
                head_task=head[0],W2_tasks=[r[0] for r in W2],add_tasks=[r[0] for r in adds],select_tasks=[r[0] for r in selects],publish_task=publish[0],
                whole_score_us=whole/1000,six_GEMM_us=retained/1000,
                whole_non_GEMM_free_us=(whole-retained)/1000,
                whole_non_GEMM_free_period_percent=100*(whole-retained)/period_ns))
            for r in [head,*W2,*adds,*selects]:
                g=regs[r[9]];key=(data['DP'],tuple(g[k] for k in ['name','opType','inputShapes','inputDataTypes','outputShapes','outputDataTypes']))
                groups[key].append((r[2]-r[1])/1000)
    assert len(clean)==112
    matched=[]
    for (dp,reg),values in groups.items():
        s,=[s for s in units['signatures'] if s['phase']=='outside_target_graph' and
            [s[k] for k in ['name','opType','inputShapes','inputDataTypes','outputShapes','outputDataTypes']]==list(reg)]
        c,=[c for c in s['conditions'] if c['dp']==dp and c['carrier'] is None]
        assert c['calls']==len(values)
        for k,v in [('min',min(values)),('median',statistics.median(values)),('max',max(values))]:assert abs(c['observed_us'][k]-v)<1e-8,(dp,reg,k,c['observed_us'][k],v)
        matched.append(dict(dp=dp,signature=list(reg),calls=len(values),us=stats(values)))
    assert len(matched)==8

    waves,hist,schedule,_,_=records()
    vocab_receipt=read(RECEIPT);vocab_cache=vocab_receipt['cache'];vocab_path=Path(vocab_cache['path'])
    if not vocab_path.is_absolute():vocab_path=BASE/vocab_path
    assert hashlib.sha256(vocab_path.read_bytes()).hexdigest()==vocab_cache['sha256'];read(vocab_path)
    raw=cached(RECEIPT);call_by_end={}
    bodies=read('.cache/first-token-v168-protocol-bodies.json.zlib')
    native_db_sha={p['database']:p['sha256'] for p in bodies['profiles']}
    for p in raw:
        PINS[p['database']]=native_db_sha[p['database']]
        for role,stream,m,rows,g in p['calls']:
            if role!='draft':continue
            assert m%5==0;b=m//5;reg=p['regimes']
            W2=[r for r in rows if reg[r[3]][0]=='MatMulV2' and reg[r[3]][1]==f'"{b},256;129280,256"']
            assert len(W2)==5
            head=rows[0];assert reg[head[3]][0]=='MatMulV2'
            assert reg[head[3]][3:] == ['DT_BF16;DT_BF16','DT_BF16']
            call_by_end[p['rank'],rows[-1][2]]=dict(B=b,head_us=(head[2]-head[1])/1000,
                W2_us=sum(r[2]-r[1] for r in W2)/1000,
                full_score_us=(rows[-1][2]-head[1])/1000,
                typed_union_us=(union((r[1],r[2]) for r in rows)+g[1]-g[0])/1000,
                base_AG_us=(g[1]-g[0])/1000,first_task=head[0],last_task=rows[-1][0])
    # Preserve old clocks/order and original next-target frontiers. This is an
    # optimistic scoped search budget, not a new causal performance estimate.
    native=read(ENDPOINTS)['endpoint_ms'];offsets=hist['clock_contract']['offset_upper_ns']
    prices={};portable=[]
    for route in ['full_score_free','W2_free','base_head_free','keep_base_head_all_other_free']:
        per={metric:[[0.]*8,[0.]*8] for metric in native}
        for n,ranks in waves.items():
            if not any(w.get('scheduled') for w in ranks.values()):continue
            timing=[w['entry'] for w in ranks.values() if w.get('scheduled')]
            credits=[]
            for rank,w in sorted(ranks.items()):
                c=call_by_end[rank,w['calls']['draft']['end']]
                value={'full_score_free':c['full_score_us'],'W2_free':c['W2_us'],
                       'base_head_free':c['head_us'],'keep_base_head_all_other_free':c['full_score_us']-c['head_us']}[route]
                credits.append(value*1000 if w.get('scheduled') else 0.)
                if route=='full_score_free':portable.append(dict(wave=n,rank=rank,kind=w['kind'],ordinal=w['ordinal'],scheduled=bool(w.get('scheduled')),**c))
            if n+1 in waves:
                starts=[waves[n+1][i]['start'] for i in range(16)]
                changes=latest_change(starts,[round(s-v) for s,v in zip(starts,credits)],offsets)
            else:changes=[min(credits)/1e6,max(credits)/1e6]
            for metric in native:
                for i,client in enumerate(schedule['client']):
                    deadline=client['end_ns'] if metric=='makespan' else client['nonempty_chunk_ns'][0]
                    if min(timing)>deadline:continue
                    for j in [0,1]:per[metric][j][i]+=changes[j]
        prices[route]={metric:dict(saved_ms=[max(native[metric])-max(old-g for old,g in zip(native[metric],v)) for v in gains],
             percent=[100*(max(native[metric])-max(old-g for old,g in zip(native[metric],v)))/max(native[metric]) for v in gains],per_request_ms=gains) for metric,gains in per.items()}
        for metric,v in prices[route].items():
            v['per_request_percent']=[[100*g/old for g,old in zip(side,native[metric])] for side in v['per_request_ms']]
            v['maximum_request_percent']=max(x for side in v['per_request_percent'] for x in side)

    B1_by_kind=defaultdict(list)
    for w in portable:
        if w['scheduled'] and w['B']==1:B1_by_kind[w['kind']].append(w)
    ordinary_score=[]
    ordinary_spec=read(ORDINARY);guards=read(GUARDS)
    for req in ordinary_spec['ordinary_requests']:
        ctrl,=[r for r in guards['stages'][req['stage']]['requests'] if r['name']==req['request']]
        faster=min(ctrl['stock_elapsed_ms'],ctrl['restored_stock_elapsed_ms'])
        rows={kind:dict(calls=len(v),full_score_free_us=statistics.median(x['full_score_us'] for x in v),
             W2_free_us=statistics.median(x['W2_us'] for x in v)) for kind,v in B1_by_kind.items()}
        scenarios={}
        for field in ['full_score_free_us','W2_free_us']:
            if 'G' not in rows:continue
            # Shape-matched B1 M5 score calls; E vs G parent differs in host
            # readiness. No claim that C8 timings reproduce ordinary cache/peers.
            prompt=rows.get('E',rows['G'])[field]
            ms=(prompt+req['draft_verifications']*rows['G'][field])/1000
            scenarios[field]=dict(conditional_response_ms=ms,percent_of_faster_reference=100*ms/faster)
        ordinary_score.append(dict(request=req['request'],input_tokens=req['input_tokens'],verifications=req['draft_verifications'],
             score_case_B1_M5=rows,scenarios=scenarios))

    context=read('full-review/DSPARK-CONTEXT-FUSION-ESTIMATES.json')
    # Native context sharing and direct-write subplans preserve WKV. Whole
    # 3-layer span includes WKV; host-launch span is a different, wider parent.
    context_values={k:v for k,v in context['C8_saved_ms']['whole'].items() if k in ['share_lookup_slot_reassembly','direct_writer_100pct','entire_three_layer_span_free','host_first_to_last_launch_span_full_propagation']}
    ordinary=[]
    for r in context['ordinary_requests']:
        ordinary.append(dict(request=r['request'],input_tokens=r['input_tokens'],verification_steps=r['verification_steps'],
            scenarios={k:v for k,v in r['scenarios'].items() if k in ['share_lookup_slot_reassembly','direct_writer_100pct','entire_three_layer_span_free','host_first_to_last_launch_span_full_propagation']}))
    result=dict(status='SOURCE_DERIVED_DSPARK_SCORE_AND_CONTEXT_PARENT_PARTIAL',
        counts=dict(accepted_decode_score_periods=len(clean),typed_cost_groups_matched=len(matched),C8_waves=len(waves),C8_score_records=len(portable)),
        actual_boundary='one TP8 BF16 base-head gather; five REPLICATED BF16 W2 projections, BF16 Add then BF16 ArgMax, serial predecessor feedback',
        accepted_decode_summary={k:stats([r[k] for r in clean]) for k in ['whole_score_us','six_GEMM_us','whole_non_GEMM_free_us','whole_non_GEMM_free_period_percent']},
        independent_clean_summary=independent_clean['summary'],
        typed_cost_groups=matched,C8_conditional_scenarios=prices,six_ordinary_score_sensitivity=ordinary_score,
        context_reused_C8_ms=context_values,context_six_ordinary=ordinary,
        source_profile_pins=PINS,limits=['Source-selected entire score and context producers are distinct disjoint regions, but their separately priced request gains must not be added; changed deadlines, clock bounds and resource ownership require joint recomputation.',
            'C8 score sensitivities preserve original next-target frontiers, external order and clock certificate. Full score free includes required matrix math/transport/feedback; no realizable algorithm or expected gain claimed.',
            'Unit-cost reference is the accepted CP+slot+CPUplan cohort, not the independent observer-free clean cohort. Both have 112 periods and the same shapes/counts but different timing; only the matching accepted cohort supplies the eight exact cost checks. No clocks or local percentages are added across cohorts.',
            'Replicated W2 means no five old Markov AGs to erase. W2 sharding introduces five dependent selection rounds; direct product reduction does not prove latency gain.',
            'Ordinary score sensitivity uses shape-matched historical B1 M5 calls and verification counts, zero replacement cost. C8 peer arrivals/cache state are not observed ordinary timings, and prompt is a B1 score-call premise, not full prompt interpolation.',
            'Context entire-three-layer-span-free excludes aux/main_proj and draft decoder. Window pruning before main is a larger parent; existing small-writer budgets do not close it.',
            'Old independent reports preserve their original scopes; no new NPU work, profile export or service request.'],new_NPU_measurements=0,service_requests=0,whole_chain_complete=False)
    (OUT/'dspark-execution-frontier.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'dspark-score-windows.json').write_text(json.dumps(dict(accepted_decode=clean,C8=portable),ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(dict(counts=result['counts'],accepted_decode=result['accepted_decode_summary'],C8_percent={r:{m:v['percent'] for m,v in ms.items()} for r,ms in prices.items()}),ensure_ascii=False))


if __name__=='__main__':main()
