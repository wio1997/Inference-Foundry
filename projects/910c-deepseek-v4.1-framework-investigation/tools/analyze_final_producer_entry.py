#!/usr/bin/env python3
"""Actual last-layer role/demand entry and same-C8 complete-parent sensitivity."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'))
from price_vocab_selection_fusion import records,cached,BODY,RECEIPT

def read(p):return json.loads(Path(p).read_bytes())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def latest_dp_change(end,start,offset):
    n=len(end);assert n==8 and len(start)==n and len(offset)==n and all(len(r)==n for r in offset)
    upper=max(min(end[i]-start[j]+offset[i][j] for j in range(n)) for i in range(n))
    lower=max(min(end[i]-start[j]-offset[j][i] for j in range(n)) for i in range(n))
    local=[a-b for a,b in zip(end,start)]
    upper=min(upper,max(local));lower=max(lower,min(local));assert lower<=upper
    return [lower/1e6,upper/1e6]
def main():
    identity=read(OUT/'metadata-readers-execution-identity.json')
    model=next(s for s in identity['sources'] if s['runtime_path'].endswith('deepseek_v41/model.py'))
    assert digest(model['local_path'])==model['runtime_sha256']
    plan_path=Path('audit/code-led/target-layer-plan.json');plan=read(plan_path)
    assert plan['source']['sha256']==model['runtime_sha256']
    config_path=Path(plan['config']['path']);assert digest(config_path)==plan['config']['sha256']
    config=read(config_path)['text_config'];role=plan['layers'][39]
    assert not role['is_kv_source'] and not role['is_index_source'] and role['kv_source_layer']==20 and role['index_source_layer']==36
    assert plan['aux_capture_contract']['zero_based_entry_layers']==[36,37,38] and not role['aux_before_layer']
    waves,hist,_,_,_=records();profiles=cached(RECEIPT);bodies=cached(BODY)['profiles']
    offsets=hist['clock_contract']['offset_upper_ns'];windows=[];by={}
    for wave,rs in waves.items():
        if rs[0]['kind']!='E':continue
        for rank,r in rs.items():
            if not r.get('scheduled'):continue
            b=bodies[rank]['bodies'][r['ordinal']];assert len(b['HcPre_TASK_ROWS'])==80
            start_id=b['HcPre_TASK_ROWS'][78];mapped={v[0]:v for v in b['rows']};start=mapped[start_id][1]
            start38_id=b['HcPre_TASK_ROWS'][76];start38=mapped[start38_id][1];assert start38<start
            target,=[c for c in profiles[rank]['calls'] if c[0]=='target' and c[3][-1][2]==r['calls']['target']['end']]
            head=target[3][0][1];assert start<b['end_ns']<=head
            rec=dict(wave=wave,rank=rank,ordinal=r['ordinal'],scheduled=r['scheduled'],
                first_HcPre_shape=b['HcPre_shape'],last_Attn_HcPre_task=start_id,last_Attn_HcPre_start_ns=start,
                eager_last_HcPost_task=b['last_HcPost_TASK_ROW'],last_HcPost_end_ns=b['end_ns'],
                head_matmul_task=target[3][0][0],head_matmul_start_ns=head,
                original_parent_ms=(head-start)/1e6,head_rows=target[2])
            rec.update(layer38_Attn_HcPre_task=start38_id,layer38_Attn_HcPre_start_ns=start38,
                       two_layer_parent_ms=(head-start38)/1e6)
            windows.append(rec);by[wave,rank]=rec
    assert len(windows)==56 # DP0 four eager bodies, DP1 three, TP8 each.
    dp_prices={};dp_prices38={}
    for wave in sorted({w['wave'] for w in windows}):
        for dp in (0,1):
            ix=list(range(dp*8,dp*8+8))
            if not all((wave,i) in by for i in ix):continue
            group=[by[wave,i] for i in ix]
            clocks=[[offsets[i][j] for j in ix] for i in ix]
            bounds=latest_dp_change([r['head_matmul_start_ns'] for r in group],
                                 [r['last_Attn_HcPre_start_ns'] for r in group],clocks)
            dp_prices[wave,dp]=list(bounds)
            dp_prices38[wave,dp]=latest_dp_change([r['head_matmul_start_ns'] for r in group],
                                                 [r['layer38_Attn_HcPre_start_ns'] for r in group],clocks)
    requests=[]
    for r in hist['requests']:
        ordinal,=[i for i,s in enumerate(hist['sections']) if s['M']==r['first_eager_M']]
        wave,=sorted({w['wave'] for w in windows if w['ordinal']==ordinal and w['rank']//8==r['DP']})
        bounds=dp_prices[wave,r['DP']]
        requests.append(dict(slot=r['slot'],DP=r['DP'],wave=wave,TTFT_ms=r['TTFT_ms'],
            all_free_parent_saved_ms=bounds,conditional_TTFT_percent=[100*x/r['TTFT_ms'] for x in bounds],
            two_layer_first_output_saved_ms=dp_prices38[wave,r['DP']]))
    requests.sort(key=lambda r:r['slot']);assert [r['slot'] for r in requests]==list(range(8))
    base=max(r['TTFT_ms'] for r in requests)
    p90=[base-max(r['TTFT_ms']-r['all_free_parent_saved_ms'][b] for r in requests) for b in (0,1)]
    # A separate favorable prefix relaxation covers the broader prefill producer.
    # Do not prune that scope using only a single first-output contribution.
    wave_upper={wave:max(v[1] for (w,dp),v in dp_prices.items() if w==wave) for wave in sorted({w for w,dp in dp_prices})}
    cumulative=[]
    for r in requests:
        saved=sum(v for w,v in wave_upper.items() if w<=r['wave'])
        cumulative.append(dict(slot=r['slot'],inclusive_eager_waves=[w for w in wave_upper if w<=r['wave']],
            optimistic_saved_ms=saved,optimistic_TTFT_percent=100*saved/r['TTFT_ms']))
    cumulative_p90=base-max(r['TTFT_ms']-c['optimistic_saved_ms'] for r,c in zip(requests,cumulative))
    wave_upper38={wave:max(v[1] for (w,dp),v in dp_prices38.items() if w==wave) for wave in wave_upper}
    cumulative38=[dict(slot=r['slot'],optimistic_saved_ms=sum(v for w,v in wave_upper38.items() if w<=r['wave'])) for r in requests]
    cumulative38_p90=base-max(r['TTFT_ms']-c['optimistic_saved_ms'] for r,c in zip(requests,cumulative38))
    assert cumulative38_p90>=cumulative_p90
    net38=[]
    for rho in (0,.25,.5,.75):
        for h in (.1,.5,1):
            gains=[rho*c['optimistic_saved_ms']-h*len(r0['inclusive_eager_waves']) for c,r0 in zip(cumulative38,cumulative)]
            gain=base-max(r['TTFT_ms']-d for r,d in zip(requests,gains))
            net38.append(dict(removable_exposed_fraction=rho,added_cost_ms_per_eager_wave=h,
                P90_conditional_net_saved_ms=gain,P90_conditional_net_percent=100*gain/base,
                per_request_signed_saved_ms=gains))
    result=dict(status='PARTIAL_LAST_LAYER_PRODUCER_ENTRY_AND_WHOLE_PARENT_PRICE',whole_chain_complete=False,
        precision='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',source_identity='metadata-readers-execution-identity.json',
        corrected_topology=dict(path=str(plan_path),sha256=digest(plan_path),layer39=role,
            sliding_window=config['sliding_window'],index_topk=config['index_topk'],aux=plan['aux_capture_contract']),
        live_outputs=[
            'current layer39 SWA128 cache writes for present/future requests; no own long-KV or index-K source writes',
            'reads shared long-KV source20 and topK source36; selected query row must map to original CP-local topK row, not arbitrary prefix',
            'query output restores TP heads and FlashComm suffix before O projection/RS; preserve all required collective participants',
            'selected normalized head rows when no prompt-logprob/pooling/raw observer; full prompt-logprob keeps prompt rows',
            'mHC rounded residual, coefficients, FFN router/expert/combined/post and collapse retain each original primitive bit contract'],
        boundaries=dict(source='model.py:879-920,1505-1608; dsa_v41.py:350-389; dsa_v41_cp.py:189-225',
            separation='Actual preprocess already separates query_hidden from cache_hidden. Last39 has no DSpark aux reader. Its own SWA write demand still has to be traced to physical slots/pages and future read horizon.',
            caution='SWA128 does not alone prove only128 rows may be kept: physical block ownership, chunk overlap, same-step selected queries, prefix sharing and future iterations must be checked.',
            precision='Naively shrinking M is not admitted. Prior compact FFN changed coefficients/router or combined/HCPost bits despite equal final readout; that exact recipe remains stopped. Original tiler/rounding or a new fully bit-validated design is required.'),
        backward_demand_extension=dict(last_aux_capture='zero-based layer38 entry, before its body',
            after_last_aux_body_layers=[38,39],layer38_role=plan['layers'][38],
            isolated_causal_prefill_example=dict(guard='one terminal selected query/request, contiguous causal positions, no prompt-logprobs or additional observers; physical slot/eviction correctness still unproved',
                layer39_query_rows=1,layer39_SWA_KV_tail_rows=128,layer38_output_query_tail_rows=128,layer38_SWA_KV_tail_rows=255),
            keep='both own SWA future state, shared longKV20 and CP-local topK36 row mapping, exact HC/quant/router/expert primitives; aux entry38 remains full'),
        C8_first_eager_price=dict(windows=windows,requests=requests,P90_all_free_saved_ms=p90,
            P90_all_free_percent=[100*x/base for x in p90],
            scope='Entire last39 Attention-HcPre through both Attention/FFN/Post and pre-head readout is free, including original waits/gaps; retain head GEMM, verification, DSpark and all other work. One first-eager contribution per request, eight-rank DP frontier under saved clocks.',
            qualification='An optimistic original-schedule component envelope, not physically realizable deletion, universal ceiling, net gain or independent stages that can be added to prior terminal budgets. Required SWA work and replacement cost are not yet subtracted.'),
        broader_C8_prefill_prefix_relaxation=dict(wave_max_independent_DP_free_ms={str(w):v for w,v in wave_upper.items()},
            requests=cumulative,P90_optimistic_saved_ms=cumulative_p90,P90_optimistic_percent=100*cumulative_p90/base,
            qualification='Credit the largest independently favorable DP parent frontier in each real eager wave, then serially propagate all inclusive eager waves to each first output. No graph/decode, head GEMM or later-wave credit. This relaxes actual critical-rank/DP/async exposure and all required SWA work; not an expected gain. The single-first-output model above is narrower and cannot prune this broader producer.'),
        two_layer_C8_prefill_prefix_relaxation=dict(wave_max_independent_DP_free_ms={str(w):v for w,v in wave_upper38.items()},
            requests=cumulative38,P90_optimistic_saved_ms=cumulative38_p90,P90_optimistic_percent=100*cumulative38_p90/base,
            qualification='Same generous inclusive eager-wave relaxation, now from layer38 Attention HcPre after the final aux capture to pre-head readout. All required SWA work, surviving rows and replacement costs must be subtracted; not measured gain or a universal ceiling.'),
        two_layer_paid_scenarios=dict(scenarios=net38,
            formula='saved_i = rho * all-free inclusive two-layer prefix_i - h * affected eager waves_i; recompute maximum of8 original TTFTs',
            qualification='rho is unknown actual removability/exposure after preserving all required cache/math/communication; h is unknown additional nonoverlapped cost per eager wave. Conditional sensitivity only, not measured speed. Negative gains are retained. Precision failures remain disqualifying regardless of these numbers.'),
        candidates=[dict(name='last39 query/output/FFN demand-driven joint producer',status='待确认，worth deeper analysis',
                         gaps=['physical SWA write/read domains and cache lifetime','selected global/SP/CP row and topK36 mapping','all primitive bit-preserving numerical geometry','typed removable stage and paid implementation cost; six ordinary load extrapolation'])],
        next_action='Trace layer38/39 SWA128 physical writer/page/slot and backward-selected query/topK36 row domains after the last aux entry38; separate required cache production from removable query/O/FFN work in the matching two-layer parent. Do not retry failed compact FFN or prune with a narrow single-output/terminal bound.',
        inputs=[dict(path=str(p),sha256=digest(p)) for p in [BODY,RECEIPT,plan_path,Path(__file__)]],
        new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[])
    out=OUT/'final-producer-entry.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(windows=len(windows),P90_first_only_all_free_ms=p90,P90_first_only_percent=result['C8_first_eager_price']['P90_all_free_percent'],
        P90_broader_prefill_optimistic_ms=cumulative_p90,P90_broader_prefill_optimistic_percent=100*cumulative_p90/base,
        P90_two_layer_prefill_optimistic_ms=cumulative38_p90,P90_two_layer_prefill_optimistic_percent=100*cumulative38_p90/base,
        layer39_KV_source=False,layer39_index_source=False,SWA=config['sliding_window'],aux_zero_based=plan['aux_capture_contract']['zero_based_entry_layers'],new_NPU_work=0)))

if __name__=='__main__':main()
