#!/usr/bin/env python3
"""Source-qualified indexer hot handoff geometry and conditional parent price.

Read existing frozen native task tables; do not launch inference/profiling.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import zlib

from analyze_c2_producer_windows import shape
from analyze_attention_output_delivery import extract

BASE = Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT = Path('audit/execution-search')


def pin(p):
    return {'path': str(p), 'sha256': sha256(p.read_bytes()).hexdigest()}


def main():
    data = extract(BASE)
    p = BASE/'.cache/first-token-v168-protocol-bodies.json.zlib'
    native = json.loads(zlib.decompress(p.read_bytes()))
    cfgp = OUT/'startup-scope-evidence.json'
    cfg = json.loads(cfgp.read_text())['model_config']['raw_config']['text_config']
    sources = cfg['index_source_layer_ids']
    assert sources == [2,8,14,20,24,28,32,36]
    units_path = BASE/'full-review/PROFILE-UNIT-COSTS.json'
    units = json.loads(units_path.read_text())['signatures']
    windows = []
    signatures = defaultdict(list)
    for prof in native['profiles']:
        regs = prof['regimes']
        for body in prof['bodies']:
            by = {r[0]:r for r in body['rows']}
            pre = [by[i] for i in body['HcPre_TASK_ROWS']]
            stream, = {r[3] for r in pre}
            post = sorted((r for r in body['rows'] if r[3]==stream and regs[r[-1]][1]=='HcPost'), key=lambda r:r[1])
            assert len(pre)==len(post)==80
            for l in range(40):
                a,b = pre[2*l],post[2*l]
                selected = [r for r in body['rows'] if a[1]<=r[1] and r[2]<=b[1]]
                def pick(n):
                    return [r for r in selected if regs[r[-1]][1] == n]
                qli = pick('VllmAscendQuantLightningIndexerV2')
                quant = pick('_quantize_indexer_query_kernel')
                sort = pick('_prepare_indexer_indices_kernel')
                if not qli:
                    assert not quant and not sort, (prof['DP'],prof['TP'],l)
                    continue
                assert l in sources and len(qli)==len(quant)==len(sort)==1
                q,k,s = quant[0],qli[0],sort[0]
                qs,ks,ss = [shape(regs[r[-1]][2]) for r in [q,k,s]]
                rows=qs[0][0]
                assert qs[0] == (rows,32,128) and ks[0] == (rows,32,128)
                assert ss[0] == (rows,512) and ss[1] == (rows,) and ss[2] == (rows,1)
                assert q[2]<=k[1] and k[2]<=s[1]  # native producer order, not other-stream whole-parent retirement
                assert rows<=1024
                row=dict(dp=prof['DP'],tp=prof['TP'],carrier=body['F21_carrier_rows'],layer=l,
                         rows=rows,ratio=cfg['compress_ratios'][l],
                         q_us=(q[2]-q[1])/1000,qli_us=(k[2]-k[1])/1000,sort_us=(s[2]-s[1])/1000,
                         task_ids=[q[0],k[0],s[0]],streams=[q[3],k[3],s[3]])
                windows.append(row)
                for r in [q,k,s]:
                    key=(prof['DP'],body['F21_carrier_rows'],*regs[r[-1]])
                    signatures[key].append((r[2]-r[1])/1000)
    # Native rows must match independent signature-cost receipts, not just names.
    for key, vals in signatures.items():
        dp,m,*reg=key
        candidates=[u for u in units if u['phase']=='eager_first_token' and
                    [u[k] for k in ['name','opType','inputShapes','inputDataTypes','outputShapes','outputDataTypes']]==reg]
        assert len(candidates)==1, key
        cond=[c for c in candidates[0]['conditions'] if (c['dp'],c['carrier'])==(dp,m)]
        assert len(cond)==1 and cond[0]['calls']==len(vals),key
        for n,v in [('min',min(vals)),('median',median(vals)),('max',max(vals))]:
            assert abs(cond[0]['observed_us'][n]-v)<1e-7
    bylayer={(w['dp'],w['tp'],w['carrier'],w['layer']):w for w in windows}
    def value(w,kind,rho,h):
        if w is None:return 0.
        old=w['q_us']+w['sort_us']+(w['qli_us'] if kind=='whole_qli_plus_handoff' else 0.)
        # Favorable sum over quant (aux stream) and main-stream tasks; not all exposed.
        return max(0.,rho*old-h)
    def price(kind,rho,h):
        carriers={m:0. for m in data['carrier_order']}
        for sec in data['arrival_sections']:
            for l in sources:
                credits=[max(value(bylayer.get((dp,tp,sec['M'],l)),kind,rho,h) for tp in range(8)) for dp in [0,1]]
                lo,hi=sec['layers'][l]['DP0_minus_DP1_us']
                def gain(d):return max(d,0)-max(d-credits[0],-credits[1])
                carriers[sec['M']]+=max(gain(lo),gain(hi))/1000
        req=[]
        for r in data['requests']:
            credit=sum(carriers[m] for m in data['carrier_order'][:data['carrier_order'].index(r['first_eager_M'])+1])
            req.append(dict(**r,conditional_credit_ms=credit,percent=100*credit/r['TTFT_ms']))
        # Fixed original collector uses sorted[floor(.9*N)]; N8 => maximum.
        base=max(r['TTFT_ms'] for r in req)
        gain=base-max(r['TTFT_ms']-r['conditional_credit_ms'] for r in req)
        return {'scope':'Favorable same-cohort per-layer DP arrival envelope; quant and main-stream durations summed optimistically. No causal/request speedup prediction.',
                'rho':rho,'new_cost_us_per_source':h,'carriers_ms':carriers,'requests':req,
                'max_request_percent':max(r['percent'] for r in req),
                'frozen_N8_P90_baseline_ms':base,'conditional_P90_saved_ms':gain,'conditional_P90_percent':100*gain/base}
    prices={}
    for kind in ['whole_qli_plus_handoff','quant_and_canonical_only']:
        for rho,h in [(1.,0.),(.5,5.),(.5,10.)]:
            prices[f'{kind}/rho{rho}/h{h}']=price(kind,rho,h)
    geometry=[]
    for dp,m in sorted({(r['dp'],r['carrier']) for r in windows}):
        ws=[w for w in windows if (w['dp'],w['carrier'])==(dp,m)]
        geometry.append({'dp':dp,'carrier':m,'source_calls':len(ws),'per_rank_rows':sorted(set(w['rows'] for w in ws)),
                         'fused_lengths_in_all_observed_calls':all(w['rows']<=1024 for w in ws)})
    out={'status':'BOUNDED_NATIVE_INDEXER_GEOMETRY_AND_CONDITIONAL_PARENT_VALUE',
         'whole_chain_complete':False,'each_replacement_bitwise':True,
         'source_pins':data['source_pins']+[pin(units_path),pin(cfgp)],
         'native_window_count':len(windows),'independent_cost_conditions_matched':len(signatures),
         'source_layers':sources,'geometry':geometry,'conditional_values':prices,
         'windows':windows,
         'findings':[
             'Actual eager query rows are per-rank CP-owned, not global carrier. All recorded topk512 rows<=1024 and output shapes include lengths; no separate large-row sum saving in these calls.',
             'Chronological canonical bank is already shared at eight index sources across later layers. Do not charge its production at all38 long-attention consumers or recompute sort inside each consumer.',
             'Quantizer stores FP16-rounded scale and uses div_rn/nearbyint; every new fused primitive must preserve exact rounding and published INT8/scale/indices/length bytes.',
             'Native identity/type name V2 in saved tasks corresponds to current Python V3 dispatch family, not a proof of loaded custom library identity or native internal equivalence. Resolve exact wrapper/binary/source before native replacement.',
             'All-free handoff child bound is not a bound on large QLI scoring/membership/consumer/layout schemes. Prior byte-membership overlay and batched-slot C8 cohort remain separate baselines/precision scopes.',
         ],
         'next_action':'Use whole parent value to choose native query/QLI/ordered-output boundary; bind exact source/loaded-library family and candidate mode1/2/3 contracts, preserving existing aux overlap and shared output generations. No new profile until a decision-changing gap is identified.',
         'new_NPU_work':0,'new_profile_collection':0,'service_changes':[]}
    (OUT/'indexer-delivery-entry.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'windows':len(windows),'geometry':geometry,'conditions_matched':len(signatures),
                      'max_request_percent':{k:v['max_request_percent'] for k,v in prices.items()},'whole_chain_complete':False}))


if __name__=='__main__':main()
