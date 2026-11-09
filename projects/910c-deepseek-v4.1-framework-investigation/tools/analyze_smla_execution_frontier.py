#!/usr/bin/env python3
"""Bind real SMLA calls and price explicit conditional transfer scenarios."""
import collections
import hashlib
import json
import statistics
from pathlib import Path
import zlib
from analyze_attention_output_delivery import extract
from analyze_c2_producer_windows import shape,stats

BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')
RATE=2405.3037563488215  # Archived two-way charged-byte reference GB/s, not sparse Cube bandwidth.


def main():
    data=extract(BASE)
    cfg_path=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/evidence/model-config.json')
    cfg=json.loads(cfg_path.read_text());cfg=cfg.get('text_config',cfg)
    roles=[]
    for i,r in enumerate(cfg['compress_ratios'][:40]):
        latest=lambda key:max((k for k in cfg[key] if k<=i),default=None) if r else None
        roles.append(dict(layer=i,ratio=r,kv_source=latest('kv_source_layer_ids'),index_source=latest('index_source_layer_ids')))
    ws=[];groups=collections.defaultdict(list)
    for w in data['windows']:
        role=roles[w['layer']]
        rec={k:w[k] for k in ['dp','tp','carrier','layer','branch']};rec.update(role)
        if w['smla']:
            s=w['smla'];reg=s['regime'];ss=shape(reg[2]);dd=reg[3].split(';')
            assert ss[0][1:]==(64,512) and dd[0]=='DT_BF16' and not ss[3]  # C8 is causal, no DSpark ori-sparse.
            assert bool(ss[2])==bool(role['ratio'])
            if role['ratio']:
                assert ss[4][1:]==(1,512) and ss[2][1:] in [(64,1,512),(128,1,512)]
                assert ss[2][1]==128//role['ratio']
            rec.update(task=s['task'],us=s['us'],query_rows=ss[0][0],input_shapes=reg[2],input_dtypes=reg[3],signature=reg)
            key=(w['dp'],w['carrier'],*reg)
            groups[key].append(s['us'])
        else:rec.update(query_rows=0,us=0.)
        ws.append(rec)
    units=json.loads((BASE/'full-review/PROFILE-UNIT-COSTS.json').read_text())
    matched=0
    for key,vals in groups.items():
        dp,carrier,*reg=key
        rows=[r for r in units['signatures'] if r['phase']=='eager_first_token' and
              [r['name'],r['opType'],r['inputShapes'],r['inputDataTypes'],r['outputShapes'],r['outputDataTypes']]==reg]
        assert len(rows)==1
        cs=[c for c in rows[0]['conditions'] if (c['dp'],c['carrier'])==(dp,carrier)];assert len(cs)==1
        c=cs[0];assert c['calls']==len(vals)
        for name,value in [('min',min(vals)),('median',statistics.median(vals)),('max',max(vals))]:assert abs(c['observed_us'][name]-value)<1e-8
        matched+=1
    def delta(w,kind,K=256,r=1.):
        if kind=='all_SMLA_free':return w['us']
        if not w['query_rows'] or not w['ratio']:return 0.
        if kind=='direct_and_first_partial':return min(w['us'],delta(w,'direct',K,r)+delta(w,'first_partial_only',K,r))
        if kind=='shared_and_first_partial':return min(w['us'],delta(w,'shared_panel_later',K,r)+delta(w,'first_partial_only',K,r))
        rowbytes=K*512*2
        if kind=='direct':saved=(4-2/r)*rowbytes
        elif kind=='shared_panel_later':saved=(2*rowbytes if w['layer']!=w['index_source'] else 0.)
        elif kind=='first_partial_only':saved=2*64*512*4  # one nonfinal FP32 read+write, later read retained.
        else:raise AssertionError(kind)
        # 50% exposure is an archived sensitivity premise, not a measured phase fraction.
        return min(w['us'],.5*saved*w['query_rows']/(RATE*1000))
    def price(kind,K=256,r=1.):
        carriers={m:0. for m in data['carrier_order']}
        for sec in data['arrival_sections']:
            for l in range(40):
                credits=[]
                for dp in [0,1]:
                    selected=[w for w in ws if (w['dp'],w['carrier'],w['layer'])==(dp,sec['M'],l)];assert len(selected)==8
                    credits.append(max(delta(w,kind,K,r) for w in selected))
                lo,hi=sec['layers'][l]['DP0_minus_DP1_us']
                gain=lambda d:max(d,0)-max(d-credits[0],-credits[1])
                carriers[sec['M']]+=max(gain(lo),gain(hi))/1000
        reqs=[]
        for req in data['requests']:
            credit=sum(carriers[m] for m in data['carrier_order'][:data['carrier_order'].index(req['first_eager_M'])+1])
            reqs.append(dict(**req,conditional_credit_ms=credit,percent=100*credit/req['TTFT_ms']))
        return dict(carriers_ms=carriers,requests=reqs,max_percent=max(r['percent'] for r in reqs))
    values={'all_SMLA_free':price('all_SMLA_free'),'first_partial_reference':price('first_partial_only')}
    for k in [128,256,512]:
        for r in [1.,.75,.5]:values[f'direct/K{k}/sparse_rate{r}']=price('direct',k,r)
        values[f'shared_later/K{k}/no_relocation_cost']=price('shared_panel_later',k)
        for r in [1.,.75]:values[f'direct_and_first_partial/K{k}/sparse_rate{r}']=price('direct_and_first_partial',k,r)
        values[f'shared_and_first_partial/K{k}/no_relocation_cost']=price('shared_and_first_partial',k)
    thresholds={}
    for kind in ['direct','direct_and_first_partial']:
        thresholds[kind]={}
        for k in [128,256,512]:
            lo,hi=.5,8.
            for _ in range(28):
                mid=(lo+hi)/2
                if price(kind,k,mid)['max_percent']>=1:hi=mid
                else:lo=mid
            thresholds[kind][k]=hi if price(kind,k,hi)['max_percent']>=1 else None
    sizes={}
    for dp,carrier in sorted({(w['dp'],w['carrier']) for w in ws}):
        q=max(w['query_rows'] for w in ws if (w['dp'],w['carrier'])==(dp,carrier))
        sizes[f'DP{dp}/carrier{carrier}']=dict(max_local_queries=q,full_K512_panel_MiB=q*512*1024/1024**2,address8B_plan_MiB=q*512*8/1024**2)
    pins=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [cfg_path,BASE/'full-review/PROFILE-UNIT-COSTS.json']]
    out=dict(status='SOURCE_DERIVED_SMLA_BRANCHES_AND_CONDITIONAL_TRANSFER_SENSITIVITY',counts=dict(windows=len(ws),native_calls=sum(w['query_rows']>0 for w in ws),no_query=sum(w['query_rows']==0 for w in ws),calls_by_ratio=dict(collections.Counter(w['ratio'] for w in ws if w['query_rows'])),independent_cost_groups_matched=matched),roles=roles,groups=[dict(dp=k[0],carrier=k[1],signature=list(k[2:]),stats=stats(v)) for k,v in groups.items()],conditional_values=values,direct_equal_sparse_rate_threshold_for_max_request_one_percent=thresholds,panel_and_address_sizes=sizes,rate_reference_GB_s_total_charged_bytes=RATE,source_pins=data['source_pins']+pins,limits=['C8 diagnostic observer cohort, not production timing; never reconstruct baseline by subtracting observer','Scenario uniform effective selected K128/256/512, sparse AIC QK/PV rates relative to old two-way charged-byte calibration,50% exposure; none is measured sparse bandwidth or phase time','Per-DP max removable duration grants all eight ranks that advance; original EP arrival deltas/schedule held fixed; request credit is optimistic conditional proxy','All_SMLA_free is a scoped original-schedule budget, not a universal bound on changed scheduling','shared_panel_later assumes first-consumer relocation free; actual placement/copy/retirement/cache effects and storage must be charged','first_partial_reference requires a first nonfinal tile and original FP32 store/rounding/sink, later partial read unchanged; not full PV/score deletion','No candidate NPU or model request run'],new_NPU_measurements=0,whole_chain_complete=False)
    (OUT/'smla-execution-parent-frontier.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    raw=json.dumps(ws,ensure_ascii=False,separators=(',',':')).encode();(OUT/'smla-execution-windows.json.zlib').write_bytes(zlib.compress(raw,9))
    print(json.dumps(dict(counts=out['counts'],values={k:round(v['max_percent'],6) for k,v in values.items()},thresholds=thresholds,sizes=sizes),ensure_ascii=False))


if __name__=='__main__':main()
