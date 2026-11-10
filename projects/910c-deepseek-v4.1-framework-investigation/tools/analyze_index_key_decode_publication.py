#!/usr/bin/env python3
"""Bind four cache-source deliveries in complete archived target graph periods.

Diagnostic cohort only. No ordinary/whole-C8 extrapolation or implementation run.
"""
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import zlib
from analyze_c2_producer_windows import shape,union_ns

BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')
FIELDS=['suffix_native_us','suffix_parent_us','producer_native_us','producer_parent_us','producer_non_GEMM_native_us','producer_non_GEMM_parent_us','producer_early_overlap_upper_us']

def stats(v):return dict(n=len(v),min=min(v),median=median(v),max=max(v))

def main():
    receipt=BASE/'full-review/runs/publication-residual-profile-cpu-plan-2979e908408a-export-result.json'
    analysis=BASE/'full-review/CPU-PLAN-RESIDUAL-ANALYSIS.json'
    unitfile=BASE/'full-review/PROFILE-UNIT-COSTS.json'
    pins={str(p):sha256(p.read_bytes()).hexdigest() for p in [receipt,analysis,unitfile]}
    inputs=json.loads(receipt.read_text())['local_cache']
    periods=json.loads(analysis.read_text())['native_windows']
    groups=defaultdict(list);windows=[];steps=[]
    for pin in inputs:
        p=Path(pin['path']);assert sha256(p.read_bytes()).hexdigest()==pin['sha256'];pins[str(p)]=pin['sha256']
        d=json.loads(zlib.decompress(p.read_bytes()));regs=d['regimes'];by={r[0]:r for r in d['rows']}
        for period in periods:
            if period['rank']!=d['rank']:continue
            start=by[period['target_launch_TASK_ROW']][1];end=by[period['target_wait_TASK_ROW']][2]
            rr=sorted([r for r in d['rows'] if start<=r[1] and r[2]<=end and r[-1]>=0],key=lambda r:(r[1],r[0]))
            pres=[r for r in rr if regs[r[-1]]['opType']=='HcPre'];posts=[r for r in rr if regs[r[-1]]['opType']=='HcPost']
            assert len(pres)==len(posts)==80
            stream,={r[3] for r in pres}
            step=dict(rank=d['rank'],dp=d['DP'],tp=d['TP'],ordinal=period['ordinal'],target_us=(end-start)/1000,period_us=period['period_ms']*1000)
            sw=[]
            for layer,ratio in [(2,2),(8,2),(14,2),(20,1)]:
                a,z=pres[2*layer],posts[2*layer]
                rs=[r for r in rr if a[1]<=r[1] and r[2]<=z[1]]
                def pick(op,pred=lambda g:True):return [r for r in rs if (regs[r[-1]]['opType'].startswith('MatMul') if op=='MATMUL' else regs[r[-1]]['opType']==op) and pred(regs[r[-1]])]
                wk,=pick('MATMUL',lambda g:shape(g['inputShapes'])[1:]==[(128,512)] and g['inputDataTypes']=='DT_BF16;DT_BF16')
                m=shape(regs[wk[-1]]['inputShapes'])[0][0];assert m==24
                kn,=pick('RmsNorm',lambda g:shape(g['inputShapes'])[0]==(m,128))
                kr,=pick('InplacePartialRotaryMul',lambda g:shape(g['inputShapes'])[0]==(m,1,1,128))
                dq,=pick('DynamicQuant',lambda g:shape(g['inputShapes'])[0]==(m,128))
                ki,=pick('ScatterNdUpdateSk',lambda g:g['inputDataTypes']=='INT8;INT32;INT8' and shape(g['inputShapes'])[-1]==(m,128))
                sc,=pick('Cast',lambda g:g['inputDataTypes']=='FLOAT' and shape(g['inputShapes'])[0]==(m,1))
                ks,=pick('ScatterNdUpdateSk',lambda g:g['inputDataTypes']=='FLOAT16;INT32;FLOAT16' and shape(g['inputShapes'])[-1]==(m,1))
                long,=[r for r in pick('ScatterNdUpdateSk',lambda g:g['inputDataTypes']=='DT_BF16;INT32;DT_BF16' and shape(g['inputShapes'])[-1]==(m,512)) if r[1]>=ks[2]]
                lr,=[r for r in pick('InplacePartialRotaryMul',lambda g:shape(g['inputShapes'])[0]==(m,1,1,512)) if ks[2]<=r[1]<long[1]]
                ln=max([r for r in pick('RmsNorm',lambda g:shape(g['inputShapes'])[0]==(m,512)) if r[2]<=wk[1]],key=lambda r:r[1])
                suffix=[ln,wk,kn,kr,dq,ki,sc,ks,lr,long]
                assert all(r[3]==stream for r in suffix) and all(x[2]<=y[1] for x,y in zip(suffix,suffix[1:]))
                if ratio==2:
                    wv=pick('MATMUL',lambda g:g['inputDataTypes']=='FLOAT;FLOAT' and shape(g['inputShapes'])==[(m,5120),(512,5120)]);assert len(wv)==2
                    cast,=[r for r in pick('Cast',lambda g:g['inputDataTypes']=='DT_BF16' and shape(g['inputShapes'])[0]==(m,5120) and g['outputDataTypes']=='FLOAT') if r[2]<=min(x[1] for x in wv)]
                    pool,=pick('_pool_kernel');ring,=pick('_cache_update_kernel')
                    zeros=[r for r in pick('ZerosLike',lambda g:shape(g['inputShapes'])[0]==(m,512)) if max(x[2] for x in wv)<=r[1]<pool[1]];assert len(zeros)==1
                    extra=[cast,*wv,*zeros,pool,ring];root=cast
                else:
                    root=max([r for r in pick('MATMUL',lambda g:g['inputDataTypes']=='DT_BF16;DT_BF16' and shape(g['inputShapes'])==[(m,5120),(512,5120)]) if r[2]<=ln[1]],key=lambda r:r[1]);extra=[root]
                producer=extra+suffix
                gemms=[r for r in producer if regs[r[-1]]['opType'].startswith('MatMul')]
                ng=[r for r in producer if not regs[r[-1]]['opType'].startswith('MatMul')]
                ag,=pick('hcom_allGather_');assert ag[2]<=root[1]
                qli,=pick('VllmAscendQuantLightningIndexerV2');qrows=shape(regs[qli[-1]]['inputShapes'])[0][0];assert qrows==3
                gemmus=union_ns((r[1],r[2]) for r in gemms)/1000
                w=dict(rank=d['rank'],dp=d['DP'],tp=d['TP'],ordinal=period['ordinal'],layer=layer,ratio=ratio,cache_carrier_rows=m,query_carrier_rows=qrows,
                    suffix_native_us=union_ns((r[1],r[2]) for r in suffix)/1000,suffix_parent_us=(long[2]-ln[1])/1000,
                    producer_native_us=union_ns((r[1],r[2]) for r in producer)/1000,producer_parent_us=(long[2]-root[1])/1000,
                    producer_GEMM_native_us=gemmus,producer_non_GEMM_native_us=union_ns((r[1],r[2]) for r in ng)/1000,
                    producer_non_GEMM_parent_us=(long[2]-root[1])/1000-gemmus,
                    producer_early_overlap_upper_us=min((long[2]-root[1])/1000,(root[1]-ag[2])/1000),
                    input_ready_parent_us=(long[2]-ag[2])/1000,
                    suffix_tasks=[r[0] for r in suffix],producer_tasks=[r[0] for r in producer],input_gather_task=ag[0])
                windows.append(w);sw.append(w)
                for r in producer:
                    g=regs[r[-1]];groups[(d['DP'],tuple(g[k] for k in ['name','opType','inputShapes','inputDataTypes','outputShapes','outputDataTypes']))].append((r[2]-r[1])/1000)
            for f in FIELDS:
                step[f]=sum(w[f] for w in sw)
                step[f+'_period_percent']=100*step[f]/step['period_us']
                step[f+'_target_percent']=100*step[f]/step['target_us']
            steps.append(step)
    assert len(steps)==112 and len(windows)==448
    units=json.loads(unitfile.read_text())['signatures'];matched=[]
    for (dp,reg),values in groups.items():
        s,=[u for u in units if u['phase']=='target_graph' and [u[k] for k in ['name','opType','inputShapes','inputDataTypes','outputShapes','outputDataTypes']]==list(reg)]
        c,=[c for c in s['conditions'] if c['dp']==dp]
        assert c['calls']>=len(values) and min(values)>=c['observed_us']['min']-1e-7 and max(values)<=c['observed_us']['max']+1e-7
        matched.append(dict(dp=dp,signature=list(reg),subset_calls=len(values),whole_signature_calls=c['calls'],us=stats(values)))
    sensitivity=[]
    for field in FIELDS:
        for rho,h in [(1.,0.),(.5,5.),(.5,20.)]:
            rows=[]
            for step in steps:
                ws=[w for w in windows if (w['rank'],w['ordinal'])==(step['rank'],step['ordinal'])]
                v=sum(max(0,rho*w[field]-h) for w in ws)
                rows.append(100*v/step['period_us'])
            sensitivity.append(dict(field=field,rho=rho,added_us_per_source=h,conditional_rank_period_percent=stats(rows)))
    result=dict(status='SAMPLED_C8_FULL_TARGET_GRAPH_SOURCE_DELIVERY_BOUND_NOT_REQUEST_GAIN',each_replacement_bitwise=True,whole_chain_complete=False,
        cohort='Existing metadata CPU-plan diagnostic candidate cohort,16ranks x7 complete native periods; target graph bank24/local3. No clean timing-pair comparison.',
        windows=windows,steps=steps,signature_subset_conditions=matched,
        summary={f:dict(us=stats([s[f] for s in steps]),period_percent=stats([s[f+'_period_percent'] for s in steps]),target_percent=stats([s[f+'_target_percent'] for s in steps])) for f in FIELDS},
        conditional_values=sensitivity,source_pins=[dict(path=k,sha256=v) for k,v in pins.items()],
        limits=['Captured carrier24/local3 do not attest per-call active/complete rows or slot values.','Rank-local interval fractions are not whole-wave/request TTFT or makespan savings; no cross-rank sum.','Parent spans include retained work and gaps; zeroing them overcredits a scope.','Full-row owner sharing adds communication; row splitting changes original GEMM geometry.','Ordinary loads and full227-wave original C8 still need separately matched native cache delivery accounting.','RMS/dynamicquant/scatter native internals remain unreviewed; no replacement or bitwise admission.'],
        new_NPU_work=0,new_profile_collection=0,service_requests=0,model_loads=0,remote_writes=[])
    (OUT/'index-key-decode-publication-profile.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(steps=len(steps),sources=len(windows),matched=len(matched),summary=result['summary']),ensure_ascii=False))

if __name__=='__main__':main()
