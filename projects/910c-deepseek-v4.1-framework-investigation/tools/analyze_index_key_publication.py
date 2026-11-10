#!/usr/bin/env python3
"""Bind original native multi-output cache producers before judging subfusions."""
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import zlib

from analyze_attention_output_delivery import extract
from analyze_c2_producer_windows import shape, union_ns

BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')


def main():
    data=extract(BASE)
    p=BASE/'.cache/first-token-v168-protocol-bodies.json.zlib'
    native=json.loads(zlib.decompress(p.read_bytes()))
    cp=BASE/'.cache/first-token-v168-collectives.json.zlib'
    comm=json.loads(zlib.decompress(cp.read_bytes()))
    cfg=json.loads((OUT/'startup-scope-evidence.json').read_text())['model_config']['raw_config']['text_config']
    layers=cfg['kv_source_layer_ids'];assert layers==[2,8,14,20]
    windows=[];groups=defaultdict(list)
    for prof,cprof in zip(native['profiles'],comm['profiles']):
        assert (prof['database'],prof['sha256'],prof['DP'],prof['TP'])==(cprof['database'],cprof['sha256'],cprof['DP'],cprof['TP'])
        regs=prof['regimes']
        for body,cbody in zip(prof['bodies'],cprof['windows']):
            assert body['F21_carrier_rows']==cbody['carrier_rows']
            by={r[0]:r for r in body['rows']};pres=[by[x] for x in body['HcPre_TASK_ROWS']]
            stream,={r[3] for r in pres}
            posts=sorted((r for r in body['rows'] if r[3]==stream and regs[r[-1]][1]=='HcPost'),key=lambda r:r[1])
            assert len(pres)==len(posts)==80
            for layer in layers:
                lo,hi=pres[2*layer],posts[2*layer]
                rs=sorted((r for r in body['rows'] if lo[1]<=r[1] and r[2]<=hi[1]),key=lambda r:(r[1],r[0]))
                def pick(op,pred=lambda g:True):return [r for r in rs if (regs[r[-1]][1].startswith('MatMul') if op=='MATMUL' else regs[r[-1]][1]==op) and pred(regs[r[-1]])]
                wk=pick('MATMUL',lambda g:shape(g[2])[1:]==[(128,512)] and g[3]=='DT_BF16;DT_BF16')
                base=dict(dp=prof['DP'],tp=prof['TP'],carrier=body['F21_carrier_rows'],layer=layer,ratio=cfg['compress_ratios'][layer])
                if not wk:
                    windows.append(dict(**base,branch='no_cache_publication',rows=0));continue
                assert len(wk)==1
                wk=wk[0];m=shape(regs[wk[-1]][2])[0][0]
                kn,=pick('RmsNorm',lambda g:shape(g[2])[0]==(m,128))
                kq,=pick('DynamicQuant',lambda g:shape(g[2])[0]==(m,128))
                kr,=pick('InplacePartialRotaryMul',lambda g:shape(g[2])[0]==(m,1,1,128))
                ki,=pick('ScatterNdUpdateSk',lambda g:g[3]=='INT8;INT32;INT8' and shape(g[2])[-1]==(m,128))
                ks,=pick('ScatterNdUpdateSk',lambda g:g[3]=='FLOAT16;INT32;FLOAT16' and shape(g[2])[-1]==(m,1))
                sc,=pick('Cast',lambda g:g[3]=='FLOAT' and shape(g[2])[0]==(m,1))
                long=[r for r in pick('ScatterNdUpdateSk',lambda g:g[3]=='DT_BF16;INT32;DT_BF16' and shape(g[2])[-1]==(m,512)) if r[1]>=ks[2]]
                assert len(long)==1;long=long[0]
                lr,=[r for r in pick('InplacePartialRotaryMul',lambda g:shape(g[2])[0]==(m,1,1,512)) if ks[2]<=r[1]<long[1]]
                ln=[r for r in pick('RmsNorm',lambda g:shape(g[2])[0]==(m,512)) if r[2]<=wk[1]];assert ln
                ln=max(ln,key=lambda r:r[1])
                suffix=[ln,wk,kn,kr,kq,ki,sc,ks,lr,long]
                assert all(r[3]==stream for r in suffix)
                assert all(a[2]<=b[1] for a,b in zip(suffix,suffix[1:]))
                if base['ratio']==2:
                    wv=pick('MATMUL',lambda g:g[3]=='FLOAT;FLOAT' and shape(g[2])==[(m,5120),(512,5120)])
                    pool,=pick('_pool_kernel');ring,=pick('_cache_update_kernel');assert len(wv)==2
                    casts=[r for r in pick('Cast',lambda g:g[3]=='DT_BF16' and shape(g[2])[0]==(m,5120) and g[5]=='FLOAT') if r[2]<=min(x[1] for x in wv)]
                    assert len(casts)==1
                    zeros=[r for r in pick('ZerosLike',lambda g:shape(g[2])[0]==(m,512)) if max(x[2] for x in wv)<=r[1]<pool[1]]
                    assert len(zeros)<=1
                    root=casts[0];extra=casts+wv+zeros+[pool,ring]
                else:
                    wv=[r for r in pick('MATMUL',lambda g:g[3]=='DT_BF16;DT_BF16' and shape(g[2])==[(m,5120),(512,5120)]) if r[2]<=ln[1]]
                    assert wv;root=max(wv,key=lambda r:r[1]);extra=[root]
                producer=sorted(extra+suffix,key=lambda r:r[1])
                gemms=[r for r in producer if regs[r[-1]][1].startswith('MatMul')]
                gemm_us=union_ns([(r[1],r[2]) for r in gemms])/1000
                non_gemms=[r for r in producer if not regs[r[-1]][1].startswith('MatMul')]
                hc_local=shape(regs[lo[-1]][2])[0][0]
                ag=[r for r in cbody['communications'] if lo[2]<=r[0] and r[1]<=hi[1] and r[4]=='hcom_allGather_']
                assert len(ag)==1 and ag[0][7:9]==[hc_local*5120,8]
                gather=ag[0];assert gather[1]<=root[1]
                qli=pick('VllmAscendQuantLightningIndexerV2')
                qrows=shape(regs[qli[0][-1]][2])[0][0] if qli else 0
                w=dict(**base,rows=m,query_rows=qrows,branch='query_and_cache' if qrows else 'cache_only',
                       suffix_native_us=union_ns([(r[1],r[2]) for r in suffix])/1000,
                       suffix_parent_us=(long[2]-ln[1])/1000,
                       producer_native_us=union_ns([(r[1],r[2]) for r in producer])/1000,
                       producer_parent_us=(long[2]-root[1])/1000,
                       producer_GEMM_native_us=gemm_us,
                       producer_non_GEMM_native_us=union_ns([(r[1],r[2]) for r in non_gemms])/1000,
                       producer_non_GEMM_parent_us=(long[2]-root[1])/1000-gemm_us,
                       input_ready_parent_us=(long[2]-gather[1])/1000,
                       producer_early_overlap_upper_us=min((long[2]-root[1])/1000,(root[1]-gather[1])/1000),
                       input_gather_end_ns=gather[1],
                       latent_norm_task=ln[0],long_publish_task=long[0],root_task=root[0],
                       suffix_tasks=[dict(task=r[0],op=regs[r[-1]][1],us=(r[2]-r[1])/1000,shape=regs[r[-1]][2],dtypes=regs[r[-1]][3]) for r in suffix])
                windows.append(w)
                for r in producer:
                    key=(prof['DP'],body['F21_carrier_rows'],*regs[r[-1]])
                    groups[key].append((r[2]-r[1])/1000)
    assert len(windows)==256
    # Producer copies are subsets of signatures that may also serve SWA/other
    # readers; do not falsely require a whole-signature call count to equal ours.
    units_path=BASE/'full-review/PROFILE-UNIT-COSTS.json';units=json.loads(units_path.read_text())['signatures'];matched=[]
    for key,vals in groups.items():
        dp,m,*reg=key
        u,=[x for x in units if x['phase']=='eager_first_token' and [x[k] for k in ['name','opType','inputShapes','inputDataTypes','outputShapes','outputDataTypes']]==reg]
        c,=[x for x in u['conditions'] if (x['dp'],x['carrier'])==(dp,m)]
        assert c['calls']>=len(vals) and min(vals)>=c['observed_us']['min']-1e-7 and max(vals)<=c['observed_us']['max']+1e-7
        matched.append(dict(dp=dp,carrier=m,signature=list(reg),our_subset_calls=len(vals),whole_signature_calls=c['calls'],our_us=dict(min=min(vals),median=median(vals),max=max(vals))))
    lookup={(w['dp'],w['tp'],w['carrier'],w['layer']):w for w in windows}
    def price(field,rho,h):
        carriers={m:0. for m in data['carrier_order']}
        for sec in data['arrival_sections']:
            for layer in layers:
                ds=[]
                for dp in [0,1]:
                    ds.append(max(max(0.,rho*w.get(field,0.)-h) for tp in range(8) for w in [lookup[dp,tp,sec['M'],layer]]))
                lo,hi=sec['layers'][layer]['DP0_minus_DP1_us']
                def gain(d):return max(d,0)-max(d-ds[0],-ds[1])
                carriers[sec['M']]+=max(gain(lo),gain(hi))/1000
        req=[]
        for r in data['requests']:
            credit=sum(carriers[m] for m in data['carrier_order'][:data['carrier_order'].index(r['first_eager_M'])+1])
            req.append(dict(slot=r['slot'],old_ms=r['TTFT_ms'],conditional_credit_ms=credit))
        old=max(r['old_ms'] for r in req);saved=old-max(r['old_ms']-r['conditional_credit_ms'] for r in req)
        return dict(field=field,rho=rho,added_us_per_source=h,P90_saved_ms=saved,P90_percent=100*saved/old,requests=req)
    prices=[price(f,r,h) for f in ['suffix_native_us','suffix_parent_us','producer_native_us','producer_parent_us','producer_non_GEMM_native_us','producer_non_GEMM_parent_us','producer_early_overlap_upper_us','input_ready_parent_us'] for r,h in [(1.,0.),(.5,5.),(.5,20.)]]
    geometry=[]
    for dp,m in sorted({(w['dp'],w['carrier']) for w in windows}):
        g=[w for w in windows if (w['dp'],w['carrier'])==(dp,m)]
        geometry.append(dict(dp=dp,carrier=m,windows=len(g),branches=dict(Counter(w['branch'] for w in g)),global_cache_rows=sorted({w['rows'] for w in g}),query_rows=sorted({w.get('query_rows',0) for w in g})))
    result=dict(status='ACTUAL_NATIVE_LATENT_MULTI_OUTPUT_PARENT_VALUE',whole_chain_complete=False,each_replacement_bitwise=True,
                windows=windows,geometry=geometry,independent_signature_subset_conditions=matched,conditional_values=prices,
                scope='Original16rank first-token native table only; includes zero-query cache-only producers. Suffix native set10 calls; parent span also includes gaps, not all removable CPU work. Non-GEMM parent subtracts original GEMMs but still overcredits all other gaps. Input-ready parent includes retained query and SWA work and is only a joint-overlap scope, not deletable work. Root/input-ready parents overlap old C2/host/SMLA parents, no additive gains. Fixed DP-arrival sensitivity, not causal/expected/request timing measurement.',
                source_pins=[*data['source_pins'],dict(path=str(cp),sha256=sha256(cp.read_bytes()).hexdigest()),dict(path=str(units_path),sha256=sha256(units_path.read_bytes()).hexdigest())],
                next_action='Use whole native/span prices to retain/prune scope; inspect actual RMS/quant/scatter implementations, complete/skip rows and input mutation lease. Shared owner/split-row schemes must retain geometry/bytes and pay communication/readiness, not count8duplicate work as critical gain.',
                new_NPU_work=0,new_profile_collection=0,model_loads=0,service_requests=0,remote_writes=[])
    (OUT/'index-key-publication-profile.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(windows=len(windows),geometry=geometry,signature_subset_conditions=len(matched),prices=[{k:v for k,v in p.items() if k!='requests'} for p in prices]),ensure_ascii=False))


if __name__=='__main__':main()
