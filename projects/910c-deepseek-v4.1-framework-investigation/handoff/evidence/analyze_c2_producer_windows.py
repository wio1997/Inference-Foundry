#!/usr/bin/env python3
"""Bind frozen C2 producer/reader windows from archived native task records.

No device run. Source-selected layers 2/8/14; no old candidate list is read.
Intervals use one rank's clock only. A native interval is not a service gain.
"""
import argparse
import collections
import hashlib
import json
import statistics
import zlib
from pathlib import Path


def shape(s):
    s=s.strip('"')
    return [tuple(int(v) for v in a.split(',')) if a else () for a in s.split(';')]


def stats(values):
    return None if not values else dict(n=len(values),min=min(values),median=statistics.median(values),max=max(values))


def union_ns(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        assert a<=b
        if end is None or a>=end:total+=b-a
        elif b>end:total+=b-end
        end=max(end or b,b)
    return total


def exposure_inputs(result, arrival):
    rows=[]
    for section in arrival['sections']:
        for layer in (2,8,14):
            original=section['layers'][layer];assert original['layer']==layer
            groups=[]
            for dp in (0,1):
                group=[w for w in result['windows'] if (w['dp'],w['carrier'],w['layer'])==(dp,section['M'],layer)]
                assert sorted(w['tp'] for w in group)==list(range(8))
                groups.append([dict(tp=w['tp'],branch=w['branch'],native_pair_us=w.get('fp32_native_union_us',0)) for w in group])
            rows.append(dict(carrier=section['M'],layer=layer,DP0_minus_DP1_us=original['DP0_minus_DP1_us'],groups=groups))
    return dict(stage_rows=rows,carrier_order=[s['M'] for s in arrival['sections']],
        requests=[{k:x[k] for k in ('slot','DP','first_eager_M','TTFT_ms')} for x in arrival['requests']],
        assumptions=['Grant a uniform group advance equal to its largest removable pair time; this overcredits partial-rank treatment.',
                     'Hold all other work, EP service and schedule effects fixed; use the archived stable clock-offset interval and original request order.',
                     'Latest EP-entry advance is a conditional exposure proxy, not a proved client TTFT shift or a universal whole-branch ceiling.'])


def price_exposure(data):
    def gain(d,a,b):return max(d,0)-max(d-a,-b)
    out={}
    for branch in ('all_FP32_pairs','cache_only_FP32_pairs'):
        for remaining in (0,.125,.25,.5):
            carriers={m:0. for m in data['carrier_order']}
            for row in data['stage_rows']:
                credit=[]
                for group in row['groups']:
                    eligible=[x['native_pair_us'] for x in group if branch=='all_FP32_pairs' or x['branch']=='cache_only']
                    credit.append(max(eligible,default=0)*(1-remaining))
                lo,hi=row['DP0_minus_DP1_us']
                carriers[row['carrier']]+=max(gain(lo,*credit),gain(hi,*credit))/1000
            requests=[]
            for req in data['requests']:
                n=data['carrier_order'].index(req['first_eager_M'])+1
                ms=sum(carriers[m] for m in data['carrier_order'][:n])
                requests.append(dict(**req,conditional_credited_ms=ms,conditional_credited_percent=100*ms/req['TTFT_ms']))
            out[f'{branch}/remaining{remaining}']=dict(carriers_ms=carriers,requests=requests,
                maximum_request_percent=max(x['conditional_credited_percent'] for x in requests),
                meets_one_percent_prioritization_reference=any(x['conditional_credited_percent']>=1 for x in requests))
    return out


def analyze(compute, comm):
    assert len(compute['profiles'])==len(comm['profiles'])==16
    windows=[]
    for p,c in zip(compute['profiles'],comm['profiles']):
        assert (p['database'],p['sha256'],p['DP'],p['TP'])==(c['database'],c['sha256'],c['DP'],c['TP'])
        for b,cw in zip(p['bodies'],c['windows']):
            assert b['F21_carrier_rows']==cw['carrier_rows']
            by={row[0]:row for row in b['rows']};h=[by[i] for i in b['HcPre_TASK_ROWS']]
            assert len(h)==80
            stream,={r[3] for r in h}
            posts=[row for row in b['rows'] if row[3]==stream and p['regimes'][row[-1]][1]=='HcPost']
            posts.sort(key=lambda x:x[1]);assert len(posts)==80
            for layer in (2,8,14):
                pre,post=h[2*layer],posts[2*layer]
                rows=[r for r in b['rows'] if pre[1]<=r[1] and r[2]<=post[1]]
                def regime(r):return p['regimes'][r[-1]]
                def pick(pred):return [r for r in rows if pred(regime(r))]
                hc_local=shape(regime(pre)[2])[0][0]
                qr=pick(lambda g:g[1].startswith('MatMul') and g[3]=='DT_BF16;DT_BF16' and shape(g[2])[1:]==[(1280,5120)])
                bf16=pick(lambda g:g[1].startswith('MatMul') and g[3]=='DT_BF16;DT_BF16' and shape(g[2])[1:]==[(512,5120)])
                fp32=pick(lambda g:g[1].startswith('MatMul') and g[3]=='FLOAT;FLOAT' and shape(g[2])[1:]==[(512,5120)])
                pool=pick(lambda g:g[1]=='_pool_kernel')
                ring=pick(lambda g:g[1]=='_cache_update_kernel')
                smla=pick(lambda g:g[1]=='SparseFlashMla')
                norms=pick(lambda g:g[1]=='RmsNorm' and shape(g[2])[0]==(hc_local,5120))
                assert len(norms)==1 and len(qr)<=1 and len(smla)<=1
                ag=[r for r in cw['communications'] if pre[2]<=r[0] and r[1]<=post[1] and r[4]=='hcom_allGather_']
                assert len(ag)==1 and ag[0][7:9]==[hc_local*5120,8]
                gather=ag[0];assert norms[0][2]<=gather[0]
                active=bool(fp32)
                global_rows=shape(regime(bf16[0])[2])[0][0] if bf16 else 0
                qrows=shape(regime(qr[0])[2])[0][0] if qr else 0
                assert qrows==max(0,min(hc_local,global_rows-p['TP']*hc_local))
                if active:
                    assert len(fp32)==2 and len(bf16)==1 and len(pool)==len(ring)==1
                    fp32.sort(key=lambda x:x[1]);a,z=fp32
                    assert all(shape(regime(r)[2])[0]==(global_rows,5120) for r in fp32)
                    assert a[3]==z[3]==pool[0][3]==ring[0][3]==stream
                    assert a[2]<=z[1]<=z[2]<=pool[0][1]<=pool[0][2]<=ring[0][1]
                    assert gather[1]<=bf16[0][1] and gather[1]<=a[1]
                    scatter=pick(lambda g:g[1]=='ScatterNdUpdateSk' and g[3]=='DT_BF16;INT32;DT_BF16' and shape(g[2])[-1]==(global_rows,512))
                    swa=[r for r in scatter if r[2]<=pool[0][1]]
                    long=[r for r in scatter if r[1]>=pool[0][2]]
                    assert len(swa)==len(long)==1
                    if smla:assert long[0][2]<=smla[0][1]
                else:
                    assert not bf16 and not pool and not ring and not qr and not smla
                def compact(r):
                    return dict(task_row=r[0],start_us=(r[1]-pre[1])/1000,end_us=(r[2]-pre[1])/1000,
                        stream=r[3],op=regime(r)[1],inputs=regime(r)[2],dtypes=regime(r)[3])
                record=dict(dp=p['DP'],tp=p['TP'],carrier=b['F21_carrier_rows'],layer=layer,
                    body_first_hc_task=b['HcPre_TASK_ROWS'][0],hc_task=pre[0],hc_local_rows=hc_local,
                    global_cache_rows=global_rows,local_query_rows=qrows,
                    branch='no_cache_or_query' if not active else 'cache_only' if not qr else 'query_and_cache',
                    SP_CP_row_intersection_matches=True,
                    norm=compact(norms[0]),gather=dict(name=gather[3],connection_id=gather[2],count=gather[7],dtype=gather[8],start_us=(gather[0]-pre[1])/1000,end_us=(gather[1]-pre[1])/1000),
                    query_projection=compact(qr[0]) if qr else None,fp32_projections=[compact(r) for r in fp32],
                    pool=compact(pool[0]) if pool else None,ring=compact(ring[0]) if ring else None,
                    smla=compact(smla[0]) if smla else None)
                if active:
                    a,z=fp32
                    other=[r for r in rows if r[0] not in {a[0],z[0]} and r[3]!=stream]
                    overlaps=[(max(a[1],r[1]),min(z[2],r[2])) for r in other if max(a[1],r[1])<min(z[2],r[2])]
                    record.update(swa_write=compact(swa[0]),long_write=compact(long[0]),
                        fp32_native_union_us=union_ns([(r[1],r[2]) for r in fp32])/1000,
                        pair_start_to_pool_us=(pool[0][1]-a[1])/1000,
                        other_stream_native_overlap_with_pair_us=union_ns(overlaps)/1000,
                        norm_to_fp32_start_us=(a[1]-norms[0][2])/1000,
                        pair_to_long_publish_us=(long[0][2]-z[2])/1000,
                        pool_end_to_smla_us=(smla[0][1]-pool[0][2])/1000 if smla else None)
                windows.append(record)
    groups={}
    for dp,carrier in sorted({(x['dp'],x['carrier']) for x in windows}):
        rows=[x for x in windows if (x['dp'],x['carrier'])==(dp,carrier)]
        names=['fp32_native_union_us','pair_start_to_pool_us','other_stream_native_overlap_with_pair_us','norm_to_fp32_start_us','pair_to_long_publish_us','pool_end_to_smla_us']
        groups[f'DP{dp}/carrier{carrier}']=dict(windows=len(rows),branches=dict(collections.Counter(x['branch'] for x in rows)),
            global_cache_rows=sorted({x['global_cache_rows'] for x in rows}),hc_local_rows=sorted({x['hc_local_rows'] for x in rows}),
            metrics={k:stats([x[k] for x in rows if x.get(k) is not None]) for k in names})
    return dict(status='ARCHIVED_STAGE_BINDING_NOT_NEW_PERFORMANCE',windows=windows,groups=groups,
        counts=dict(windows=len(windows),branches=dict(collections.Counter(x['branch'] for x in windows)),
            native_FP32_calls_by_op=dict(collections.Counter(r['op'] for x in windows for r in x['fp32_projections']))),
        limits=['HcPre/FFN pair ordinal and exact typed shape/op order bind the archived source role, not tensor-address identity or current binary attestation.',
                'Same-rank intervals only; other-stream absence does not prove AIC idle, DMA freedom, no host waits or request criticality.',
                'Pair is before pool on the original main stream; deleting/accelerating it is not automatically a request gain under changed collective readiness and EP joins.',
                'Large prefill and tiny/cache-only branches remain separate. No extrapolated local-M FP32 kernel cost or numerical equivalence is asserted.'],new_NPU_measurements=0)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--exposure-inputs',type=Path);ap.add_argument('--recompute-exposure',type=Path);a=ap.parse_args()
    if a.recompute_exposure:
        result=price_exposure(json.loads(a.recompute_exposure.read_text()))
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('Portable conditional exposure recomputed; not a device measurement.')
        raise SystemExit(0)
    if not a.archive:ap.error('--archive is required unless --recompute-exposure is used')
    names=['first-token-v168-protocol-bodies.json.zlib','first-token-v168-collectives.json.zlib']
    paths=[a.archive/'.cache'/n for n in names]
    result=analyze(*(json.loads(zlib.decompress(p.read_bytes())) for p in paths))
    result['inputs']=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    arrival_path=a.archive/'full-review/PREFILL-OA-JOINT-ARRIVAL.json'
    ei=exposure_inputs(result,json.loads(arrival_path.read_text()))
    ei['source_pins']=result['inputs']+[dict(path=str(arrival_path),sha256=hashlib.sha256(arrival_path.read_bytes()).hexdigest())]
    result['conditional_exposure']=price_exposure(ei)
    if a.exposure_inputs:a.exposure_inputs.write_text(json.dumps(ei,ensure_ascii=False,indent=2)+'\n')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(counts=result['counts'],conditional_exposure={k:dict(maximum_request_percent=v['maximum_request_percent'],carriers_ms=v['carriers_ms']) for k,v in result['conditional_exposure'].items()}),ensure_ascii=False))
