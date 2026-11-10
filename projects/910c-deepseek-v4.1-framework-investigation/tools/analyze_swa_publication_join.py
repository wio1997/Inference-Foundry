#!/usr/bin/env python3
"""Fixed-arrival SWA/Qb join sensitivity from saved native tasks, no runtime replay."""
from bisect import bisect_left
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json,statistics,zlib

ROOT=Path('audit/execution-search')
def start(r):return Decimal(r['data']['Start Time(us)'].strip())
def duration(r):return Decimal(r['data']['Duration(us)'])
def end(r):return start(r)+duration(r)
def clean(s):return s.strip('"')

def main():
    arch=ROOT/'swa-publication-parent-calls.json.zlib';sitefile=ROOT/'cache-epilogue-balanced-sites.json'
    profiles=json.loads(zlib.decompress(arch.read_bytes()));sites=json.loads(sitefile.read_text())['profiles'];out=[]
    for p in profiles:
        s=next(s for s in sites if s['ep']==p['ep']);fam=next(f for f in s['families'] if f['name']=='norm_rope_cache512');rows=p['selected'];byrow={r['csv_row']:r for r in rows}
        qb=defaultdict(list);qrope=defaultdict(list);smla=defaultdict(list)
        for r in rows:
            d=r['data'];t=d['Type'];stream=d['Stream ID']
            if t=='QuantBatchMatmulV3' and clean(d['Output Shapes'])=='3,32768':qb[stream].append(r)
            if t=='InplacePartialRotaryMul' and clean(d['Input Shapes'])=='3,1,64,512;3,1,1,64;3,1,1,64':qrope[stream].append(r)
            if t=='SparseFlashMla':smla[stream].append(r)
        assert sorted(len(v) for v in qb.values())==[36,480]
        for pool in [qb,qrope,smla]:
            for v in pool.values():v.sort(key=start)
        binds=[]
        for n,count in [(24,480),(20,36)]:
            chains=sorted([c for c in fam['chains_detail'] if c['shapes'][0][0][0]==n],key=lambda c:start(byrow[c['csv_rows'][0]]))
            stream=next(k for k,v in qb.items() if len(v)==count);assert len(chains)==count
            qr_times=[start(r) for r in qrope[stream]];sm_times=[start(r) for r in smla[stream]]
            for c,q in zip(chains,qb[stream]):
                chain=[byrow[i] for i in c['csv_rows']];a=start(chain[0]);e=end(chain[-1]);qe=end(q)
                k=bisect_left(qr_times,qe);qr=qrope[stream][k]
                assert start(qr)>=e,('queryRoPE before cache end',p['ep'],c['csv_rows'])
                assert abs(start(q)-a)<Decimal(1000),('nonlocal match',p['ep'],c['csv_rows'])
                m=bisect_left(sm_times,end(qr));sm=smla[stream][m]
                assert start(sm)-start(qr)<Decimal(10000)
                binds.append(dict(row_count=n,cache_stream=c['stream'],main_stream=stream,cache_csv_rows=c['csv_rows'],qb_csv_row=q['csv_row'],query_rope_csv_row=qr['csv_row'],smla_csv_row=sm['csv_row'],
                    norm_start_us=str(a),cache_end_us=str(e),qb_start_us=str(start(q)),qb_end_us=str(qe),
                    native_sum_us=c['native_sum_us'],span_us=c['span_us'],
                    cache_later_than_qb_us=float(e-qe),post_ready_to_query_rope_us=float(start(qr)-max(e,qe)),
                    query_rope_to_smla_us=float(start(sm)-end(qr))))
        assert len(binds)==516 and len({b['qb_csv_row'] for b in binds})==516 and len({b['query_rope_csv_row'] for b in binds})==516
        scenarios=[]
        for fraction in [0.25,0.5,1.0]:
            for h in [0,5,20]:
                vals=[]
                for b in binds:
                    e=Decimal(b['cache_end_us']);q=Decimal(b['qb_end_us']);change=Decimal(str(b['native_sum_us']))*Decimal(str(fraction))-Decimal(h)
                    vals.append(float(max(e,q)-max(e-change,q)))
                net=sum(vals)
                scenarios.append(dict(native_fraction_saved=fraction,added_us_per_chain=h,fixed_qb_arrival_join_net_us=net,window_pct=net*100/s['window_us'],negative_chains=sum(v<0 for v in vals),zero_chains=sum(v==0 for v in vals),positive_chains=sum(v>0 for v in vals)))
        groups=[]
        for n in [24,20]:
            b=[b for b in binds if b['row_count']==n];exposed=[max(0,x['cache_later_than_qb_us']) for x in b]
            groups.append(dict(row_count=n,chains=len(b),main_stream=b[0]['main_stream'],cache_stream=b[0]['cache_stream'],cache_later_count=sum(x['cache_later_than_qb_us']>0 for x in b),cache_before_count=sum(x['cache_later_than_qb_us']<0 for x in b),
                positive_cache_lateness_sum_us=sum(exposed),cache_lateness_median_us=statistics.median(x['cache_later_than_qb_us'] for x in b),
                post_ready_to_query_rope_median_us=statistics.median(x['post_ready_to_query_rope_us'] for x in b),post_ready_to_query_rope_max_us=max(x['post_ready_to_query_rope_us'] for x in b)))
        out.append(dict(ep=p['ep'],dp=p['dp'],window_us=s['window_us'],csv_sha256=p['sha256'],chains=516,groups=groups,scenarios=scenarios,binds=binds))
    d=dict(status='SOURCE_CONSISTENT_NATIVE_BIJECTION_FIXED_OTHER_ARRIVALS_NOT_EVENT_PROOF',whole_chain_complete=False,every_replacement_bitwise=True,profiles=out,
        limits=['516 cache chains bijectively matched to516 typed Qb and forward-query-RoPE tasks per representative rank; local time distance<1ms, cache completion precedes query-RoPE and next SMLA. This is source-consistent native correlation, not runtime event IDs or loaded object proof.',
            'Fixed other-stream Qb completion; replacement may change Cube/Vector contention and arrivals. Scenario computes max(cacheEnd,qbEnd)-max(cacheEnd-fraction*nativeSum+added,qbEnd), preserves negative cases.',
            'Does not remove already-covered Qb time, nor post-ready-to-queryRoPE gap. No host waiting or whole-request/sixordinary credit. Not all TP/DP ranks or per-invocation exact pointer/generation.',
            '24-row480 and20-row36 stream pools are geometry bindings; target/draft labels still need recorded runtime parent correlation.'],
        inputs=[dict(path=str(arch),sha256=sha256(arch.read_bytes()).hexdigest()),dict(path=str(sitefile),sha256=sha256(sitefile.read_bytes()).hexdigest())],
        new_NPU_work=0,new_profile=0,service_requests=0,analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    p=ROOT/'swa-publication-join-price.json';p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(ep=p['ep'],groups=p['groups'],scenarios=p['scenarios']) for p in out],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
