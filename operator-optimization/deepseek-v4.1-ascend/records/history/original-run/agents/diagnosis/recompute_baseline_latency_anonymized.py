#!/usr/bin/env python3
"""Recompute frozen AISBench latency, export numeric timing only, never response/input."""
import hashlib, io, json, pathlib, sqlite3, statistics
import numpy as np
ROOT = pathlib.Path('/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/results')
OUT = pathlib.Path(__file__).with_name('baseline-latency-anonymized.json')
phases={}; idsets={}
for phase in ('measure1','measure2'):
    base=ROOT/phase
    details=list(base.rglob('*details.jsonl')); dbs=list(base.rglob('*.db'))
    if len(details)!=1 or len(dbs)!=1: raise ValueError('ambiguous source')
    conn=sqlite3.connect(f'file:{dbs[0]}?mode=ro',uri=True)
    parsed=[]; refs=[]
    with details[0].open() as handle:
        for line in handle:
            if not line.strip(): continue
            record=json.loads(line)
            # Retain only the fields required to join the stored timing and audit cardinality.
            data_id=record['data_id']; token_count=record['output_tokens']; ref=record['time_points']['__db_ref__']; ok=record['success']; del record
            if not ok or token_count!=1024: raise ValueError('failed or differing output count')
            dbrow=conn.execute('SELECT arr_blob FROM numpy_store WHERE id=?',(ref,)).fetchall()
            if len(dbrow)!=1: raise ValueError('missing/non-unique reference')
            points=np.load(io.BytesIO(dbrow[0][0]),allow_pickle=False)
            if points.ndim!=1 or len(points)<3 or not np.isfinite(points).all() or (np.diff(points)<0).any(): raise ValueError('invalid timepoints')
            refs.append(ref);parsed.append((data_id,token_count,points))
    conn.close()
    if len(parsed)!=8 or len(set(refs))!=8 or len({r[0] for r in parsed})!=8: raise ValueError('ambiguous record join')
    parsed.sort(key=lambda x:x[0]); idsets[phase]=set(r[0] for r in parsed)
    origin=min(r[2][0] for r in parsed); records=[]
    for i,(_,count,a) in enumerate(parsed,1):
        records.append({'ordinal_by_data_id':i,'start_offset_ms':float((a[0]-origin)*1000),'first_response_offset_ms':float((a[1]-origin)*1000),'last_response_offset_ms':float((a[-1]-origin)*1000),'ttft_ms':float((a[1]-a[0])*1000),'tpot_ms':float((a[-1]-a[1])*1000/(count-1)),'client_latency_ms':float((a[-1]-a[0])*1000),'recorded_non_done_sse_count':len(a)-1})
    summary={}
    for key in ('start_offset_ms','first_response_offset_ms','last_response_offset_ms','ttft_ms','tpot_ms','client_latency_ms'):
        v=[r[key] for r in records];summary[key]={'min':min(v),'max':max(v),'mean':statistics.mean(v),'spread':max(v)-min(v)}
    phases[phase]={'records':records,'summary':summary,'details_sha256':hashlib.sha256(details[0].read_bytes()).hexdigest(),'db_sha256':hashlib.sha256(dbs[0].read_bytes()).hexdigest()}
if idsets['measure1']!=idsets['measure2']: raise ValueError('positions do not match across datasets')
a=phases['measure1']['records'];b=phases['measure2']['records']
deltas=[y['ttft_ms']-x['ttft_ms'] for x,y in zip(a,b)]
result={'definition':'start=before POST; first=first nonempty noncomment non-DONE SSE line recorded before parsing; last=last recorded non-DONE SSE line; TTFT=first-start, TPOT=(last-first)/(1024-1); not device time or per-token event','caveat':'matching data_id across phases identifies dataset positions only; does not prove prompt identity, engine identity, device origin, or causality','cross_round_ttft_delta_ms':{'min':min(deltas),'max':max(deltas),'mean':statistics.mean(deltas),'max_abs':max(map(abs,deltas))},'phases':phases}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('saved',OUT)
for name,phase in phases.items(): print(name,'start-spread',phase['summary']['start_offset_ms']['spread'],'first-spread',phase['summary']['first_response_offset_ms']['spread'],'last-spread',phase['summary']['last_response_offset_ms']['spread'],'ttft',phase['summary']['ttft_ms'],'tpot',phase['summary']['tpot_ms'])
print('cross-round-ttft',result['cross_round_ttft_delta_ms'])
