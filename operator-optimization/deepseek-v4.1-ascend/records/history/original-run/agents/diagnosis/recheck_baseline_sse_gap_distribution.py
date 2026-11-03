#!/usr/bin/env python3
"""Summarize inter-SSE timing from frozen benchmark with no input/output text or identity."""
import argparse, hashlib, io, json, pathlib, sqlite3
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-root',required=True,type=pathlib.Path,help='Frozen benchmark results directory containing measure1 and measure2')
root=parser.parse_args().results_root
prior=json.loads(pathlib.Path(__file__).with_name('baseline-latency-anonymized.json').read_text())
results={}
for phase in ('measure1','measure2'):
 base=root/phase; ds=list(base.rglob('*details.jsonl')); db=list(base.rglob('*.db'))
 assert len(ds)==len(db)==1
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert digest(ds[0])==prior['phases'][phase]['details_sha256']
 assert digest(db[0])==prior['phases'][phase]['db_sha256']
 conn=sqlite3.connect(f'file:{db[0]}?mode=ro',uri=True); allgaps=[]; maxima=[]; above=[]; counts=[];refs=[];endpoints=[]
 for line in ds[0].open():
  if not line.strip():continue
  r=json.loads(line);ref=r['time_points']['__db_ref__'];assert r['success'] and r['output_tokens']==1024
  rows=conn.execute('SELECT arr_blob FROM numpy_store WHERE id=?',(ref,)).fetchall();assert len(rows)==1
  a=np.load(io.BytesIO(rows[0][0]),allow_pickle=False); assert a.ndim==1 and len(a)>2 and np.isfinite(a).all() and (np.diff(a)>=0).all()
  gaps=np.diff(a[1:])*1000;assert len(gaps)>0
  allgaps.extend(gaps.tolist());maxima.append(float(gaps.max()));above.append(int((gaps>100).sum()));counts.append(len(gaps));refs.append(ref)
  for i,gap in enumerate(gaps):
   if gap>100:endpoints.extend([(float(a[i+1]),1),(float(a[i+2]),-1)])
 conn.close();assert len(refs)==len(set(refs))==8
 q=np.percentile(allgaps,[50,90,99,100],method='linear')
 active=peak=0
 for _,delta in sorted(endpoints,key=lambda x:(x[0],x[1])):
  active+=delta;assert active>=0;peak=max(peak,active)
 assert active==0 and peak<=8
 results[phase]={'request_count':8,'sse_gap_count':len(allgaps),'sse_gaps_per_request_min_max':[min(counts),max(counts)],'pooled_quantiles_ms':dict(zip(('p50','p90','p99','max'),map(float,q))),'gaps_over_100ms_count':sum(above),'peak_simultaneous_requests_in_gaps_over_100ms':peak,'request_count_with_no_gap_over_100ms':above.count(0),'per_request_max_ms_min_max':[min(maxima),max(maxima)],'input_details_sha256':digest(ds[0]),'input_db_sha256':digest(db[0])}
print(json.dumps({'method':'Each gap is between successive observed non-DONE SSE lines; pooled within a single round, NumPy percentile linear method; output count 1024 per request.','limits':['SSE line not equal to one token, decoded timestep, or NPU kernel','client buffering network scheduling and profiler-free server are not individually separable','pooled gaps are dependent within requests and not independent samples','overlapping long client gaps do not identify a shared server cause or prove simultaneous device inactivity','100ms threshold is descriptive and post-hoc, not a service-level bound','measurement captured by one client and two rounds only'] ,'rounds':results},ensure_ascii=False,indent=2))
