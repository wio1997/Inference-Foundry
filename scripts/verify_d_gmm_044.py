#!/usr/bin/env python3
"""Validate profiler numeric fields for dominant-stream GMM samples.

The metrics are not independent timings, not calibrated bandwidth or proof of
same expert workload; this offline calculation does not access the device.
"""
import gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'evidence/d_gmm_043/counters.json.gz'
SHA='e50dead51a02affaff6870e9a4eb6a64896cb290dc0567f8957d93828e78f811'
FIELDS=('aic_mac_time(us)','aic_mte1_time(us)','aic_mte2_time(us)','aicore_time(us)','Wait Time(us)')


def need(ok,msg):
 if not ok:raise ValueError(msg)


def calculate():
 raw=gzip.decompress(SRC.read_bytes());need(hashlib.sha256(raw).hexdigest()==SHA,'source SHA mismatch');d=json.loads(raw)
 ranks={(r['engine'],r['rank']):r for r in d['ranks']};need(len(ranks)==len(d['ranks'])==16,'rank coverage')
 out={}
 for shape in ('8192,2048','8192,4096'):
  rows=[]
  for rank in range(8):
   r=ranks[('sg',rank)];found=[s for s in r['signatures'] if s['input_shape'].startswith(shape+';')];need(len(found)==1,'shape missing');o=found[0]['counter_observations'];streams=[(k,v) for k,v in o['by_stream'].items() if v['rows']==336];need(len(streams)==1 and o['rows']==360,'dominant-stream sample mismatch');stream,entry=streams[0]
   item={'rank':rank,'stream':stream,'event_count':entry['rows']}
   for name in FIELDS:
    m=entry['metrics'][name];need(m['available']==336 and m['unavailable']==0,'counter coverage');v=m['p50'];need(isinstance(v,(float,int)) and math.isfinite(v) and v>=0,'nonfinite metric');item[name]=v
   rows.append(item)
  out[shape]=rows
 need(abs(out['8192,2048'][0]['aic_mac_time(us)']-7.207)<1e-9 and all(abs(r['aic_mac_time(us)']-1.441)<1e-9 for r in out['8192,2048'][1:]),'reference mac medians')
 return {'source_sha256':SHA,'profiles':out,'scope':'dominant stream per rank/shape; independent medians, event overlap, unmatched token/group-list/step'}


if __name__=='__main__':
 try:print(json.dumps(calculate(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
