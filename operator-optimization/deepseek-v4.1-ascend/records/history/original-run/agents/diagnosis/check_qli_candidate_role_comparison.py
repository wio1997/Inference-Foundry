#!/usr/bin/env python3
"""Compare existing QLI tasks by physical IO signature, not infer tensor edges."""
import hashlib,json,statistics
from decimal import Decimal
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
sha=hashlib.sha256(p.read_bytes()).hexdigest()
assert sha=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
want='db852bff987d2b98d3dc0734e3731ec5528e37dcf6e874860d1ebb4fd12eefc8'
layers=(20,24,28,32,36); records={}
for line in p.open():
 j=json.loads(line)
 if j.get('kind')!='rank': continue
 for s in j['data']['layers']:
  if s['scheduler_digest']!=want or s['ep'] not in range(8) or s['layer'] not in layers: continue
  groups=[g for g in s['operator_groups'] if g['operator_name']=='VllmAscendQuantLightningIndexerV2']
  ts=[(g,t) for g in groups for t in g['tasks']]
  assert len(ts)==1
  g,t=ts[0];key=(s['ep'],s['layer']); assert key not in records
  assert Decimal(t['duration_abs_difference_us'])<=Decimal('.001')
  records[key]={'signature':g['physical_shape_dtype_format'],'duration':Decimal(t['trace_duration_us']),'track':(t['device_track_pid'],t['device_track_tid']),'core':t['csv']['Accelerator Core'],'block':t['csv']['Block Num'],'mix':t['csv']['Mix Block Num']}
assert len(records)==40
for ep in range(7):
 for layer in layers:
  z=records[ep,layer];ref=records[0,layer]
  assert z['signature']==ref['signature'] and (z['core'],z['block'],z['mix'])==('MIX_AIC','24','48')
  assert z['track']==records[ep,20]['track']
a=records[0,20]['signature'];b=records[0,24]['signature']
assert all(records[0,l]['signature']==b for l in layers[1:])
# Differences are physical signature fields, not actual mode, bytes transferred, or alias identity.
assert a['Input Shapes'].endswith('1024;"') and b['Input Shapes'].endswith('1024;1023,1,2048"')
assert a['Output Shapes']=='"1023,1,512;;1023,1,2048"' and b['Output Shapes']=='"1023,1,512;;"'
assert a['Input Data Types'].split(';')[-1]=='DT_UNDEFINED' and b['Input Data Types'].split(';')[-1]=='INT32'
assert a['Output Data Types'].split(';')[-1]==b['Output Data Types'].split(';')[-1]=='INT32'
assert a['Input Formats'].split(';')[-1]=='NULL' and b['Input Formats'].split(';')[-1]=='ND'
assert a['Output Formats']==b['Output Formats']=='ND;ND;ND'
for field in ('Input Shapes','Input Data Types','Input Formats'):
 assert a[field].strip('\"').split(';')[:-1] == b[field].strip('\"').split(';')[:-1]
assert a['Output Shapes'].strip('\"').split(';')[:-1] == b['Output Shapes'].strip('\"').split(';')[:-1]
assert a['Output Data Types']==b['Output Data Types']
assert '1019,32,128' in records[7,20]['signature']['Input Shapes']
ratios=[records[e,l]['duration']/records[e,20]['duration'] for e in range(7) for l in layers[1:]]
result={'source_sha256':sha,'selection':'same saved digest; EP0-6; layer 20 paired to 24,28,32,36 within EP','physical_signatures':{'layer20':a,'later':b},'per_ep_duration_us':{str(e):{str(l):str(records[e,l]['duration']) for l in layers} for e in range(7)},'ratio_min_median_max':[str(min(ratios)),str(statistics.median(ratios)),str(max(ratios))],'ep7_exclusion':'recorded query shape uses 1019 rows, not 1023','limits':['Different layers/roles and workload; no same-input experiment, tensor/storage identity, values, effective length, mode, binary or profile perturbation established.','Recorded task time is device trace interval matched to CSV within 0.001us, not client or unprofiled end-to-end latency.','Block fields and physical shapes are not actual executed instruction counts or proof of no contention.']}
print(json.dumps(result,ensure_ascii=False,indent=2))
