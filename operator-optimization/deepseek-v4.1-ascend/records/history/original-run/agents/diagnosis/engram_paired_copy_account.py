#!/usr/bin/env python3
"""Read retained diagnostic aggregates in the stage container; emit only counters."""
import collections, hashlib, json
from pathlib import Path
RUN='001d14496b2146bba25dae0f27c780b5'
SHA='b32f03b3aca61939fbf76e56a6c7faaa0d1d8a9c4d225515f03e72082c44cfbf'
base=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029')/('engram-diagnostic-'+RUN)/'runtime'
p=json.loads((base/'engram-client-results/progress.json').read_text());assert p['run_id']==RUN and p['state']=='COMPLETE';win=p['window_id']
raw=(base/'metadata/engram-metadata.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA
events=[json.loads(x) for x in raw.splitlines()];sts=[json.loads(x.read_text()) for x in (base/'metadata').glob('observer-*.json')]
status=collections.defaultdict(list)
for s in sts:status[s.get('process_instance')].append(s)
beg=collections.defaultdict(list);end=collections.defaultdict(list)
for e in events:
 if e.get('event') in ('embed_begin','embed_end') and e.get('window_id')==win:
  (beg if e['event']=='embed_begin' else end)[(e.get('process_instance'),e.get('call'))].append(e)
shapes=collections.Counter()
for k,values in beg.items():
 es=end.get(k,[]);ss=status[k[0]]
 if len(values)!=1 or len(es)!=1 or es[0].get('outcome')!='returned' or len(ss)!=1:continue
 b=values[0];s=ss[0];t=b.get('tensor')
 if not (s.get('run_id')==RUN and s.get('window_id')==win and s.get('pid')==b.get('pid')==es[0].get('pid') and isinstance(t,dict)):continue
 shape=t.get('shape');assert t.get('dtype')=='torch.int32' and t.get('device_type')=='npu' and t.get('is_contiguous') is False and t.get('stride')==[48,1] and len(shape)==2 and shape[1]==24 and shape[0]>=24
 shapes[shape[0]]+=1
assert sum(shapes.values())==826
print(json.dumps({'run_id':RUN,'log_sha256':SHA,'paired_records':sum(shapes.values()),'paired_rows_histogram':dict(sorted(shapes.items())),'conditional_intermediate_logical_bytes':sum(row*24*4*n for row,n in shapes.items()),'not_actual_hbm_or_performance':True},indent=2))
