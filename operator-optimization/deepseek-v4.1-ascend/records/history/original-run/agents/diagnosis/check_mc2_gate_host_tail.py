#!/usr/bin/env python3
"""Check the saved gating host API tail only; no device access."""
from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent/'mc2-db852-ep3-host-gap.json'
raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='f1d627e1788e4b411d539010387dc4b0a31a3504c903ad0e55546913f6591877'
j=json.loads(raw)
g=[r for r in j['CANN_API'] if r['name']=='aclnnMoeGatingTopKHash'];assert len(g)==1;g=g[0]
t=[r for r in j['target_tasks'] if r['taskId']==40490];assert len(t)==1
l=[r for r in j['task_launches'] if r['connectionId']==t[0]['connectionId'] and r['globalTid']==g['globalTid']];assert len(l)==1;l=l[0]
assert l['connectionId']!=g['connectionId'] and g['startOffsetNs']<=l['startOffsetNs']<l['endOffsetNs']<=g['endOffsetNs']
other=[r for r in j['CANN_API'] if r['globalTid']==g['globalTid'] and r is not g and r['startOffsetNs']<g['endOffsetNs'] and r['endOffsetNs']>0]
assert len(other)==0
print('launch to host API end us:',(g['endOffsetNs']-l['endOffsetNs'])/1000)
print('window start to API end us:',g['endOffsetNs']/1000)
print('other same-thread CANN_API within window:',len(other))
print('No statement about omitted events, pre-window coverage, internal CPU state or causality.')
