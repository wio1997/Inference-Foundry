#!/usr/bin/env python3
"""Compare saved same-rank replay-wait envelope to layer0, as recorded only."""
import csv, json, pathlib, statistics, hashlib
T=pathlib.Path(__file__).resolve().parents[2]
files=['model-execute-wait-chain-v2/calls.json','mc2-preceding-analysis/rank_layer0_timeline.csv']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
expected=['ecb1ba603e52cf3cf6722f4b8cb0c1aad3b295c596f4f60c7fc047ecff6ef925', 'd3e0d9cbeddcde97df34dcf36b386cc8201b71c82dde8ab4ed42dd713fde7dfb']
if [sha(T/f) for f in files]!=expected: raise RuntimeError('source hash mismatch')
calls=json.loads((T/files[0]).read_text()); rows=list(csv.DictReader((T/files[1]).open())); index={}
for c in calls:
 k=(c['rank'],int(c['ordinal']))
 if k in index:raise RuntimeError('duplicate call')
 index[k]=c
out={}
for r in rows:
 if r['interior_runs_1_2_3']!='True':continue
 k=(r['rank'],int(r['target_forward_ordinal'])); c=index[k]
 wait=[t for t in c['tasks'] if t[5]=='MODEL_WAIT_COMPLETE']
 if len(wait)!=1 or not c['one_execute_one_wait'] or not c['source_validated_layer_within_linked_wait']:raise RuntimeError('invalid graph link')
 start=int(r['layer_start_ns']); w=wait[0]
 if not (w[0]<=start<w[1]):raise RuntimeError('layer not enclosed')
 out.setdefault((k[1],r['dp']),[]).append((start-w[0])/1000)
for key,values in sorted(out.items()):
 if len(values)!=8:raise RuntimeError('incomplete group')
 print(key, 'count',len(values),'min',round(min(values),3),'median',round(statistics.median(values),3),'max',round(max(values),3))
print('input SHA256:',*expected)
