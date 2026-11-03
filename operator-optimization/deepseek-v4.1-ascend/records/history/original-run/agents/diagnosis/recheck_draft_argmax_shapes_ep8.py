#!/usr/bin/env python3
"""Cross-check saved stage243 rows against stage242 exact event anchors."""
import collections,hashlib,json
from pathlib import Path
T=Path(__file__).resolve().parents[2]
def read(p,sha):
 b=(T/p).read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
x=read('agents/diagnosis/draft-argmax-intervals-shapes-ep8.json','59930fe7433cb31708f906713b8f24685dd307cb43ad1ab7a0b2d35310766dfe')
a=read('agents/diagnosis/draft-argmax-between-ep8.json','a20fd5761c9ca88b5161ea624c267c248342a21d230bb69a7f83ee589fdc0666')
c=read('evidence/model-config.json','43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7')
assert x['rank']==a['rank'] and x['csv_sha256']==a['csv_sha256'] and x['anchors_sha256']=='a20fd5761c9ca88b5161ea624c267c248342a21d230bb69a7f83ee589fdc0666'
expected={}
for gap in a['intervals']:
 for e in gap['events']:
  key=(gap['ordinal'],gap['gap_index'],gap['start_ns']+e['start_rel_ns'],gap['start_ns']+e['end_rel_ns'],e['type'],e['task_id'])
  assert key not in expected;expected[key]=(e['model'],e['stream'])
assert len(expected)==80
sigs=collections.defaultdict(collections.Counter);seen=set();missing=[]
for r in x['rows']:
 assert r['matches']==1 and r['match'] is not None
 item=r['match'];t=item['anchor'];f=item['fields'];key=(t['ordinal'],t['gap_index'],t['start_ns'],t['end_ns'],r['type'],f['Task ID'].strip())
 assert key in expected and key not in seen and (f['Model ID'].strip(),f['Stream ID'].strip())==expected[key] and f['Type']==r['type'];seen.add(key)
 missing.extend(item['missing_columns'])
 fields=('Name','Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Accelerator Core')
 sigs[r['type']][tuple(f.get(k) for k in fields)]+=1
assert seen==set(expected) and len(x['rows'])==80
assert c['text_config']['vocab_size']==129280 and c['text_config']['dspark_markov_rank']==256
print(json.dumps({'scope':'80 exact anchored EP8 events, only exported metadata and config',
 'rows':len(seen),'missing_columns':sorted(set(missing)),'grouped_signatures':{k:[{'fields':dict(zip(fields,s)),'count':n} for s,n in v.items()] for k,v in sorted(sigs.items())},
 'text_config':{'vocab_size':129280,'dspark_markov_rank':256},
 'arithmetic':{'5_vocab':5*129280,'4_times_5_vocab':4*5*129280,'4_times_6':4*6},
 'limits':['Names and compatible shapes do not bind source call, tensor identity, token value, effective row count, request or dependencies.',
 'Configuration does not prove historical runtime object or same capture branch.',
 'Input metadata not actual contents; no NPU or new profiler.']},ensure_ascii=False,indent=2))
