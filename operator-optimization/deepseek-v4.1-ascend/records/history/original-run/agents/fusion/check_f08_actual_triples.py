#!/usr/bin/env python3
"""Read-only: match selected, task-ID-adjacent F08-compatible triples in stored scope export."""
import json, hashlib, collections, pathlib
from decimal import Decimal
p=pathlib.Path('diagnostic-20261029/target-layer-all-kernels-16rank.jsonl')
sha=hashlib.sha256(p.read_bytes()).hexdigest()
result=[]; scopes=0
for line in p.open():
 d=json.loads(line)
 if d.get('kind')!='rank': continue
 for layer in d['data']['layers']:
  scopes+=1
  task_index=collections.defaultdict(list)
  for g in layer['operator_groups']:
   for t in g['tasks']:
    c=t['csv']; task_index[(c['Stream ID'], c['Task ID'])].append((g['operator_name'],t))
  for (stream, tid), entries in task_index.items():
   if not tid.isdigit() or len(entries)!=1 or entries[0][0]!='RmsNorm':continue
   a=entries[0][1]; ca=a['csv']; sh=ca['Input Shapes'].strip('"').split(';')
   dim=sh[0].split(',')
   if len(dim)!=2 or dim[1]!='512' or len(sh)<2 or sh[1]!='512':continue
   n=int(tid); pairs=[task_index.get((stream,str(n+i)),[]) for i in (1,2)]
   if any(len(x)!=1 for x in pairs):continue
   (nb,b),(nc,c)=pairs[0][0],pairs[1][0]; cb=b['csv'];cc=c['csv']
   if nb!='InplacePartialRotaryMul' or 'ScatterNdUpdateSk' not in nc:continue
   rb=cb['Input Shapes'].strip('"').split(';');rc=cc['Input Shapes'].strip('"').split(';');t=dim[0]
   if not (len(rb)>=3 and rb[:3]==[f'{t},1,1,512',f'{t},1,1,64',f'{t},1,1,64'] and len(rc)>=3 and rc[0].endswith(',128,512') and rc[1:3]==[f'{t},2',f'{t},512']):continue
   if not (Decimal(a['device_start_us']) < Decimal(b['device_start_us']) < Decimal(c['device_start_us'])):continue
   if not (ca['Output Shapes'].strip('"').split(';')[0]==f'{t},512' and cb['Output Shapes'].strip('"')==f'{t},1,1,512'):continue
   if not (ca['Input Data Types']=='DT_BF16;DT_BF16' and ca['Output Data Types']=='DT_BF16;FLOAT' and cb['Input Data Types']=='DT_BF16;FLOAT;FLOAT' and cb['Output Data Types']=='DT_BF16' and cc['Input Data Types']=='DT_BF16;INT32;DT_BF16' and cc['Output Data Types']=='DT_BF16'):continue
   result.append({'ep':d['data']['ep'],'layer':layer['layer'],'digest':layer['scheduler_digest'],'scope_id':layer['layer_scope_id'],'stream':stream,'tasks':[tid,str(n+1),str(n+2)],'rows':int(t),'shapes':[ca['Input Shapes'],cb['Input Shapes'],cc['Input Shapes']],'starts_us':[a['device_start_us'],b['device_start_us'],c['device_start_us']],'flow_ids':[a['flow_id'],b['flow_id'],c['flow_id']]})
summary={'source':str(p),'source_sha256':sha,'scope_count':scopes,'triples':len(result),'unique_scopes':len({(r['ep'],r['scope_id']) for r in result}),'eps':sorted(set(r['ep'] for r in result)),'layers':sorted(set(r['layer'] for r in result)),'digest_counts':dict(collections.Counter(r['digest'] for r in result)),'count_by_ep':dict(sorted(collections.Counter(str(r['ep']) for r in result).items(),key=lambda a:int(a[0]))),'rows_counts':dict(collections.Counter(str(r['rows']) for r in result)),'records':result,'limitations':['Unique selected task ID in one scope and task adjacency do not prove data identity, runtime callsite, cache content, causal dependence or omitted task absence.','Shape match to source is conditional on branch, implementation and graph capture.','Recorded time order does not establish duration saved or critical path.','No device workload or target implementation edits.']}
out=pathlib.Path('agents/fusion/f08-actual-triples.json');out.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2));print('example',next((x for x in result if x['ep']==8 and x['layer']==14),None))
