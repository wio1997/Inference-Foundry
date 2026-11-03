#!/usr/bin/env python3
"""Join six existing EP8 tasks to existing graph parent candidates locally."""
import hashlib,json
from pathlib import Path
T=Path(__file__).resolve().parents[2];tp=T/'agents/diagnosis/engram-matmul-task-links.json';cp=T/'model-execute-wait-chain-v2/calls.json'
targets=json.loads(tp.read_text())['samples'];chains=json.loads(cp.read_text())
rank='dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt';rows=[]
for s in targets:
 task=s['exact_task'];t=s['target'];hits=[]
 for c in chains:
  if c['rank']!=rank or c['model_id']!=task['modelId'] or not c['one_execute_one_wait']:continue
  for w in c['tasks']:
   if w[5]=='MODEL_WAIT_COMPLETE' and w[2]==task['modelId'] and int(w[0])<=int(task['startNs']) and int(task['endNs'])<=int(w[1]):
    assert len(c['cann_execute_api'])==1
    hits.append({'ordinal':c['ordinal'],'parent_connection_id':c['connection_id'],
       'parent_task_name':w[5],'parent_model_id':w[2], 'parent_start_ns':w[0],'parent_end_ns':w[1],
       'parent_api_name':c['cann_execute_api'][0][2]})
 if len(hits)!=1:raise RuntimeError(f'nonunique parent candidate {t} count={len(hits)}')
 rows.append({'run':t['run'],'after_layer':t['after_layer'],'task_global_id':task['globalTaskId'],
              'task_connection_id':task['connectionId'],'parent_candidate':hits[0]})
assert len(rows)==6 and {(r['run'],r['after_layer']) for r in rows}=={(r,l) for r in (1,2,3) for l in (0,13)}
out={'scope':'EP8 six selected tasks matched to same-rank MODEL_WAIT_COMPLETE intervals, graph API by existing exact parent connection',
     'task_links_sha256':hashlib.sha256(tp.read_bytes()).hexdigest(),'graph_calls_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
     'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,
     'limitations':['Enclosing same-model task is a graph-parent candidate, not tensor or source function identity.',
       'Parent and child have different connection IDs; exact CANN join is at parent graph level, not child task level.',
       'MODEL_WAIT_COMPLETE is a profiler label; duration is not exclusively waiting.',
       'No host/device timestamp difference or cross-rank clock assumption is used.']}
dst=T/'agents/diagnosis/engram-graph-enclosure.json';assert not dst.exists();dst.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'COMPLETED','output':str(dst),'rows':len(rows)}))
