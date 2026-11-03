"""Aggregate same-domain F21 work/time without per-call eager matching."""
import collections
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis')
O=A/'joint170'
state={'state':'RUNNING','pid':os.getpid(),'start_ticks':Path('/proc/'+str(os.getpid())+'/stat').read_text().rsplit(')',1)[1].split()[19],'started_unix':time.time(),'completed':0,'CPU_only':True}
def status(): (O/'AGGREGATE-STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
def one(p):
 d=json.loads(p.read_text());db=d['database'];c=sqlite3.connect('file:'+db+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row;s=dict(c.execute('select id,value from STRING_IDS'));name=next(k for k,v in s.items() if v=='DispatchFFNCombine')
 info={r['globalTaskId']:(s[r['inputShapes']],s[r['outputShapes']]) for r in c.execute('select globalTaskId,inputShapes,outputShapes from COMPUTE_TASK_INFO where name=?',(name,))}
 sql='select rowid TASK_ROW,globalTaskId,startNs,endNs,streamId,modelId,globalPid from TASK where globalTaskId in (select globalTaskId from COMPUTE_TASK_INFO where name=?)'
 plan=[list(r) for r in c.execute('explain query plan '+sql,(name,))];groups=collections.defaultdict(lambda:{'calls':0,'native_work_ns':0,'streams':set()})
 for t in c.execute(sql,(name,)):
  ins,outs=info[t['globalTaskId']];key=(ins,outs,t['modelId']!=4294967295);r=groups[key];r['calls']+=1;r['native_work_ns']+=int(t['endNs'])-int(t['startNs']);r['streams'].add(t['streamId']);assert t['globalPid']==d['pid']
 c.close()
 return {'pid':d['pid'],'process_uuid':d['process_uuid'],'database_sha256_reused':d['database_sha256'],'query_plan':plan,'groups':[{'input_shapes':k[0],'output_shapes':k[1],'captured':k[2],**{a:sorted(b) if isinstance(b,set) else b for a,b in v.items()}} for k,v in sorted(groups.items())]}
status();paths=sorted((A/'joint169').glob('*-SUMMARY.json'));assert len(paths)==16;rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for r in pool.map(one,paths):rows.append(r);state['completed']=len(rows);status()
r={'status':'F21_SHAPE_AGGREGATES_NO_PER_CALL_EAGER_MATCH','profiles':rows,'calls':sum(g['calls'] for p in rows for g in p['groups']),'native_work_ns':sum(g['native_work_ns'] for p in rows for g in p['groups']),'no_new_device_work':True,'overall_complete':False};f=O/'F21-SHAPE-AGGREGATES.json';f.write_text(json.dumps(r,indent=2)+'\n');state.update(state='RETURNED',elapsed_seconds=time.time()-state['started_unix'],result=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest());status();print(json.dumps({k:v for k,v in state.items()}))
