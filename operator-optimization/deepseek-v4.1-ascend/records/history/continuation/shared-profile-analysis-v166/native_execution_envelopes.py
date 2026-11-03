"""NPU-only graph envelopes,keeping the absent Python graph binding explicit."""
import collections,concurrent.futures,json,sqlite3
from pathlib import Path
H=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/npu166-official-db')
def one(db):
 c=sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
 strings=dict(c.execute('select id,value from STRING_IDS'));s=lambda i:strings.get(i,str(i))
 tasks=[dict(r) for r in c.execute('select rowid,* from TASK')];info={r['globalTaskId']:dict(r) for r in c.execute('select * from COMPUTE_TASK_INFO')}
 launches=collections.defaultdict(list);waits=collections.defaultdict(list)
 for t in tasks:
  key=tuple(t[k] for k in ['connectionId','deviceId','globalPid','streamId','modelId'])
  if s(t['taskType'])=='MODEL_EXECUTE':launches[key].append(t)
  elif s(t['taskType'])=='MODEL_WAIT_COMPLETE':waits[key].append(t)
 env=[];issues=[]
 for key,ls in launches.items():
  ws=waits.get(key,[])
  if key[0]<0 or key[-1]==4294967295 or len(ls)!=1 or len(ws)!=1:
   issues.append({'key':key,'launch_count':len(ls),'wait_count':len(ws)});continue
  a=ls[0];b=ws[0]
  if a['startNs']>b['endNs']:issues.append({'key':key,'reason':'INVALID_ENVELOPE'});continue
  env.append({'native_connection_id':key[0],'device':key[1],'native_pid':key[2],'stream':key[3],'native_model_id':key[4],
      'start_ns':a['startNs'],'end_ns':b['endNs'],'duration_ns':b['endNs']-a['startNs'],
      'launch_rowid':a['rowid'],'completion_rowid':b['rowid'],'operators':{},'captured_compute_tasks':0,
      'Python_graph_identity':'NOT_OBSERVED_IN_THIS_NPU_ONLY_COHORT'})
 assigned=collections.Counter()
 for t in tasks:
  x=info.get(t['globalTaskId'])
  if not x or t['modelId']==4294967295:continue
  ms=[e for e in env if e['native_model_id']==t['modelId'] and e['device']==t['deviceId'] and e['native_pid']==t['globalPid'] and e['start_ns']<=t['startNs']<=t['endNs']<=e['end_ns']]
  assigned['unique' if len(ms)==1 else 'missing' if not ms else 'ambiguous']+=1
  if len(ms)!=1:continue
  e=ms[0];e['captured_compute_tasks']+=1;name=s(x['name']);g=e['operators'].setdefault(name,{'count':0,'sum_ns':0});g['count']+=1;g['sum_ns']+=t['endNs']-t['startNs']
 c.close();return {'profile':db.parent.parent.name,'envelopes':env,'issues':issues,'captured_compute_assignment':dict(assigned)}
if __name__=='__main__':
 dbs=sorted((H/'profiles').glob('*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db'));assert len(dbs)==32
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,dbs))
 counts=collections.Counter()
 for r in rows:counts.update(r['captured_compute_assignment'])
 out={'status':'NPU_ONLY_NATIVE_MODEL_COMPLETION_ENVELOPES','profiles':rows,'envelope_count':sum(len(r['envelopes']) for r in rows),'assignment':dict(counts),
      'contract':'Exact same native connection ID plus model/device/pid/stream binds launch/wait;children require unique full containment and same model/device/pid. No nearest-time or cross-cohort ID joins.',
      'limits':['No Python graph object or per-operator CPU source binding is inferred for this cohort.','An execution envelope is not the entire request critical path or all-reader retirement.','Different input validity/receiver work and window extents preclude a causal speedup claim.'],
      'historical_native_or_phase_promotions':0,'whole_modules_closed':0,'overall_complete':False}
 (H/'NATIVE-ENVELOPES-v166.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'status':out['status'],'envelopes':out['envelope_count'],'assignment':dict(counts),'issues':sum(len(r['issues']) for r in rows)}))
