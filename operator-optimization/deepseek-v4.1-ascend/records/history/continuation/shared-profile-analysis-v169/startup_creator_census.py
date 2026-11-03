"""Factorized CPU/source census of existing startup profiles;no device work.

Native connections and callchains stored once,not expanded for every execution.
"""
import collections,concurrent.futures,hashlib,json,os,sqlite3,time,traceback
from pathlib import Path
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168')
OUT=P.parent/'joint169'/'startup-creators';OUT.mkdir(exist_ok=True)
def one(db):
 out=OUT/(db.parent.parent.name+'.json')
 if out.exists():return json.loads(out.read_text())['summary']
 c=sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row;s=dict(c.execute('select id,value from STRING_IDS'))
 cpu={r['CPU_ROW']:dict(r) for r in c.execute('select rowid CPU_ROW,* from PYTORCH_API where type=50001 and connectionId is not null')}
 by_set=collections.defaultdict(list)
 for rowid,r in cpu.items():by_set[r['connectionId']].append(rowid)
 by_native=collections.defaultdict(set)
 for pair in c.execute('select id,connectionId from CONNECTION_IDS'):
  if pair[0] in by_set:by_native[pair[1]].update(by_set[pair[0]])
 chains=collections.defaultdict(list)
 for r in c.execute('select * from PYTORCH_CALLCHAINS'):chains[r['id']].append({'depth':r['stackDepth'],'text':s[r['stack']]})
 for rows in chains.values():rows.sort(key=lambda r:r['depth'])
 executions=[];assignment=collections.Counter();names=collections.Counter();used_cpu=set();used_chain=set();total_ns=0
 # The TASK table is small in this creation/warmup domain;one keyed metadata lookup.
 info={r['globalTaskId']:dict(r) for r in c.execute('select * from COMPUTE_TASK_INFO')}
 for r in c.execute('select rowid TASK_ROW,* from TASK'):
  x=info.get(r['globalTaskId'])
  if x is None:continue
  ps=sorted(by_native.get(r['connectionId'],()));assignment['unique' if len(ps)==1 else ('missing' if not ps else 'ambiguous')]+=1;names[s[x['name']]]+=1;total_ns+=r['endNs']-r['startNs'];used_cpu.update(ps)
  executions.append({'TASK_ROW':r['TASK_ROW'],'native_pid':r['globalPid'],'model_id':r['modelId'],'stream_id':r['streamId'],'task_id':r['taskId'],'native_name':s[x['name']],'duration_ns':r['endNs']-r['startNs'],'native_connection':r['connectionId'],'input_shapes':s.get(x['inputShapes']),'output_shapes':s.get(x['outputShapes']),'CPU_rows':ps})
 compact_cpu={}
 for rowid in used_cpu:
  r=cpu[rowid];ch=r['callchainId'];used_chain.add(ch);compact_cpu[rowid]={'name':s[r['name']],'connection_set_id':r['connectionId'],'callchain_id':ch,'global_tid':r['globalTid']}
 frame_counts=collections.Counter()
 for cid in used_chain:
  for frame in chains.get(cid,[]):frame_counts[frame['text']]+=1
 summary={'profile':db.parent.parent.name,'database':str(db),'database_sha256':hashlib.sha256(db.read_bytes()).hexdigest(),'native_compute_executions':len(executions),'CPU_assignment':dict(assignment),'used_CPU_rows':len(used_cpu),'unique_callchains':len(used_chain),'recorded_native_work_ns':total_ns,'native_names':dict(names),'source_frames':dict(frame_counts),
  'scope':'Exactsame-profile nativeconnection in officialCPUconnectionsets;startup dummy/warmup/creation domain,not formalC8 or oldcohort backfill. Nativeworksum is not requestlatency. No all-reader/independence or physicalupper proof.','overall_complete':False}
 out.write_text(json.dumps({'summary':summary,'executions':executions,'CPU_rows':compact_cpu,'callchains':{ch:chains.get(ch,[]) for ch in used_chain}},separators=(',',':'))+'\n');c.close();return summary
if __name__=='__main__':
 state={'state':'RUNNING','pid':os.getpid(),'start_ticks':Path('/proc/'+str(os.getpid())+'/stat').read_text().rsplit(')',1)[1].split()[19],'started_unix':time.time(),'CPU_only':True,'completed':0};(OUT/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
 try:
  dbs=sorted((P/'startup').glob('*_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db'));assert len(dbs)==16
  rows=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for r in pool.map(one,dbs):rows.append(r);state['completed']=len(rows);(OUT/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
  counts=collections.Counter()
  for r in rows:counts.update(r['CPU_assignment'])
  result={'status':'EXACT_STARTUP_CREATOR_CENSUS_RETURNED','profiles':rows,'native_compute_executions':sum(r['native_compute_executions'] for r in rows),'assignment':dict(counts),'historical_native_or_phase_promotions':0,'overall_complete':False};(OUT/'REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');state.update(state='RETURNED',elapsed_seconds=time.time()-state['started_unix']);(OUT/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
 except BaseException:state.update(state='ERROR',error=traceback.format_exc());(OUT/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n');raise
