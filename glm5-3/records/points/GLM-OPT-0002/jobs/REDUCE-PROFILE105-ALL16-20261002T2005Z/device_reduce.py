from pathlib import Path
import sqlite3,json,sys,collections,statistics,hashlib
lo,hi=map(int,sys.argv[1:3]);lo*=1000;hi*=1000
roots=[Path('/data/tiankuan/wio/glm52-pd/deploy/private/profile_run105_export_rank0'),Path('/data/tiankuan/wio/glm52-pd/deploy/private/profile_run105_export_remaining')]
ds=sorted(d for root in roots for d in root.glob('PROF_*'));assert len(ds)==16
def union(iv):
 end=lo;total=0
 for a,b in sorted(iv):
  if b>end:total+=b-max(end,a);end=b
 return total
def quant(v):
 v=sorted(v);return dict(n=len(v),min=v[0]if v else None,p50=statistics.median(v)if v else None,p90=v[min(int(.9*(len(v)-1)),len(v)-1)]if v else None,max=v[-1]if v else None)
out=[]
for d in ds:
 f=next(d.glob('msprof*.db'));db=sqlite3.connect('file:'+str(f)+'?mode=ro&immutable=1',uri=True)
 by=collections.defaultdict(list);task_types=collections.Counter();multiplicity=collections.Counter()
 # globalTaskId identifies captured operation reused over many graph replays.
 # Retain every concrete interval; union removes nested/subtask overlap.
 sql="SELECT t.startNs,t.endNs,t.modelId,s.value,t.globalTaskId FROM TASK t JOIN STRING_IDS s ON s.id=t.taskType WHERE t.endNs>? AND t.startNs<?"
 markers=[];marker_connections=[]
 for a,b,m,typ,gid in db.execute(sql,(lo,hi)):
  task_types[typ]+=1
  if typ=='MODEL_EXECUTE':markers.append(a)
  if typ.startswith('KERNEL_')or typ in ['AI_CORE','MIX_AIC','MIX_AIV','AI_CPU']:by[str(m)].append((max(a,lo),min(b,hi)));multiplicity[gid]+=1
 graph=[x for k,v in by.items()if k!='4294967295'for x in v];eager=by.get('4294967295',[])
 markers=sorted(set(markers));steps=[]
 for a,b in zip(markers,markers[1:]):
  if a<lo or b>hi:continue
  g=union([(max(x,a),min(y,b))for x,y in graph if x<b and y>a]);e=union([(max(x,a),min(y,b))for x,y in eager if x<b and y>a]);allc=union([(max(x,a),min(y,b))for x,y in graph+eager if x<b and y>a])
  steps.append(dict(start_ns=a,end_ns=b,wall_ms=(b-a)/1e6,graph_union_ms=g/1e6,non_graph_union_ms=e/1e6,overlap_ms=(g+e-allc)/1e6,uncovered_ms=(b-a-allc)/1e6))
 api=collections.defaultdict(list)
 sql="SELECT a.startNs,a.endNs,s.value FROM CANN_API a JOIN STRING_IDS s ON s.id=a.name WHERE a.endNs>? AND a.startNs<? AND s.value IN ('aclrtSynchronizeEvent','aclrtSynchronizeStreamWithTimeout','aclmdlRIExecuteAsync')"
 for a,b,n in db.execute(sql,(lo,hi)):api[n].append((max(lo,a),min(hi,b)))
 conn=db.execute("SELECT t.startNs,a.startNs,a.endNs,t.connectionId FROM TASK t JOIN STRING_IDS s ON s.id=t.taskType JOIN CANN_API a ON a.connectionId=t.connectionId JOIN STRING_IDS n ON n.id=a.name WHERE s.value='MODEL_EXECUTE' AND n.value='aclmdlRIExecuteAsync' AND t.startNs>=? AND t.startNs<?",(lo,hi)).fetchall()
 dev=next(d.glob('device_*'));tdb=sqlite3.connect('file:'+str(dev/'sqlite/time.db')+'?mode=ro&immutable=1',uri=True);clock=tdb.execute('SELECT * FROM Time').fetchall();tdb.close()
 # Sum fields below remain reported interval sums; no serial criticalcost claim.
 rawops=db.execute("SELECT s.value,count(*),sum(min(t.endNs,?)-max(t.startNs,?))/1e6 FROM TASK t JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId JOIN STRING_IDS s ON s.id=c.opType WHERE t.endNs>? AND t.startNs<? GROUP BY c.opType ORDER BY sum(min(t.endNs,?)-max(t.startNs,?)) DESC LIMIT 12",(hi,lo,lo,hi,hi,lo)).fetchall()
 info=json.loads((d/'host/info.json').read_text())
 out.append(dict(pid=int(info['pid']),device=dev.name,path=str(d),db=str(f),clock_calibration=clock,task_types=task_types,compute_interval_count=sum(map(len,by.values())),graph_compute_union_ms=union(graph)/1e6,non_graph_compute_union_ms=union(eager)/1e6,all_compute_union_ms=union(graph+eager)/1e6,reported_interval_sum_ms=sum(b-a for v in by.values()for a,b in v)/1e6,graph_model_ids=[k for k in by if k!='4294967295'],step_marker_count=len(markers),steps=steps,step_statistics={k:quant([x[k]for x in steps])for k in ['wall_ms','graph_union_ms','non_graph_union_ms','overlap_ms','uncovered_ms']},host_top_level_api={k:dict(count=len(v),reported_sum_ms=sum(b-a for a,b in v)/1e6,union_ms=union(v)/1e6)for k,v in api.items()},native_graph_launch_connections=dict(matched_rows=len(conn),sample=conn[:3],device_marker_minus_host_api_start_ms=quant([(x[0]-x[1])/1e6 for x in conn])),top_reported_compute_operators=rawops))
 db.close()
print(json.dumps(dict(window_ns=[lo,hi],window_s=(hi-lo)/1e9,profiles=out,profile_count=len(out),limits=['Native host/device clock calibration rows and connectionIDs retained; no crosshost166/client clock alignment or guaranteed nanosecond accuracy.','Graph markers partition observed device intervals, not proven complete semantic decode iterations; concrete graph modelIDs and anonymous tasks must be interpreted with native source.','Task interval union avoids overlap inflation; reported operator sums/subtasks cannot be treated as serialcriticalchain or hardwarelowerbound.','Heavy profiler overhead; no ordinaryE2ETPS/current/KEEP/stablecapacity claim.'])))
