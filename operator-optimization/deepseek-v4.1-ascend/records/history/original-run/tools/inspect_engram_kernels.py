"""Read-only existing 16-rank SQLite: exact Engram kernel names, no NPU."""
from pathlib import Path
from collections import defaultdict,Counter
import json,sqlite3,statistics
R=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
N=('_engram_req_index_kernel','_write_hash_cache_kernel','_hash_ids_kernel','_engram_host_uva_gather_dequant_kernel')
M=('aiv_vec_ratio','aiv_mte2_ratio','aiv_mte3_ratio','aiv_scalar_ratio','aiv_total_cycles','aic_total_cycles')
by=defaultdict(lambda:defaultdict(list));ranks=[]
for d in sorted(R.glob('*_ascend_pt')):
 dbs=list((d/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'));assert len(dbs)==1
 c=sqlite3.connect('file:'+str(dbs[0])+'?mode=ro',uri=True);c.execute('pragma query_only=ON');c.execute('pragma temp_store=MEMORY')
 q='select s.value,t.startNs,t.endNs,t.globalTaskId,t.connectionId,t.streamId,t.modelId,a.name from COMPUTE_TASK_INFO i join STRING_IDS s on s.id=i.name join TASK t on t.globalTaskId=i.globalTaskId left join CANN_API a on a.connectionId=t.connectionId where s.value in (?,?,?,?)'
 rows=c.execute(q,N).fetchall();mp={row[3]:row[0] for row in rows};assert len(mp)==len(rows)
 mids={i:v for i,v in c.execute('select id,value from STRING_IDS where value in (?,?,?,?,?,?)',M)}
 launchids={x[0] for x in c.execute('select id from STRING_IDS where value=?',('launch',))};assert launchids
 counts=Counter()
 for name,start,end,gid,cid,stream,model,api in rows:
  assert end>=start and api in launchids;counts[name]+=1;by[name]['duration_us'].append((end-start)/1000)
 pmu_seen=defaultdict(set)
 for gid,mid,val in c.execute('select globalTaskId,name,value from TASK_PMU_INFO'):
  if gid in mp and mid in mids and val is not None:by[mp[gid]][mids[mid]].append(float(val));pmu_seen[mp[gid]].add(gid)
 for name in N:assert counts[name]==len(pmu_seen[name]),(d.name,name,counts[name],len(pmu_seen[name]))
 ranks.append({'rank':d.name,'database_path':str(dbs[0]),'counts':dict(counts),'all_tasks_unique_launch_and_pmu':True});c.close()
assert len(ranks)==16
stats={name:{metric:{'n':len(v),'median':statistics.median(v),'min':min(v),'max':max(v)} for metric,v in values.items()} for name,values in by.items()}
o={'status':'EXACT_NAMED_ENGRAM_TASKS_NOT_FULL_MODULE','ranks':ranks,'all_rank_event_statistics':stats,'scope':'existing profiler window, exact task names, mixed shape and phase; no end-to-end critical path or full-module ranking', 'caveats':['TASK->CANN launch via connectionId; lookup/hash task names do not attribute TP gather, events, mask or all consumers.','Different kernel invocations and ranks overlap; do not sum as end-to-end elapsed time.','Profiler ratio fields report per-pipeline activity; not whole NPU utilization, full peak efficiency or wait percentage.','Shapes and phases mixed, distributions do not establish baseline service latency.'],'no_device_work':True}
p=T/'engram-kernel-profile';p.mkdir(exist_ok=False);(p/'summary.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps({'ranks':len(ranks),'counts':{k:v['duration_us']['n'] for k,v in stats.items()},'median_us':{k:v['duration_us']['median'] for k,v in stats.items()}},indent=2))
