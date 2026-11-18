python3 - <<'PY'
from pathlib import Path
import datetime,hashlib,json,socket,sqlite3
assert socket.gethostname()=='<ROLE_HOSTNAME>';root=Path('/PRIVATE_TASK_ROOT');r=root/'vllm_pd_g727';reports=[]
for rank in range(8):
 o=r/'offline_rank0_g809' if rank==0 else r/'offline_remaining_g816'/f'rank{rank}'
 p=o/'result.json'
 if not p.exists():continue
 result=json.loads(p.read_text())
 if result['returncode'] or not result['original_raw_SHA256_unchanged'] or not result['op_statistic_present']:continue
 output=Path(result['output']);db=output/f'ascend_pytorch_profiler_{rank}.db';assert db.is_file();con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
 annotations=[{'start_ns':int(a),'end_ns':int(b),'global_tid':tid,'logical':json.loads(name.removeprefix('ds41.scheduler:'))} for a,b,tid,name in con.execute("SELECT p.startNs,p.endNs,p.globalTid,s.value FROM PYTORCH_API p JOIN STRING_IDS s ON p.name=s.id WHERE s.value LIKE 'ds41.scheduler:%' ORDER BY CAST(p.startNs AS INTEGER)")]
 assert len(annotations)==198 and all(row['logical']['rank']==rank for row in annotations)
 assert sum(row['logical']['scheduled_query_tokens'] for row in annotations)==524272
 devices=[{'device_id':device,'model_id':model,'rows':count,'first_start_ns':first,'last_end_ns':last,'sum_duration_ns_not_walltime':total} for device,model,count,first,last,total in con.execute('SELECT deviceId,modelId,COUNT(*),MIN(startNs),MAX(endNs),SUM(endNs-startNs) FROM TASK GROUP BY deviceId,modelId')]
 assert devices and all(item['device_id']==rank for item in devices)
 compute=con.execute('SELECT COUNT(*) FROM TASK t JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId').fetchone()[0];assert compute>0
 invalid=con.execute('SELECT COUNT(*) FROM TASK WHERE endNs<startNs').fetchone()[0];assert invalid==0
 spans=list(con.execute('SELECT startNs,endNs FROM TASK ORDER BY startNs'));busy=0;end=None
 for start,stop in spans:
  if end is None or start>end:busy+=stop-start;end=stop
  elif stop>end:busy+=stop-end;end=stop
 con.close();report={'rank':rank,'parsed_CPU_annotations':len(annotations),'scheduled_query_tokens':524272,'device_TASK':devices,'compute_task_rows':compute,'all_tasks_busy_union_ns_including_WAIT':busy,'scope':'ActualparsedSQLite deviceTASK/compute rows andlogicalCPUrange coverage; busyunionincludesWAIT andwholecapturehousekeeping, notcomputeutilization or matchedstepcriticalpath. No crosshostclocksubtraction.'};reports.append(report)
 resultpath=o/'parent_device_coverage_g824.json';assert not resultpath.exists();resultpath.write_text(json.dumps(report,indent=2)+'\n')
summary={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ready_P_ranks_reviewed':[item['rank'] for item in reports],'reports':reports,'all8_P_parsed_device_and_CPU_coverage_reviewed':len(reports)==8,'D_all8_accepted':False,'whole_goal_complete':False};path=r/'P_ready_device_review_g824.json';assert not path.exists();path.write_text(json.dumps(summary,indent=2)+'\n');print('G824_READY_P_DEVICE_REVIEW_JSON '+json.dumps(summary),flush=True)
PY
