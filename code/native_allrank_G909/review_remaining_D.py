from pathlib import Path
import datetime,hashlib,json,os,socket,sqlite3,time
assert socket.gethostname()=='<ROLE_HOSTNAME>';root=Path('/PRIVATE_TASK_ROOT');cpu=root/'native_profile_CPU_g838';out=cpu/'parent_remaining_D_ranks_g905';out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(range(172,184)));os.environ['TMPDIR']=str(out);reports=[];began=time.monotonic()
for rank in range(3,8):
 run=cpu/f'jobs_cli_g853/rank{rank}';p=run/'result.json'

 while not p.exists():
  job=Path('/proc','2488681','cmdline');batch=Path('/proc','2403460','cmdline');assert (job.exists() and b'parse_D_rank.py' in job.read_bytes()) or (batch.exists() and b'parse_remaining_D_g866.py' in batch.read_bytes()),'Noactualparsinghandleandmissingrankresult'
  assert time.monotonic()-began<600,'Parentreviewbound';time.sleep(10)
 result=json.loads(p.read_text());state=json.loads((run/'state.json').read_text());assert state['phase']=='rank_complete_pending_parent_review' and result['returncode']==0 and result['original_raw_SHA256_unchanged'] and not result['NPU_devices']
 db=Path(result['database']);assert db.is_relative_to(cpu/'jobs_cli_g853') and db.is_file();con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON');con.execute('PRAGMA temp_store=MEMORY');con.set_progress_handler(lambda:int(time.monotonic()-began>600),100000)
 rows=[json.loads(name.split(':',1)[1]) for (name,) in con.execute("SELECT s.value FROM PYTORCH_API p JOIN STRING_IDS s ON p.name=s.id WHERE s.value LIKE 'ds41.scheduler:%'")];assert len(rows)==913 and all(x['rank']==rank for x in rows);q=sum(x['scheduled_query_tokens'] for x in rows);assert q==21450
 norm=[dict(row,rank=0) for row in sorted(rows,key=lambda x:x['iteration'])];digest=hashlib.sha256(json.dumps(norm,sort_keys=True).encode()).hexdigest();assert digest=='3fb57d03ce648d8bba525f00b93dedcb93e52115756f90e2dfccc7f595a78f7d'
 groups=list(con.execute('SELECT deviceId,modelId,COUNT(*) FROM TASK GROUP BY deviceId,modelId'));assert all(row[0]==rank for row in groups);assert sum(row[2] for row in groups)==result['NPU_TASK_rows']
 n=con.execute('SELECT COUNT(*) FROM TASK t JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId').fetchone()[0];assert n==result['compute_task_rows'] and n>0
 g=con.execute("SELECT COUNT(*) FROM CANN_API a JOIN STRING_IDS s ON a.name=s.id WHERE s.value='aclmdlRIExecuteAsync'").fetchone()[0];assert g==1812
 assert con.execute('SELECT COUNT(*) FROM TASK WHERE endNs<startNs').fetchone()[0]==0;con.close()
 reports.append({'rank':rank,'CPU_annotations':len(rows),'scheduled_query_tokens':q,'normalized_logical_sequence_SHA256':digest,'device_model_counts':groups,'compute_task_rows':n,'RI_execute_calls':g,'database_bytes':db.stat().st_size,'worker_result_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'original_raw_SHA256_unchanged':True,'scope':'Parentactualread-onlyDBcoveragecheck; fulloriginalrank. Notgain/allrankacceptanceuntilall8present.'})
prior=json.loads((cpu/'parent_ready_ranks_g888/result.json').read_text());assert prior['ranks_parent_reviewed']==[0,1,2];reports=prior['reports']+reports
summary={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ranks_parent_reviewed':[r['rank'] for r in reports],'reports':reports,'D_all8_original_device_and_CPU_coverage_reviewed':len(reports)==8,'scope':'Fulloriginalcaptured8rankDBparentreview. Per-rankTASKcountsnotrequiredidentical; rawunchangedandactualdevicerank/913CPU/21450Q/1812RIchecked. NotcrossframeworkmatchedprofileorE2Egain.','elapsed_s':time.monotonic()-began,'whole_goal_complete':False};(out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print('G905_NATIVE_ALL8_D_PARENT_REVIEW_JSON '+json.dumps(summary),flush=True)