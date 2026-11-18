"""Parse one full captured D rank in the SDK container without NPU devices."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,socket,sqlite3,subprocess,time
ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,choices=range(0,8),required=True);ap.add_argument('--cpu-set',choices=['160-171','172-183'],required=True);a=ap.parse_args();rank=a.rank
assert socket.gethostname()=='<ROLE_HOSTNAME>'
cores=[int(x) for x in a.cpu_set.split('-')];os.sched_setaffinity(0,set(range(cores[0],cores[1]+1)))
ROOT=Path('/PRIVATE_TASK_ROOT');CPU=ROOT/'native_profile_CPU_g838';ORIGINAL=ROOT/'vllm_pd_g727';OUT=CPU/'jobs_cli_g853'/f'rank{rank}';OUT.mkdir(parents=True,exist_ok=False);began=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1048576),b''):h.update(part)
 return h.hexdigest()
def save(name,value):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def state(phase,**more):
 d={'phase':phase,'pid':os.getpid(),'rank':rank,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-began,**more};save('state.json',d);print(json.dumps(d),flush=True)
try:
 lease=json.loads((CPU/'controller_state.json').read_text());assert lease['phase']=='CPU_SDK_ready_no_NPU_devices';p=Path('/proc',str(lease['pid']),'cmdline');assert p.exists() and b'cpu_parser_controller.py' in p.read_bytes()
 receipt=json.loads((CPU/'create_receipt.json').read_text());cid=receipt['container_id'];c=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert c['State']['Running'] and c['Image']==receipt['image_id'] and not c['HostConfig']['Devices'] and not c['HostConfig']['Privileged']
 source=next(p for p in (ORIGINAL/'profiles').glob('*_ascend_pt') if re.search(r'_rank'+str(rank)+r'_',p.name));expected=json.loads((ORIGINAL/'raw_logical_work_g812'/f'rank{rank}_summary.json').read_text());assert expected['annotations']==913 and expected['total_scheduled_query_tokens']==21450
 state('freezing_full_original_raw',source=str(source),CPU_affinity=a.cpu_set,CPU_only=True,NPU_devices=[])
 frozen={str(p.relative_to(source)):{'sha256':sha(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in source.rglob('*') if p.is_file()};save('original_raw_SHA256.json',frozen)
 copied=OUT/source.name;subprocess.run(['cp','-a','--reflink=auto',str(source),str(copied)],cwd=CPU,check=True,timeout=180)
 for name,item in frozen.items():assert sha(copied/name)==item['sha256']
 (OUT/'tmp').mkdir();inside='/task/native_profile_CPU_g838/jobs_cli_g853/rank'+str(rank)+'/'+source.name
 adapter='/task/native_profile_CPU_g838/adapter_g853'
 argv=['docker','exec','-e',f'TMPDIR=/task/native_profile_CPU_g838/jobs_cli_g853/rank{rank}/tmp','-e','DS41_CPU_CANN_ADAPTER=g853','-e',f'DS41_CPU_CLI_LOG_DIR=/task/native_profile_CPU_g838/jobs_cli_g853/rank{rank}','-e','PYTHONPATH='+adapter,cid,'taskset','-c',a.cpu_set,'timeout','--signal=TERM','--kill-after=30','2400','python3','-u',adapter+'/run_sdk_cpu.py',inside]
 with (OUT/'SDK.log').open('x') as stream:
  p=subprocess.Popen(argv,cwd=CPU,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);save('actual_parse_handle.json',{'pid':p.pid,'argv':argv,'CPU_container_id':cid,'inner_deadline_s':2400,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()});state('CPU_only_CANN_Python_backend_DB_parsing',docker_exec_pid=p.pid);returncode=p.wait(timeout=2460)
 unchanged=all(sha(source/name)==item['sha256'] and (source/name).stat().st_size==item['bytes'] and (source/name).stat().st_mtime_ns==item['mtime_ns'] for name,item in frozen.items());assert unchanged
 db=copied/'ASCEND_PROFILER_OUTPUT'/f'ascend_pytorch_profiler_{rank}.db';assert returncode==0 and db.is_file(),('OfflineSDKfailed',returncode)
 con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);rows=[json.loads(name.removeprefix('ds41.scheduler:')) for (name,) in con.execute("SELECT s.value FROM PYTORCH_API p JOIN STRING_IDS s ON p.name=s.id WHERE s.value LIKE 'ds41.scheduler:%'")];assert len(rows)==913 and all(item['rank']==rank for item in rows);assert sum(item['scheduled_query_tokens'] for item in rows)==21450
 normalized=[dict(item,rank=0) for item in sorted(rows,key=lambda item:item['iteration'])];assert hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest()==expected['normalized_logical_sequence_sha256']
 task_rows=con.execute('SELECT COUNT(*) FROM TASK').fetchone()[0];compute_rows=con.execute('SELECT COUNT(*) FROM TASK t JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId').fetchone()[0];assert task_rows>0 and compute_rows>0
 if rank==0:assert task_rows==8081676 and compute_rows==2358816,('CPU_native_D0_full_coverage_mismatch',task_rows,compute_rows)
 groups=list(con.execute('SELECT deviceId,modelId,COUNT(*) FROM TASK GROUP BY deviceId,modelId'));assert all(row[0]==rank for row in groups)
 graph_calls=con.execute("SELECT COUNT(*) FROM CANN_API a JOIN STRING_IDS s ON a.name=s.id WHERE s.value='aclmdlRIExecuteAsync'").fetchone()[0];assert graph_calls==1812;con.close()
 report={'rank':rank,'returncode':returncode,'original_raw_SHA256_unchanged':unchanged,'CPU_container_id':cid,'CLI_adapter':'g853','NPU_devices':[],'database':str(db),'database_bytes':db.stat().st_size,'CPU_annotations':len(rows),'scheduled_query_tokens':21450,'normalized_logical_sequence_sha256':expected['normalized_logical_sequence_sha256'],'NPU_TASK_rows':task_rows,'compute_task_rows':compute_rows,'device_model_counts':groups,'RI_graph_calls':graph_calls,'scope':'Fulloriginalcapturedrank, offlineCPUonly; notNPUrecapture/notoptimizationgain. Allrankacceptanceandgraphfamily/workmapremainparentreview.'};save('result.json',report);state('rank_complete_pending_parent_review',**report)
except BaseException as error:
 state('failed_requires_review',error=repr(error));raise
