import pathlib,json,socket,hashlib,subprocess,os,time,fcntl
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
r=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030');controller=r/'moe_observer_controller166.py'
RUN='observer-20261030T154429Z-4d760411'
SHA='ebc9946fe651684005f5ebf44178b9bdf4d4e1381b7e4b5dd4662fd96fe9881d'
pre=json.loads((r/'controller-preflight.json').read_text());assert pre['status']=='PASS_CONTROLLER_CPU_AND_LAZY_IMPORT_PREFLIGHT' and pre['controller_sha256']==SHA;assert hashlib.sha256(controller.read_bytes()).hexdigest()==SHA
lock=open(r/'.device-diagnostic.lock','a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
assert not (r/RUN).exists();latest=r/'moe-observer-latest.json';assert not latest.exists(),'Existing observer run; inspect rather than duplicate'
logs=r/'launches';logs.mkdir(exist_ok=True);env=os.environ.copy();env['WIO_RUN_ID']=RUN
with (logs/(RUN+'.stdout')).open('xb') as out,(logs/(RUN+'.stderr')).open('xb') as err:child=subprocess.Popen(['python3',str(controller)],env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'run_id':RUN,'controller_pid':child.pid,'controller_sha256':SHA,'controller':str(controller),'run_root':str(r/RUN),'stdout':str(logs/(RUN+'.stdout')),'stderr':str(logs/(RUN+'.stderr')),'state':'LAUNCHED_PENDING_CONTROLLER_PREFLIGHT','model_capture_values_not_yet_available':True,'overall_goal_complete':False,'started_unix':time.time()};(logs/(RUN+'.json')).write_text(json.dumps(record,indent=2)+'\n');latest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
