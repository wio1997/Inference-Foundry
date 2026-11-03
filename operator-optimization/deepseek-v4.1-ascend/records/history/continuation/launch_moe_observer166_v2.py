import pathlib,json,socket,subprocess,hashlib,fcntl,os,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
r=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030');controller=r/'moe_observer_controller166_v2.py'
RUN='observer-20261030T160315Z-76b79f0f'
SHA='2494451c2c27ae719104b53bee3469529533d58cdb9674fb02f094f096a8ce5c'
pre=json.loads((r/'controller-preflight-v2.json').read_text());assert pre['status']=='PASS_REVISED_CONTROLLER_CPU_PREFLIGHT' and pre['controller_sha256']==SHA;assert hashlib.sha256(controller.read_bytes()).hexdigest()==SHA
old=json.loads((r/'moe-observer-latest.json').read_text());oldroot=pathlib.Path(old['run_root']);assert json.loads((oldroot/'controller-finished.json').read_text())['state']=='RESTORED_AFTER_ERROR';assert json.loads((oldroot/'restore-result.json').read_text())['status']=='PASS_ORIGINAL_BASELINE_RESTORED'
lock=open(r/'.device-diagnostic.lock','a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close();assert not (r/RUN).exists()
logs=r/'launches';env=os.environ.copy();env['WIO_RUN_ID']=RUN
with (logs/(RUN+'.stdout')).open('xb') as out,(logs/(RUN+'.stderr')).open('xb') as err:child=subprocess.Popen(['python3',str(controller)],env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'run_id':RUN,'controller_pid':child.pid,'controller_sha256':SHA,'controller':str(controller),'run_root':str(r/RUN),'stdout':str(logs/(RUN+'.stdout')),'stderr':str(logs/(RUN+'.stderr')),'previous_failed_run':old['run_id'],'state':'REVISED_RUN_LAUNCHED_PENDING_PREFLIGHT','model_capture_values_not_yet_available':True,'overall_goal_complete':False,'started_unix':time.time()};(logs/(RUN+'.json')).write_text(json.dumps(record,indent=2)+'\n');temp=r/'moe-observer-latest.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(r/'moe-observer-latest.json');print(json.dumps(record))
