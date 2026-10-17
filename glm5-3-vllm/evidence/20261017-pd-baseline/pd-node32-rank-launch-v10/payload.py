import datetime,hashlib,json,os,pathlib,socket,subprocess,sys
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(a,t=120): return subprocess.run(a,capture_output=True,text=True,timeout=t)
RUN=pathlib.Path(__file__).resolve().parent
plan=json.loads((RUN/'plan.json').read_text())
setup=json.loads((RUN/'setup.json').read_text())
names=('setup.json','plan.json','payload.py','start_capacity.py','manifest.json','context.json','launch.json')
def emit(status,**extra):
 o={'status':status,'run_dir':str(RUN),'model_ready_checked':False,'sha256':{n:hashlib.sha256((RUN/n).read_bytes()).hexdigest() for n in names if (RUN/n).is_file()}}
 o.update(extra); print(json.dumps(o))
ctx={'host_checked_utc':utc(),'scope':'launch only: preconditions asserted, one container started; model readiness is not checked',
 'client':{k:setup[k] for k in ('client_started_utc','source_script','source_script_sha256','plan_sha256','payload_sha256','git_sha','git_diff_sha256','lease')}}
launch=None
try:
 expected_host=plan.get('target_hostname','B5-5U-Atlas900RCKA3-33')
 assert expected_host in ('B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33')
 assert os.uname().nodename==expected_host,'Wrong host identity'
 assert plan['serve_argv'][:2]==['vllm','serve'],'serve argv does not start with vllm serve'
 tr=pathlib.Path(plan['remote']['transfer_root']); wm=plan['weights_manifest']
 receipts=json.loads((tr/wm['receipt']).read_text()); ready=json.loads((tr/wm['ready']).read_text())
 ident=hashlib.sha256(json.dumps(receipts,sort_keys=True).encode()).hexdigest()
 assert ident==wm['manifest_sha256']==ready['manifest_sha256'],'weights manifest identity mismatch: '+ident
 weights=pathlib.Path(plan['remote']['weights_dir']); bad=[]; total=0
 assert ready['all_file_payload_sha256_verified'] is True and ready['files']==196 and ready['bytes']==764240832702
 assert weights.is_dir() and '.incoming' not in str(weights.resolve()),'Incomplete weights path'
 for name,exp in receipts.items():
  assert not name.startswith('/') and '..' not in pathlib.PurePosixPath(name).parts,'unsafe receipt path '+name
  p=weights/name; total+=exp['bytes']
  if not p.is_file() or p.stat().st_size!=exp['bytes']: bad.append(name)
 assert not bad and len(receipts)==wm['file_count'] and total==wm['total_bytes'],'weights stat mismatch: %d bad %s'%(len(bad),bad[:5])
 ctx['weights']={'manifest_sha256':ident,'files':len(receipts),'total_bytes':total,'size_mismatch':bad,'payload_rehash':False,'weights_dir':str(weights),'transfer_root':str(tr)}
 ins=run(['docker','image','inspect',plan['image']['ref'],'--format','{{.Id}}']); image_id=ins.stdout.strip()
 assert ins.returncode==0 and image_id==plan['image']['id'],'image id mismatch: '+repr(image_id)+' '+ins.stderr.strip()
 ctx['image']={'ref':plan['image']['ref'],'id':image_id,'executed_by':'image id'}
 ex=run(['docker','inspect',plan['container']['name'],'--format','{{.Id}}'])
 assert ex.returncode!=0 and not ex.stdout.strip(),'container name already exists: '+plan['container']['name']
 s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); free=False
 try: s.bind((plan['container']['host_ip'],plan['container']['port'])); free=True
 except OSError: free=False
 finally: s.close()
 assert free,'port in use: %s:%d'%(plan['container']['host_ip'],plan['container']['port'])
 ctx['preconditions']={'container_name_absent':True,'port_free':True}
 devs=['/dev/davinci%d'%i for i in plan['devices']['davinci']]+['/dev/'+n for n in plan['devices']['shared']]
 for p in devs: assert pathlib.Path(p).exists(),'missing device node '+p
 ro=[p for p in plan['remote']['ro_paths_required'] if pathlib.Path(p).exists()]
 assert len(ro)==len(plan['remote']['ro_paths_required']),'missing required read-only mount(s)'
 absent=[p for p in plan['remote']['ro_paths_optional'] if not pathlib.Path(p).exists()]
 ro+=[p for p in plan['remote']['ro_paths_optional'] if pathlib.Path(p).exists()]
 ctx['mounts']={'weights_ro':plan['remote']['weights_dir'],'out_rw':str(RUN),'ro_extra':ro,'ro_absent_not_guessed':absent,'devices':devs}
 logrel=plan['remote']['ascend_log_path'][len(plan['remote']['out_mount']):].lstrip('/')
 assert logrel and not pathlib.PurePosixPath(logrel).is_absolute(),'unexpected ascend log path: '+plan['remote']['ascend_log_path']
 (RUN/logrel).mkdir(parents=True,exist_ok=True)
 argv=['docker','run']+plan['docker_argv']+['--name',plan['container']['name'],'-v',plan['remote']['weights_dir']+':'+plan['remote']['weights_mount']+':ro','-v',str(RUN)+':'+plan['remote']['out_mount']]
 for p in ro: argv+=['-v',p+':'+p+':ro']
 for p in devs: argv+=['--device',p]
 if 'runtime_overlay' in plan:
  smi=run(['npu-smi','info'],t=30)
  assert smi.returncode==0 and smi.stdout.count('No running processes found in NPU')==8,'Device process present; inspect before model launch'
  (RUN/'npu-smi-before.txt').write_text(smi.stdout+smi.stderr)
  ctx['npu_idle_precondition']={'empty_physical_cards':8,'raw_sha256':hashlib.sha256((smi.stdout+smi.stderr).encode()).hexdigest()}
  ov=plan['runtime_overlay'];source=pathlib.Path(ov['source'])
  assert str(source)=='/data/wio/glm5-3-vllm-execution/20261017-memcache/loader-fix-v3/site-packages'
  assert hashlib.sha256((source/'memcache_hybrid/__init__.py').read_bytes()).hexdigest()==ov['python_init_sha256']
  assert hashlib.sha256((source.parent/'aliases.json').read_bytes()).hexdigest()==ov['aliases_sha256']
  argv+=['-v',str(source)+':/opt/memcache-runtime:ro']
  original_env=json.loads(subprocess.check_output(['docker','image','inspect',image_id,'--format','{{json .Config.Env}}']))
  ld=next((v.split('=',1)[1] for v in original_env if v.startswith('LD_LIBRARY_PATH=')), '')
  py=next((v.split('=',1)[1] for v in original_env if v.startswith('PYTHONPATH=')), '')
  python_path='/opt/memcache-runtime'+(':'+py if py else '')
  probe=run(['docker','run','--rm','--pull','never','--network','none','--read-only','-e','PYTHONPATH='+python_path,'--entrypoint','python3',image_id,'-c','import importlib.util; s=importlib.util.find_spec("acl"); assert s is not None; print(s.origin)'])
  assert probe.returncode==0,'CANN Python module path unavailable: '+probe.stderr
  ctx['acl_module_preflight']=probe.stdout.strip()
  argv+=['-e','PYTHONPATH='+python_path,'-e','LD_LIBRARY_PATH=/opt/memcache-runtime/memcache_hybrid/lib:/opt/memcache-runtime/memfabric_hybrid/lib:'+ld]
  config=pathlib.Path(plan['memcache_config_source'])
  assert config.is_dir() and hashlib.sha256((config/'mmc-local.conf').read_bytes()).hexdigest()==plan['memcache_config_sha256']
  argv+=['-v',str(config)+':/memcache-config:ro']
  ctx['runtime_overlay']=ov
  ctx['memcache_config_sha256']=plan['memcache_config_sha256']
 for patch in plan.get('runtime_patches',[]):
  source=pathlib.Path(patch['source'])
  assert str(source).startswith('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/')
  assert patch['target'] in (
   '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py',
   '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py')
  assert hashlib.sha256(source.read_bytes()).hexdigest()==patch['sha256']
  argv+=['-v',str(source)+':'+patch['target']+':ro']
 ctx['runtime_patches']=plan.get('runtime_patches',[])
 for k,v in plan['env'].items(): argv+=['-e',k+'='+v]
 assert json.dumps(plan['additional_config'],separators=(',',':'))==plan['serve_argv'][-1],'additional-config element mismatch'
 argv+=['--entrypoint','/usr/bin/timeout',image_id,'--signal=TERM','--kill-after=30',str(plan['container']['lifetime_seconds'])]+plan['serve_argv']
 for e in argv: assert e and not any(c.isspace() for c in e),'argv element is not a single token: '+repr(e)
 launch={'argv':argv,'image_id':image_id,'started_host_utc':utc(),'ended_host_utc':None,'docker_run_rc':None,'stdout':None,'stderr':None,'container_id':None,'inspect':None}
 proc=run(argv); launch['docker_run_rc']=proc.returncode; launch['stdout']=proc.stdout; launch['stderr']=proc.stderr
 launch['container_id']=proc.stdout.strip(); launch['ended_host_utc']=utc()
 (RUN/'launch.json').write_text(json.dumps(launch,indent=2)+chr(10))
 if proc.returncode==0:
  ins=run(['docker','inspect',launch['container_id'],'--format','{{.Id}} {{.Image}} {{.Name}} {{.State.Status}} {{.State.ExitCode}}'])
  f=ins.stdout.strip().split()
  fields=dict(zip(['id','image','name','state_status','state_exit_code'],f)) if ins.returncode==0 and len(f)==5 else None
  launch['inspect']={'rc':ins.returncode,'stdout':ins.stdout.strip(),'stderr':ins.stderr.strip(),'fields':fields}
  assert fields and fields['id']==launch['container_id'] and fields['image']==image_id,'post-run inspect mismatch: '+ins.stderr.strip()
 (RUN/'launch.json').write_text(json.dumps(launch,indent=2)+chr(10))
except Exception as exc:
 ctx['failure']='%s: %s'%(type(exc).__name__,exc)
 if launch is not None: (RUN/'launch.json').write_text(json.dumps(launch,indent=2)+chr(10))
 (RUN/'context.json').write_text(json.dumps(ctx,indent=2)+chr(10))
 emit('failed',failure=ctx['failure'])
 raise
(RUN/'context.json').write_text(json.dumps(ctx,indent=2)+chr(10))
emit('launched' if launch['docker_run_rc']==0 else 'docker_run_failed',container_id=launch['container_id'],docker_run_rc=launch['docker_run_rc'])
sys.exit(0 if launch['docker_run_rc']==0 else 1)
