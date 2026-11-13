from pathlib import Path
import datetime,json,subprocess,urllib.request,time,re,os,argparse
p=argparse.ArgumentParser();p.add_argument('--label',choices=['screen','full'],required=True);p.add_argument('--profiles',type=int,nargs='+',choices=[32768,65536,131072,262144],default=[32768,65536,131072,262144]);args=p.parse_args();engine='sglang'
phase_bound=900 if args.label=='screen' else 7200;total_bound=1800 if args.label=='screen' else 21600
port={'vllm':18168,'sglang':18167}[engine]
R=Path('/data/sglang-dsv41');C='yxy-v41-align-aisbench-166-20261113';S=None;run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_ascops42_'+args.label;O=R/'results'/engine/run_id;state=R/('aisbench_run_state_'+engine+'_ascops42_'+args.label+'.json')
def stamp(phase,**kw):
 d={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':run_id,'engine':engine,'variant':'attentionTP4_DP4_prefill2048_HCCL1024_G42_FusedRMS_FP32proj_MarkovLoaded_MHC4096_Engram1024','requested_profiles':args.profiles,'phase':phase,**kw};p=state.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(state);print(json.dumps(d),flush=True)
def api(path,method='GET'):
 with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}'+path,method=method),timeout=60) as f:return f.status,f.read()
def inspect(name):return json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
assert not state.exists(),'Previous AISBench run requires review, no automatic reuse'
gate=json.loads((R/'service_gate_sglang_ascops42.json').read_text());assert gate['engram_placement']=='cpu' and gate['native_ready_verified'] is True and gate['required_optimizations_reviewed'] is True
S=gate['container']
for name in [C,S]:
 d=inspect(name);assert d['Config']['Labels']['codex.task']==R.name and d['State']['Running']
manifest=json.loads((R/'datasets/manifest.json').read_text());assert manifest['concurrency']==16 and manifest['output_tokens']==1024
for profile in manifest['profiles']:
 expected=16 if profile['phase']=='warmup' else 32
 assert profile['requests']==expected and profile['min_actual']==profile['max_actual']==profile['input_tokens']
O.mkdir(parents=True,exist_ok=False);status,raw=api('/server_info' if engine=='sglang' else '/server_info?config_format=json');assert status==200;(O/'server_info_before.json').write_bytes(raw)
started=time.monotonic()
service_container_id=inspect(S)['Id']
def verify_npu_ownership(tag):
 q=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15)
 assert q.returncode==0, 'NPU ownership query failed; measurement cannot be validated'
 foreign=[];own=set()
 for line in q.stdout.splitlines():
  f=[x.strip() for x in line.split('|')[1:-1]]
  if len(f)<4 or not re.fullmatch(r'\d+\s+\d+',f[0]) or not f[1].isdigit():continue
  try:cg=(Path('/proc')/f[1]/'cgroup').read_text()
  except FileNotFoundError:continue
  if service_container_id in cg:own.add(int(f[1]))
  else:foreign.append({'pid':int(f[1]),'npu_chip':f[0],'memory_mib':f[3],'container_ids':sorted(set(re.findall(r'[0-9a-f]{64}',cg)))})
 sample={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tag':tag,'own_npu_process_count':len(own),'foreign':foreign}
 with (O/'npu_ownership_samples.jsonl').open('a') as log:log.write(json.dumps(sample)+'\n')
 if foreign:
  (O/'invalid_external_NPU_workload.json').write_text(json.dumps({'sample':sample,'npu_smi':q.stdout},indent=2))
  # Both containers were explicitly checked as owned by this task above.
  for own_name in [C,S]:
   assert inspect(own_name)['Config']['Labels']['codex.task']==R.name
   subprocess.run(['docker','stop','--time','20',own_name],capture_output=True,text=True,timeout=45)
  raise RuntimeError('External NPU workload overlapped measurement; partial results retained, own client/service stopped only')
 assert len(own)>=16, 'Missing all16 modelrank contexts; measurement cannot proceed'
try:
 for size in args.profiles:
  for phase,num in [('warmup',16),('formal',32)]:
   assert time.monotonic()-started<total_bound,'6hour total bound reached'
   code,body=api('/flush_cache?timeout=60' if engine=='sglang' else '/reset_prefix_cache','POST');assert code==200 and (b'Cache flushed' in body if engine=='sglang' else body.strip().lower()!=b'false')
   (O/f'{size}_{phase}_flush.txt').write_bytes(body)
   target=f'/task/results/{engine}/{run_id}/{phase}_{size}';argv=['docker','exec',C,'timeout','--signal=TERM','--kill-after=30',str(phase_bound),'ais_bench',f'/task/configs/aisbench_{engine}_{phase}_{size}.py','--mode','perf','--num-warmups','0','--num-prompts',str(num),'--max-num-workers','1','--work-dir',target]
   (O/f'{size}_{phase}_command.json').write_text(json.dumps(argv,indent=2));stamp('aisbench_running',input_tokens=size,output_tokens=1024,concurrency=16,requests=num,scored=phase=='formal',subphase=phase)
   verify_npu_ownership(f'{size}_{phase}_before')
   with (O/f'{size}_{phase}_cli.log').open('w') as out:
    proc=subprocess.Popen(argv,stdout=out,stderr=subprocess.STDOUT);deadline=time.monotonic()+phase_bound+40;next_check=time.monotonic()+30
    try:
     while proc.poll() is None:
      if time.monotonic()>=deadline:raise TimeoutError('AISBench phase exceeded configured bound')
      if time.monotonic()>=next_check:
       verify_npu_ownership(f'{size}_{phase}_during');next_check=time.monotonic()+30
      time.sleep(2)
     rc=proc.returncode
    finally:
     if proc.poll() is None:
      assert inspect(C)['Config']['Labels']['codex.task']==R.name
      subprocess.run(['docker','stop','--time','20',C],capture_output=True,text=True,timeout=45)
      try:proc.wait(timeout=30)
      except subprocess.TimeoutExpired:proc.kill();proc.wait()
   stamp('aisbench_cli_terminal',input_tokens=size,subphase=phase,exit_code=rc,raw_output_dir=target,results_validated=False)
   assert rc==0,f'AISBench {phase} {size} failed; inspect raw logs before any retry'
   verify_npu_ownership(f'{size}_{phase}_after')
  status,raw=api('/server_info' if engine=='sglang' else '/server_info?config_format=json');assert status==200;(O/f'server_info_after_{size}.json').write_bytes(raw)
 stamp('all_four_cli_runs_terminal_require_raw_result_review' if args.profiles==[32768,65536,131072,262144] else 'selected_profiles_cli_terminal_require_raw_result_review',raw_output_dir=str(O),performance_validated=False)
except BaseException as e:
 stamp('stopped_require_review',exception=type(e).__name__,error=str(e),performance_validated=False);raise
