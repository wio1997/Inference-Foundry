import json,subprocess,sys,time,urllib.request,shlex
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
root=Path(__file__).parent;deploy=Path("/data/tiankuan/wio/glm52-pd/deploy");events=[];opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def event(name,**fields):
 events.append({'at':utc(),'event':name,**fields});atomic_json(root/'deployment_state.json',{'status':name,'events':events});print(json.dumps(events[-1]),flush=True)
identities={}
for role,node,port in [('P','166','9081'),('D','167','9900')]:
 def command(argv):
  if node=='167':argv=['ssh','-o','BatchMode=yes','root@172.16.10.167',shlex.join(argv)]
  p=subprocess.run(argv,capture_output=True,text=True)
  if p.returncode!=0:raise RuntimeError(role+' identity command failed: '+p.stderr)
  return p.stdout
 text=command(['docker','top','glm52-single','-eo','pid,ppid,comm,args'])
 rows=[line for line in text.splitlines() if '/bin/vllm serve ' in line and '--port '+port in line]
 if len(rows)!=1:raise RuntimeError(role+' model owner not uniquely known')
 pid=rows[0].split()[0]
 code="import json,pathlib;print(json.dumps([x.decode() for x in pathlib.Path('/proc/"+pid+"/cmdline').read_bytes().split(bytes([0])) if x]))"
 args=json.loads(command(['python3','-c',code]));flags={k:args[args.index(k)+1] for k in ['--max-num-batched-tokens','--tensor-parallel-size','--decode-context-parallel-size','--speculative-config','--compilation-config','--kv-transfer-config','--gpu-memory-utilization']}
 if flags['--gpu-memory-utilization']!='0.87' or flags['--max-num-batched-tokens']!=('4096' if role=='P' else '8192') or flags['--tensor-parallel-size']!='16' or flags['--decode-context-parallel-size']!='16' or json.loads(flags['--speculative-config'])['num_speculative_tokens']!=5 or json.loads(flags['--compilation-config'])['cudagraph_mode']!='FULL_DECODE_ONLY' or json.loads(flags['--kv-transfer-config'])['kv_role']!=('kv_producer' if role=='P' else 'kv_consumer'):raise RuntimeError(role+' candidate configuration differs')
 if role in ['P','D']:
  assert args[args.index('--tool-call-parser')+1]=='glm47_contract'
  assert args[args.index('--tool-parser-plugin')+1]==('/data/tiankuan/wio/glm52-pd/deploy/scripts/glm_tool_contract_run21.py' if role=='P' else '/data/tiankuan/wio/glm52-pd/deploy/scripts/glm_tool_contract_run22.py')
 identity=command(['python3','-c',"import json,pathlib;raw=pathlib.Path('/proc/"+pid+"/stat').read_text();print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':raw[raw.rfind(')')+2:].split()[19],'raw_stat':raw}))"])
 identities[role]={'host':node,'pid':int(pid),'identity':json.loads(identity),'argv':args,'source_run':'GLM-RUN-0021' if role=='P' else 'GLM-RUN-0022'}
prior=json.loads((root.parent/'GLM-RUN-0022/adopted_model_identities.json').read_text())
for role,value in identities.items():
 previous=prior[role]
 assert value['pid']==previous['pid'] and value['argv']==previous['argv']
 for key in ['boot_id','start_ticks']:assert value['identity'][key]==previous['identity'][key]
atomic_json(root/'adopted_model_identities.json',identities);event('both_existing_model_owners_verified',pids={k:v['pid'] for k,v in identities.items()})
end=time.monotonic()+1800
while True:
 ready={}
 for name,url in [('P','http://172.16.10.166:9081'),('D','http://172.16.10.167:9900'),('proxy','http://172.16.10.166:8000')]:
  try:
   with opener.open(url+('/healthcheck' if name=='proxy' else '/health'),timeout=3) as response:ready[name]=response.status==200
  except Exception:ready[name]=False
 event('readiness',ready=ready)
 if all(ready.values()):event('ready');break
 for role,node in [('P','166'),('D','167')]:
  # Each log is already known to belong to the adopted entry; no start/truncate race.
  argv=['tail','-n','300',str(deploy/('logs/P_166.log' if role=='P' else 'logs/D_167.log'))]
  if node=='167':argv=['ssh','-o','BatchMode=yes','root@172.16.10.167',shlex.join(argv)]
  p=subprocess.run(argv,capture_output=True,text=True);text=p.stdout
  if 'Engine core initialization failed' in text or 'OutOfMemoryError' in text or 'vllm serve: error:' in text:raise RuntimeError(role+' adopted startup failed; no replay')
 if time.monotonic()>end:raise RuntimeError('adopted startup timed out; no replay')
 time.sleep(15)
