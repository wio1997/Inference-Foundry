"""Read exact long-client PID and own services; never restart or send generation."""
import datetime,hashlib,json,pathlib,re,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];tag=sys.argv[1];assert re.fullmatch(r'L039-R\d{3}',tag);O=R/f'runs/{tag}-long-state';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import datetime,hashlib,json,pathlib,re,subprocess,sys,time,urllib.request
j=json.load(sys.stdin);item=j['item'];base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36'
x=json.loads(subprocess.run(['docker','inspect','--format','{{json .}}',item['container']],capture_output=True,text=True,check=True,timeout=25).stdout);assert x['Id']==item['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']==item['run']
v={'node':item['node'],'container_id':x['Id'],'state':x['State'],'host_monotonic_ns':time.monotonic_ns(),'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'inference_requests':0,'restarted':False}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));url='http://172.16.12.'+str(item['node'])+':'+str(39044 if item['node']==44 else 39046)
try:
 with opener.open(url+'/ready',timeout=5) as response:v['pure_ready']=response.status
 with opener.open(url+'/server_info',timeout=8) as response:info=json.load(response)
 v['selected_runtime_fields']={k:info.get(k) for k in ['context_length','max_running_requests','kv_cache_dtype','speculative_algorithm','speculative_num_steps','speculative_eagle_topk','speculative_num_draft_tokens','disaggregation_mode','disaggregation_transfer_backend','cuda_graph_config']}
 v['selected_internal_states']=[{k:d.get(k) for k in ['memory_usage','effective_max_running_requests_per_dp','decode_cuda_graph_bs','decode_cuda_graph_memory_gb','avg_spec_accept_length']} for d in info.get('internal_states',[])]
except Exception as error:v['API_error']=type(error).__name__+': '+str(error)
p=subprocess.run(['docker','logs','--tail','250',item['container']],capture_output=True,text=True,timeout=25);v['own_log_tail']=p.stdout+p.stderr;v['own_log_tail_sha256']=hashlib.sha256(v['own_log_tail'].encode()).hexdigest()
p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);v['HBM_used_MiB']=list(map(int,re.findall(r'(\d+)\s*/\s*65536',p.stdout)));v['NPU_inventory_sha256']=hashlib.sha256(p.stdout.encode()).hexdigest()
if item['node']==46:
 h=j['long_handle'];assert h['task']=='sglang-pd-spec-36' and h['run']=='L039-R087';root=pathlib.Path(h['root']);clientroot=pathlib.Path(h['client_root']);assert root.parent==clientroot.parent==base and json.loads((root/'handle.json').read_text())==h
 assert hashlib.sha256((root/'client.py').read_bytes()).hexdigest()==h['client_sha256'] and hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()==h['plan_sha256']
 proc=pathlib.Path('/proc')/str(h['pid']);v['long_worker']={'handle':h,'PID_exists':proc.exists(),'actual_worker_exit_code':'UNKNOWN (not an adopted child)'}
 if proc.exists():
  stat=(proc/'stat').read_text().split();cmd=(proc/'cmdline').read_bytes().split(b'\0');assert stat[21]==h['proc_start_ticks'] and h['source_path'].encode() in cmd
  v['long_worker'].update(process_state=stat[2],confirmed_live=stat[2] not in ['Z','X'])
 else:v['long_worker']['confirmed_live']=False
 q=clientroot/'worker-handle.json'
 if q.exists():
  worker=json.loads(q.read_text());assert worker['pid']==h['pid'] and worker['proc_start_ticks']==h['proc_start_ticks'] and worker['client_sha256']==h['client_sha256'];v['long_worker']['worker_self_handle']=worker
 q=clientroot/'result.json'
 if q.exists():data=q.read_bytes();assert len(data)<16*1024**2;v['long_worker'].update(result_sha256=hashlib.sha256(data).hexdigest(),result=json.loads(data))
 for filename in ['stdout.json','stderr.log']:
  q=root/filename;data=q.read_bytes();v['long_worker'][filename]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  if filename=='stderr.log':v['long_worker'][filename]['tail']=data[-18000:].decode(errors='replace')
print(json.dumps(v))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin);v={'read_only':True,'inference_requests':0,'restarted':False,'leases':{}}
for n,l in j['leases'].items():
 x=json.loads(subprocess.run(['kubectl','get','pod','-n','default',l['name'],'-o','json'],capture_output=True,text=True,check=True,timeout=25).stdout);assert x['metadata']['uid']==l['uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36';v['leases'][n]={'uid':x['metadata']['uid'],'phase':x['status']['phase'],'ready':next((c['status'] for c in x['status'].get('conditions',[]) if c['type']=='Ready'),None)}
def one(item):
 p=subprocess.run(['kubectl','exec','-i','-n','default',f"sglang-container-access-{item['node']}-20261013",'-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps({'item':item,'long_handle':j['long_handle']}),capture_output=True,text=True,timeout=100);return str(item['node']),{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:v['nodes']=dict(pool.map(one,j['targets']))
print(json.dumps(v));sys.exit(0 if all(x['exit_code']==0 for x in v['nodes'].values()) else 1)
'''
launch=json.loads((R/'runs/L039-R087-pd-launch/remote-result.json').read_text());assert launch['status']=='LONG_PD_LAUNCHED_NOT_VERIFIED';h=json.loads((R/'specs/L039-R077-targets.json').read_text());payload={'host':HOST,'long_handle':launch['launch']['handle'],'targets':h['targets'],'leases':h['resource_leases']};cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];started=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=170)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);receipt={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'inference_requests':0,'restarted':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}};(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
