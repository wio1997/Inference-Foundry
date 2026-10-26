#!/usr/bin/env python3
"""Read exact model/store handles and owned pure readiness endpoints; no inference request or restart."""
import datetime,hashlib,json,pathlib,re,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];tag=sys.argv[1];assert re.fullmatch(r'L039-R\d{3}',tag);O=R/'runs'/(tag+'-full-model-state');O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import datetime,hashlib,json,pathlib,re,subprocess,sys,time,urllib.error,urllib.request
j=json.load(sys.stdin);base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36';v={'node':j['node'],'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host_monotonic_ns':time.monotonic_ns(),'read_only':True,'inference_requests':0,'restarted':False}
def run(a,t=25):return subprocess.run(a,capture_output=True,text=True,timeout=t)
p=run(['docker','inspect','--format','{{json .}}',j['container']]);v['inspect_exit']=p.returncode
if p.returncode!=0:v['status']='HANDLE_MISSING';v['error']=p.stderr;print(json.dumps(v));sys.exit(0)
x=json.loads(p.stdout);assert x['Id']==j['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R058';v['selected_container_env']={key:next((e.split('=',1)[1] for e in x['Config']['Env'] if e.startswith(key+'=')),None) for key in ['HCCL_BUFFSIZE','HCCL_IF_IP','ASCEND_RT_VISIBLE_DEVICES']};v['container']=j['container'];v['container_id']=x['Id'];v['state']=x['State'];v['status']='STARTING' if x['State']['Running'] else 'MODEL_PROCESS_TERMINAL';root=pathlib.Path(j['root']);assert root.parent==base
for filename in ['preflight.json','handle.json']:
 p=root/filename
 if p.is_file():v[filename]=json.loads(p.read_text())
p=run(['docker','logs','--tail','220',j['container']]);v['own_log_tail']={'exit_code':p.returncode,'stdout':p.stdout[-180000:],'stderr':p.stderr[-180000:]};v['own_log_tail_sha256']={k:hashlib.sha256(v['own_log_tail'][k].encode()).hexdigest() for k in ['stdout','stderr']}
p=run(['docker','top',j['container'],'-eo','pid,ppid,comm']);v['own_processes']={'exit_code':p.returncode,'stdout':p.stdout[:25000],'stderr':p.stderr}
p=run(['npu-smi','info']);v['NPU_inventory']={'exit_code':p.returncode,'sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'HBM_used_MiB':list(map(int,re.findall(r'(\d+)\s*/\s*65536',p.stdout))),'no_running_process_groups':p.stdout.count('No running processes found in NPU'),'stderr':p.stderr}
if x['State']['Running']:
 p=run(['docker','stats','--no-stream','--format','{{json .}}',j['container']],t=20)
 if p.returncode==0:v['own_resource_snapshot']=json.loads(p.stdout)
allowed={'model_path','tokenizer_path','model_type','is_generation','dtype','kv_cache_dtype','quantization','tp_size','ep_size','dp_size','pp_size','dcp_size','attn_cp_size','enable_dp_attention','speculative_algorithm','speculative_num_steps','speculative_eagle_topk','speculative_num_draft_tokens','speculative_draft_model_path','speculative_draft_model_quantization','disaggregation_mode','disaggregation_transfer_backend','context_length','chunked_prefill_size','max_running_requests','mem_fraction_static','attention_backend','prefill_attention_backend','decode_attention_backend','page_size','sampling_backend','cuda_graph_config','disable_overlap_schedule','disable_radix_cache','disaggregation_decode_extra_slots','max_total_num_tokens','max_prefill_tokens','max_req_input_len','max_req_len','version','num_hidden_layers','vocab_size','moe_a2a_backend','deepep_mode'}
def select(obj,path=''):
 result={}
 if isinstance(obj,dict):
  for k,z in obj.items():
   key=path+'.'+k if path else k
   if k in allowed:result[key]=z
   elif isinstance(z,(dict,list)):result.update(select(z,key))
 elif isinstance(obj,list):
  for i,z in enumerate(obj[:32]):result.update(select(z,path+f'[{i}]'))
 return result
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));v['HTTP']={};port=39044 if j['node']==44 else 39046
for endpoint in ['ready','get_model_info','get_server_info']:
 url=f"http://172.16.12.{j['node']}:{port}/{endpoint}"
 try:
  with opener.open(url,timeout=3) as r:data=r.read(2*1024**2);item={'status':r.status,'response_bytes':len(data)}
  try:item['selected_facts']=select(json.loads(data))
  except (json.JSONDecodeError,UnicodeDecodeError):item['body_prefix']=data[:200].decode(errors='replace') if endpoint=='ready' else 'NON_JSON'
 except urllib.error.HTTPError as e:item={'status':e.code,'error':str(e)}
 except Exception as e:item={'status':None,'error':type(e).__name__+': '+str(e)}
 v['HTTP'][endpoint]=item
if x['State']['Running'] and all(v['HTTP'][e].get('status')==200 for e in ['ready','get_model_info','get_server_info']):v['status']='API_READY_ONLY_NO_PD_WORKLOAD'
if j['node']==46:
 store=j['store'];p=run(['docker','inspect','--format','{{json .}}',store['name']]);assert p.returncode==0;x=json.loads(p.stdout);assert x['Id']==store['container_id'] and x['Config']['Labels']['run']=='L039-R033';v['store']={'container_id':x['Id'],'state':x['State'],'status_file':json.loads((pathlib.Path(store['root'])/'state.json').read_text())}
print(json.dumps(v))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin)
def observe(item):
 node=item['node'];p=subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{node}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(item),capture_output=True,text=True,timeout=105);return str(node),{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(observe,j['targets']))
leases={}
for node,expected in j['leases'].items():
 p=subprocess.run(['kubectl','get','pod','-n','default',expected['name'],'-o','json'],capture_output=True,text=True,timeout=20)
 if p.returncode==0:
  x=json.loads(p.stdout);leases[node]={'name':x['metadata']['name'],'uid':x['metadata']['uid'],'expected_uid':expected['uid'],'phase':x['status']['phase'],'reason':x['status'].get('reason'),'message':x['status'].get('message'),'ready':next((c['status'] for c in x['status'].get('conditions',[]) if c['type']=='Ready'),None)}
 else:leases[node]={'exit_code':p.returncode,'stderr':p.stderr}
print(json.dumps({'read_only':True,'inference_requests':0,'restarted':False,'nodes':results,'resource_leases':leases,'PD_MTP_workload_executed':False,'performance':'NOT_MEASURED'}));sys.exit(0 if all(x['exit_code']==0 for x in results.values()) else 1)
'''
launch=json.loads((R/'runs/L039-R058-full-bootstrap/remote-result.json').read_text());store=json.loads((R/'runs/L039-R033-shared-store/remote-result.json').read_text())['store']['data'];targets=[]
for n,x in launch['nodes'].items():
 d=x['data'];assert d and d['created'] and d.get('container_id'),'No model handle created on'+n;targets.append({'node':int(n),'container':d['name'],'container_id':d['container_id'],'root':d['root'],'store':{k:store[k] for k in ['name','container_id','root']}})
leases=json.loads((R/'specs/L039-full-bootstrap-plan-v6.json').read_text())['resource_leases']
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':HOST,'targets':targets,'leases':leases}),capture_output=True,text=True,timeout=140)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'inference_requests':0,'restarted':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
