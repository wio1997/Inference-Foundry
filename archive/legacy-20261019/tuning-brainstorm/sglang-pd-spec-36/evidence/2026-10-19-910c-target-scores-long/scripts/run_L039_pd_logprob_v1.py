"""Exact own-service guards, then run frozen real PD client on910C46."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];tag=sys.argv[1];assert tag in ['L039-R085','L039-R086'];O=R/f'runs/{tag}-pd-functional';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import datetime,hashlib,json,pathlib,subprocess,sys,time,urllib.request
j=json.load(sys.stdin);item=j['item'];base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36'
x=json.loads(subprocess.run(['docker','inspect','--format','{{json .}}',item['container']],capture_output=True,text=True,check=True,timeout=25).stdout);assert x['Id']==item['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']==item['run'] and x['State']['Running']
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));url=j['url']
with opener.open(url+'/ready',timeout=6) as response:assert response.status==200
with opener.open(url+'/server_info',timeout=10) as response:info=json.load(response)
for k,value in {'tp_size':16,'ep_size':16,'dp_size':1,'disaggregation_mode':item['role'],'disaggregation_transfer_backend':'ascend','speculative_algorithm':'EAGLE','speculative_num_steps':3,'speculative_eagle_topk':1,'speculative_num_draft_tokens':4,'kv_cache_dtype':'bfloat16','context_length':33792,'max_running_requests':3,'mem_fraction_static':0.9}.items():assert info[k]==value,(k,info.get(k),value)
v={'node':item['node'],'container_id':x['Id'],'state':x['State'],'pure_ready':200,'host_monotonic_ns':time.monotonic_ns(),'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':j['phase'],'source_preflight':json.loads((pathlib.Path(item['root'])/'preflight.json').read_text())}
if j['phase']=='post':
 p=subprocess.run(['docker','logs','--tail','400',item['container']],capture_output=True,text=True,timeout=25);text=p.stdout+p.stderr;v['own_log_tail']=text;v['own_log_tail_sha256']=hashlib.sha256(text.encode()).hexdigest()
print(json.dumps(v))
'''
JUMP=r'''
import concurrent.futures,hashlib,json,subprocess,sys
j=json.load(sys.stdin);v={}
for node,lease in j['leases'].items():
 x=json.loads(subprocess.run(['kubectl','get','pod','-n','default',lease['name'],'-o','json'],capture_output=True,text=True,check=True,timeout=25).stdout);assert x['metadata']['uid']==lease['uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['spec']['nodeName']==f'a3-5u-atlas900rcka3-{node}' and x['status']['phase']=='Running' and next(c['status'] for c in x['status']['conditions'] if c['type']=='Ready')=='True'
def call(node,code,payload,timeout):
 return subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{node}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',code],input=json.dumps(payload),capture_output=True,text=True,timeout=timeout)
def snapshot(phase):
 def one(item):
  p=call(item['node'],j['host'],{'item':item,'url':j['plan']['endpoints'][item['role']],'phase':phase},75);assert p.returncode==0,p.stderr;return str(item['node']),json.loads(p.stdout)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:return dict(pool.map(one,j['targets']))
v['pre']=snapshot('pre');assert hashlib.sha256(j['client'].encode()).hexdigest()==j['plan']['client_sha256']
p=call(46,j['client'],{'plan':j['plan'],'inputs':j['inputs']},430);v['client']={'exit_code':p.returncode,'result':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
try:v['post']=snapshot('post')
except BaseException as error:v['post_error']=str(error)
v['status']='PASS_FUNCTIONAL_PAIR_ONLY' if p.returncode==0 and 'post' in v else 'FAIL';print(json.dumps(v));sys.exit(0 if v['status']=='PASS_FUNCTIONAL_PAIR_ONLY' else 1)
'''
plan=json.loads((R/f'specs/{tag}-pd-plan.json').read_text());inputs=[]
for item in plan['inputs']:
 data=(R/item['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256'];ids=json.loads(data)['input_ids'];assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==item['wire_ids_sha256'];inputs.append({**item,'input_ids':ids})
client=(R/'scripts/L039_pd_logprob_client_v1.py').read_text();h=json.loads((R/'specs/L039-R077-targets.json').read_text());payload={'plan':plan,'inputs':inputs,'client':client,'host':HOST,'targets':[{**t,'role':'prefill' if t['node']==44 else 'decode'} for t in h['targets']],'leases':h['resource_leases']}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=700)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);v=json.loads(p.stdout) if p.stdout else {};receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'status':v.get('status','UNKNOWN'),'actual_requests_on_910C':True,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
