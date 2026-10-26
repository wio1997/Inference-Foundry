"""Guard exact P/D services, observe native chunk, run one owned cancellation lifecycle."""
import ast, datetime, hashlib, json, pathlib, shlex, subprocess, sys

R = pathlib.Path(__file__).resolve().parents[1]; tag = 'L039-R092'
O = R / f'runs/{tag}-pd-lifecycle'; O.mkdir(parents=True, exist_ok=False)
tree = ast.parse((R / 'scripts/run_L039_pd_eos_v1.py').read_text())
HOST = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HOST' for t in n.targets))
HOST = HOST.replace('print(json.dumps(v))', "v['runtime']={k:info.get(k) for k in ['cuda_graph_config','chunked_prefill_size','mem_fraction_static','enable_deterministic_inference','speculative_use_rejection_sampling']};v['internal_memory']=[x.get('memory_usage') for x in info.get('internal_states',[])];print(json.dumps(v))")

LAUNCH = r'''
import hashlib,json,pathlib,subprocess,sys,time
j=json.load(sys.stdin);plan=j['plan'];tag=plan['run'];assert tag=='L039-R092'
base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36'
root=base/(tag+'-pd-lifecycle');assert not root.exists(),'Never replay an existing lifecycle run'
assert hashlib.sha256(j['client'].encode()).hexdigest()==plan['client_sha256']
root.mkdir(mode=0o700)
for name,text in [('client.py',j['client']),('payload.json',json.dumps({'plan':plan,'inputs':j['inputs']})),('plan.json',json.dumps(plan,indent=2)+'\n')]:
 p=root/name;p.write_text(text);p.chmod(0o400)
with (root/'payload.json').open('r') as stdin,(root/'stdout.json').open('x') as stdout,(root/'stderr.log').open('x') as stderr:
 p=subprocess.Popen(['python3','-u',str(root/'client.py')],stdin=stdin,stdout=stdout,stderr=stderr,cwd=root)
 h={'task':'sglang-pd-spec-36','run':tag,'pid':p.pid,'proc_start_ticks':pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],'source_path':str(root/'client.py'),'client_sha256':plan['client_sha256'],'root':str(root)};(root/'parent-handle.json').write_text(json.dumps(h,indent=2)+'\n')
 try:code=p.wait(timeout=500)
 except subprocess.TimeoutExpired:
  print(json.dumps({'status':'STILL_RUNNING_REOBSERVE_EXACT_HANDLE','handle':h,'actual_worker_exit_code':None}));sys.exit(124)
result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else None
comp={'handle':h,'actual_worker_exit_code':code,'host_monotonic_ns':time.monotonic_ns(),'stdout_sha256':hashlib.sha256((root/'stdout.json').read_bytes()).hexdigest(),'stderr_sha256':hashlib.sha256((root/'stderr.log').read_bytes()).hexdigest(),'result_sha256':hashlib.sha256((root/'result.json').read_bytes()).hexdigest() if result is not None else None};(root/'completion.json').write_text(json.dumps(comp,indent=2)+'\n')
print(json.dumps({'completion':comp,'result':result,'stderr':(root/'stderr.log').read_text()[-20000:]}));sys.exit(code)
'''

WATCH = r'''
import datetime,hashlib,json,pathlib,re,subprocess,sys,time
j=json.load(sys.stdin);item=j['item'];base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36'
x=json.loads(subprocess.run(['docker','inspect','--format','{{json .}}',item['container']],capture_output=True,text=True,check=True,timeout=15).stdout);assert x['Id']==item['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['State']['Running']
deadline=time.monotonic()+120;last=''
while time.monotonic()<deadline:
 p=subprocess.run(['docker','logs','--since',j['since_utc'],item['container']],capture_output=True,text=True,timeout=12);assert p.returncode==0,p.stderr;last=p.stdout+p.stderr
 for line in last.splitlines():
  if 'Prefill batch' in line and '#new-seq: 1,' in line and '#new-token: 4096,' in line:
   m=re.search(r'#pending-token: (\d+)',line)
   if m and int(m.group(1))>0:
    print(json.dumps({'run':j['run'],'node':44,'container_id':x['Id'],'new_tokens':4096,'pending_tokens':int(m.group(1)),'line':line,'log_since_P_host_utc':j['since_utc'],'observed_P_host_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observed_P_monotonic_ns':time.monotonic_ns(),'log_sha256':hashlib.sha256(last.encode()).hexdigest(),'identity_basis':'Onlyvictimsubmittedbeforethisevent; initialidleflushmustpassclientbeforegeneration'}));sys.exit(0)
 time.sleep(0.2)
print(json.dumps({'status':'NO_NATIVE_CHUNK_EVENT','last_log_tail':last[-20000:]}));sys.exit(1)
'''

GATE = r'''
import hashlib,json,pathlib,sys,time
j=json.load(sys.stdin);plan=j['plan'];assert plan['run']=='L039-R092';base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36';root=base/(plan['run']+'-pd-lifecycle');assert root.is_dir() and hashlib.sha256((root/'client.py').read_bytes()).hexdigest()==plan['client_sha256'] and json.loads((root/'plan.json').read_text())==plan
h=json.loads((root/'worker-handle.json').read_text());ph=json.loads((root/'parent-handle.json').read_text());assert h['pid']==ph['pid'] and h['proc_start_ticks']==ph['proc_start_ticks'];proc=pathlib.Path('/proc')/str(h['pid']);stat=(proc/'stat').read_text().split();assert stat[21]==h['proc_start_ticks'] and stat[2] not in ['Z','X'] and h['source_path'].encode() in (proc/'cmdline').read_bytes().split(b'\0')
v={'run':plan['run'],'P_event':j['P_event'],'gate_created_D_monotonic_ns':time.monotonic_ns(),'worker_handle':h};q=root/'chunk-gate.json';assert not q.exists();q.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
'''

JUMP = r'''
import concurrent.futures,hashlib,json,subprocess,sys
j=json.load(sys.stdin);v={'run':j['plan']['run'],'services_restarted':False,'abort_all_used':False}
for node,lease in j['leases'].items():
 x=json.loads(subprocess.run(['kubectl','get','pod','-n','default',lease['name'],'-o','json'],capture_output=True,text=True,check=True,timeout=25).stdout);assert x['metadata']['uid']==lease['uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['spec']['nodeName']==f'a3-5u-atlas900rcka3-{node}' and x['status']['phase']=='Running' and next(c['status'] for c in x['status']['conditions'] if c['type']=='Ready')=='True'
def call(node,code,payload,timeout):
 return subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{node}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',code],input=json.dumps(payload),capture_output=True,text=True,timeout=timeout)
def snapshot(phase):
 def one(item):
  p=call(item['node'],j['host'],{'item':item,'url':j['plan']['endpoints'][item['role']],'phase':phase,**{k:j['plan'][k] for k in ['required_source_hashes','model_config_sha256','generation_config_sha256','model_eos_ids']}},75);assert p.returncode==0,p.stderr;return str(item['node']),json.loads(p.stdout)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:return dict(pool.map(one,j['targets']))
v['pre']=snapshot('pre')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 client=pool.submit(call,46,j['launcher'],{k:j[k] for k in ['plan','inputs','client']},550)
 item=next(x for x in j['targets'] if x['node']==44)
 p=call(44,j['watcher'],{'run':j['plan']['run'],'item':item,'since_utc':v['pre']['44']['observed_utc']},145)
 v['chunk_watch']={'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
 if p.returncode==0:
  p=call(46,j['gate'],{'plan':j['plan'],'P_event':v['chunk_watch']['data']},30);v['gate']={'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
 p=client.result();v['client']={'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
try:v['post']=snapshot('post')
except BaseException as error:v['post_error']=str(error)
v['status']='PASS_PREFILL_CANCEL_SURVIVORS_RETRY' if v['client']['exit_code']==0 and v['chunk_watch']['exit_code']==0 and v.get('gate',{}).get('exit_code')==0 and 'post' in v else 'FAIL'
print(json.dumps(v));sys.exit(0 if v['status']=='PASS_PREFILL_CANCEL_SURVIVORS_RETRY' else 1)
'''

plan = json.loads((R / f'specs/{tag}-pd-plan.json').read_text()); inputs = []
for item in plan['inputs']:
    data = (R / item['path']).read_bytes(); assert hashlib.sha256(data).hexdigest() == item['sha256']
    ids = json.loads(data)['input_ids']; assert hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == item['wire_ids_sha256']
    inputs.append({**item, 'input_ids': ids})
client = (R / 'scripts/L039_pd_cancel_client_v1.py').read_text(); assert hashlib.sha256(client.encode()).hexdigest() == plan['client_sha256']
h = json.loads((R / 'specs/L039-R077-targets.json').read_text())
payload = {'plan': plan, 'inputs': inputs, 'client': client, 'host': HOST, 'launcher': LAUNCH, 'watcher': WATCH, 'gate': GATE, 'targets': [{**t, 'role': 'prefill' if t['node'] == 44 else 'decode'} for t in h['targets']], 'leases': h['resource_leases']}
cmd = ['ssh', '-S', str(R / 'connections/l038.sock'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-p', '60011', 'root@61.241.77.34', shlex.join(['python3', '-c', JUMP])]
started = datetime.datetime.now(datetime.timezone.utc).isoformat(); p = subprocess.run(cmd, input=json.dumps(payload), capture_output=True, text=True, timeout=700)
(O / 'remote-result.json').write_text(p.stdout); (O / 'stderr.log').write_text(p.stderr)
value = json.loads(p.stdout) if p.stdout else {}
receipt = {'started_utc': started, 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'exit_code': p.returncode, 'status': value.get('status', 'UNKNOWN'), 'actual_requests_on_910C': True, 'script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), 'artifacts': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n'); print(json.dumps(receipt)); sys.exit(p.returncode)
