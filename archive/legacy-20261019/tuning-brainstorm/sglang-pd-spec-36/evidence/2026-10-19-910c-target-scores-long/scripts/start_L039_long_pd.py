"""Launch the frozen original long PD workload once; preserve an inspectable PID."""
import ast,datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];tag='L039-R087';O=R/f'runs/{tag}-pd-launch';O.mkdir(parents=True,exist_ok=False)
tree=ast.parse((R/'scripts/run_L039_pd_client_v1.py').read_text());HOST=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HOST' for t in n.targets))
LAUNCH=r'''
import hashlib,json,pathlib,subprocess,sys,time
j=json.load(sys.stdin);plan=j['plan'];tag=plan['run'];assert tag=='L039-R087';base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36'
root=base/(tag+'-pd-launch');clientroot=base/(tag+'-pd-client');assert not root.exists() and not clientroot.exists(),'Do not replay an existing long run'
assert hashlib.sha256(j['client'].encode()).hexdigest()==plan['client_sha256'];assert hashlib.sha256(j['plan_text'].encode()).hexdigest()==j['plan_sha256'];assert json.loads(j['plan_text'])==plan
root.mkdir(mode=0o700)
for name,text in [('client.py',j['client']),('plan.json',j['plan_text']),('payload.json',json.dumps({'plan':plan,'inputs':j['inputs']}))]:p=root/name;p.write_text(text);p.chmod(0o400)
with (root/'payload.json').open('r') as stdin,(root/'stdout.json').open('x') as stdout,(root/'stderr.log').open('x') as stderr:
 p=subprocess.Popen(['python3','-u',str(root/'client.py')],stdin=stdin,stdout=stdout,stderr=stderr,cwd=root,start_new_session=True)
 ticks=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21]
handle={'task':'sglang-pd-spec-36','run':tag,'node':46,'pid':p.pid,'proc_start_ticks':ticks,'root':str(root),'client_root':str(clientroot),'client_sha256':plan['client_sha256'],'plan_sha256':j['plan_sha256'],'source_path':str(root/'client.py'),'status':'LAUNCHED_NOT_VERIFIED','host_monotonic_ns':time.monotonic_ns(),'real_workload':'3x32768->256 originalPD+MTP withtop2observation','no_replay':True};(root/'handle.json').write_text(json.dumps(handle,indent=2)+'\n');print(json.dumps(handle))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin);v={}
for node,lease in j['leases'].items():
 x=json.loads(subprocess.run(['kubectl','get','pod','-n','default',lease['name'],'-o','json'],capture_output=True,text=True,check=True,timeout=25).stdout);assert x['metadata']['uid']==lease['uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['spec']['nodeName']==f'a3-5u-atlas900rcka3-{node}' and x['status']['phase']=='Running' and next(c['status'] for c in x['status']['conditions'] if c['type']=='Ready')=='True'
def call(node,code,payload,timeout):return subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{node}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',code],input=json.dumps(payload),capture_output=True,text=True,timeout=timeout)
def one(item):
 p=call(item['node'],j['host'],{'item':item,'url':j['plan']['endpoints'][item['role']],'phase':'pre'},75);assert p.returncode==0,p.stderr;return str(item['node']),json.loads(p.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:v['pre']=dict(pool.map(one,j['targets']))
p=call(46,j['launcher'],{k:j[k] for k in ['plan','plan_text','plan_sha256','inputs','client']},40);v['launch']={'exit_code':p.returncode,'handle':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr};v['status']='LONG_PD_LAUNCHED_NOT_VERIFIED' if p.returncode==0 else 'FAIL';print(json.dumps(v));sys.exit(p.returncode)
'''
native=json.loads((R/'runs/L039-R086-pd-functional/remote-result.json').read_text());assert native['status']=='PASS_FUNCTIONAL_PAIR_ONLY' and len(native['client']['result']['target_top2'])==2 and all(x['all_chosen_at_returned_top2_max'] for x in native['client']['result']['target_top2'])
plan_text=(R/f'specs/{tag}-pd-plan.json').read_text();plan=json.loads(plan_text);inputs=[]
for item in plan['inputs']:
 data=(R/item['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256'];ids=json.loads(data)['input_ids'];assert len(ids)==32768 and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==item['wire_ids_sha256'];inputs.append({**item,'input_ids':ids})
client=(R/'scripts/L039_pd_long_client_v1.py').read_text();h=json.loads((R/'specs/L039-R077-targets.json').read_text());payload={'plan':plan,'plan_text':plan_text,'plan_sha256':hashlib.sha256(plan_text.encode()).hexdigest(),'inputs':inputs,'client':client,'host':HOST,'launcher':LAUNCH,'targets':[{**t,'role':'prefill' if t['node']==44 else 'decode'} for t in h['targets']],'leases':h['resource_leases']}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];started=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=190)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);v=json.loads(p.stdout) if p.stdout else {};receipt={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'status':v.get('status','UNKNOWN'),'native_long_functional_result':'PENDING','correctness':'NOT_VERIFIED','script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
