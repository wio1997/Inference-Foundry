#!/usr/bin/env python3
"""Read exact current resource objects after full-bootstrap preflight refusal; no service mutation."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R037-bootstrap-gate';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,pathlib,re,subprocess,sys
j=json.load(sys.stdin);base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36'
p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);q=subprocess.run(['docker','ps','-a','--filter','label=task=sglang-pd-spec-36','--format','{{json .}}'],capture_output=True,text=True,timeout=25)
print(json.dumps({'node':j['node'],'NPU_inventory_exit':p.returncode,'NPU_inventory_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'NPU_no_running_processes':p.stdout.count('No running processes found in NPU')==8,'NPU_HBM_used_MiB':list(map(int,re.findall(r'(\d+)\s*/\s*65536',p.stdout))),'NPU_Health_rows':[line for line in p.stdout.splitlines() if 'Ascend910' in line],'own_containers_exit':q.returncode,'own_containers':[{'ID':x['ID'],'Names':x['Names'],'State':x['State'],'Status':x['Status']} for x in [json.loads(s) for s in q.stdout.splitlines()]],'R036_root_exists':(base/('L039-R036-prefill' if j['node']==44 else 'L039-R036-decode')).exists(),'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin);v={'pods':{},'nodes':{},'hosts':{}}
for node,lease in j['leases'].items():
 p=subprocess.run(['kubectl','get','pod','-n','default',lease['name'],'-o','json'],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);v['pods'][node]={'name':x['metadata']['name'],'uid':x['metadata']['uid'],'expected_uid':lease['uid'],'labels':x['metadata']['labels'],'deletionTimestamp':x['metadata'].get('deletionTimestamp'),'spec_node':x['spec'].get('nodeName'),'resources':x['spec']['containers'][0]['resources'],'status':x['status']}
 p=subprocess.run(['kubectl','get','node',f'a3-5u-atlas900rcka3-{node}','-o','json'],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);v['nodes'][node]={'uid':x['metadata']['uid'],'taints':x['spec'].get('taints',[]),'conditions':x['status']['conditions'],'nodeInfo':x['status']['nodeInfo'],'capacity':x['status']['capacity'],'allocatable':x['status']['allocatable']}
def host(node):
 p=subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{node}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps({'node':int(node)}),capture_output=True,text=True,timeout=70);return node,{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:v['hosts']=dict(pool.map(host,['44','46']))
print(json.dumps(v));sys.exit(0 if all(x['exit_code']==0 for x in v['hosts'].values()) else 1)
'''
plan=json.loads((R/'specs/L039-full-bootstrap-plan-v1.json').read_text());cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':HOST,'leases':plan['resource_leases']}),capture_output=True,text=True,timeout=160)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
