"""Render host identity/ownership guards for the frozen vLLM1757 shared containers."""
import json, re

HOST = r'''import pathlib,subprocess,json,hashlib,re,socket
def command(argv,timeout=30):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
 assert r.returncode==0,(argv,r.stderr)
 return r.stdout
d=json.loads(command(['docker','inspect',CID]))[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE
assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID
assert d['State']['Running'] and not d['HostConfig']['Privileged']
root=pathlib.Path(REMOTE)
supervisor=json.loads((root/'supervisor-status.json').read_text())
provider=json.loads((root/'provider-status.json').read_text())
assert supervisor['run_id']==RID and supervisor['phase']=='POOL_READY_WAITING_MODEL_SIGNAL' and supervisor['model_started'] is False
assert provider['phase']=='READY' and provider['configuration']['node']==NODE and provider['pid']==supervisor['provider_pid']
assert provider['configuration']['dram_size']==DRAM and provider['configuration']['max_dram_size']=='192GB'
assert provider['configuration']['world_size']==128 and provider['configuration']['protocol']=='device_sdma'
for relative,item in FILES.items():assert hashlib.sha256((pathlib.Path(STAGED)/relative).read_bytes()).hexdigest()==item['sha256']
top=command(['docker','top',CID,'-eo','pid,args']);candidates=[]
for row in top.splitlines()[1:]:
 fields=row.split(None,1)
 if len(fields)==2 and fields[0].isdigit() and '/opt/glm53/bundle/provider.py' in fields[1]:
  pid=int(fields[0]);status=pathlib.Path('/proc')/str(pid)/'status'
  nspids=[line.split()[1:] for line in status.read_text().splitlines() if line.startswith('NSpid:')]
  assert len(nspids)==1 and int(nspids[0][-1])==provider['pid']
  cgroup=(pathlib.Path('/proc')/str(pid)/'cgroup').read_text();assert CID in cgroup
  candidates.append(pid)
assert len(candidates)==1,('Cannot map exact SDK child host PID',top)
provider_host_pid=candidates[0]
def check_npu():
 raw=command(['npu-smi','info']);rows=[]
 for line in raw.split('Process id',1)[-1].splitlines():
  columns=[item.strip() for item in line.split('|')]
  if len(columns)<5:continue
  card_chip=columns[1].split()
  if len(card_chip)==2 and all(x.isdigit() for x in card_chip) and columns[2].isdigit():rows.append((int(card_chip[0]),int(card_chip[1]),int(columns[2])))
 assert rows and {row[2] for row in rows}=={provider_host_pid}
 assert len({row[0] for row in rows})+raw.count('No running processes')==8,raw
 return {'provider_host_pid':provider_host_pid,'rows':rows,'empty_cards':raw.count('No running processes'),'raw_sha256':hashlib.sha256(raw.encode()).hexdigest()}
npu_before=check_npu()
'''

def render(node, action, capacity, files, staged):
    item = capacity['providers'][str(node)]
    values = {'CID': item['container_id'], 'NAME': item['name'], 'NODE': node,
              'IMAGE': 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83',
              'RID': '20261017T1757-vllm-shared-pool-v3',
              'REMOTE': '/data/wio/inference-foundry/glm53/runs/20261017T1757-vllm-shared-pool-v3',
              'FILES': files, 'STAGED': staged, 'DRAM': {30:'192GB',31:'192GB',43:'128GB',45:'56GB'}[node]}
    text = HOST + action
    for key, value in values.items():
        text = re.sub(r'\b' + key + r'\b', lambda _: repr(value), text)
    return text
