"""Once-only exact-image CPU qualification with no NPU visibility or SDK calls."""
import ast,base64,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261017T2022-sglang-buffer-dedup-CPU-v1'
MANIFEST='config/diagnostic-runs-v1/sglang-buffer-dedup-CPU-v1.json'
MANIFEST_SHA='9f9fd622b75eed7696620782f8e4cbac09714ebd4660c2669ce384483c62e222'
def load(name,path,sha):
 p=ROOT/path;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('inventory','scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
codec=load('codec','scripts/glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
HOST=r'''import pathlib,subprocess,json,hashlib,gzip,base64,socket
assert socket.gethostname()==HOSTNAME
def cmd(args,timeout=30):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout);return {'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
image=cmd(['docker','image','inspect',IMAGE,'--format','{{.Id}}']);assert image['rc']==0 and image['stdout'].strip()==IMAGE
archive=pathlib.Path(OUTDIR);assert not archive.exists();archive.mkdir(parents=True)
assert cmd(['docker','inspect',NAME])['rc']!=0
for relative,item in FILES.items():
 path=archive/relative;assert path.resolve().is_relative_to(archive.resolve());path.parent.mkdir(parents=True,exist_ok=True)
 raw=base64.b64decode(item['data']);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];path.write_bytes(raw)
args=['docker','run','--pull=never','--network=none','--cpus=2','--memory=2g','--pids-limit=128','--label','project_id=glm53','--label','run_id='+RID,'--name',NAME,'-v',str(archive)+':/run/glm53','-e','PYTHONPATH=/run/glm53/python','-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','python3',IMAGE,'/run/glm53/driver.py']
try:executed=cmd(args,280)
except subprocess.TimeoutExpired:
 inspect=cmd(['docker','inspect',NAME]);assert inspect['rc']==0;info=json.loads(inspect['stdout'])[0];assert info['Config']['Labels']['project_id']=='glm53' and info['Config']['Labels']['run_id']==RID and info['Image']==IMAGE
 executed={'rc':124,'stop':cmd(['docker','stop','--time','10',info['Id']],20)}
inspect=cmd(['docker','inspect',NAME]);assert inspect['rc']==0;info=json.loads(inspect['stdout'])[0];assert info['Image']==IMAGE and not info['State']['Running'] and info['HostConfig']['NetworkMode']=='none' and not info['HostConfig']['Devices'] and not info['HostConfig'].get('DeviceRequests')
out={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'NPU_devices':False,'execution':executed,'container':{k:info[k] for k in ['Id','Name','Image','State']},'network_mode':'none','source_files':{k:{key:value for key,value in item.items() if key!='data'} for k,item in FILES.items()},'private_logs':[]}
result=archive/'result.json'
if result.is_file():out['result']=json.loads(result.read_text())
for path in archive.glob('*.txt'):
 raw=path.read_bytes();assert len(raw)<=4*1024**2;dest=path.with_suffix(path.suffix+'.gz');dest.write_bytes(gzip.compress(raw,mtime=0));out['private_logs'].append({'path':str(path),'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
'''
def render(manifest,files):
 values={'HOSTNAME':base.NODES[34][1],'IMAGE':manifest['image'],'OUTDIR':'/data/wio/inference-foundry/glm53/runs/'+RID,'NAME':'glm53-buffer-dedup-CPU-v1-20261017-34','RID':RID,'FILES':files}
 code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),HOST)+codec.ENCODER;ast.parse(code);return code
def main():
 path=ROOT/MANIFEST;assert hashlib.sha256(path.read_bytes()).hexdigest()==MANIFEST_SHA;manifest=json.loads(path.read_text());assert manifest['run_id']==RID and manifest['project_id']=='glm53'
 files={}
 for relative,item in manifest['files'].items():
  assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
  raw=(ROOT/item['local']).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];files[relative]={'data':base64.b64encode(raw).decode(),'sha256':item['sha256'],'bytes':len(raw)}
 assert sum(item['bytes'] for item in files.values())<=512*1024
 prior=json.loads((ROOT/'runs/20261017T2020-sglang-memcache-LRU-native-v2/receipt.json').read_text());assert prior['nodes']=={'34':0}
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'manifest_sha256':MANIFEST_SHA,'model_requests':0,'SDK_init_calls':0,'nodes':{}}
 try:
  pod,_=base.NODES[34];mapping=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapping['rc']==0 and mapping['stdout']=='172.16.12.34'
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);wire=base.invoke([command],render(manifest,files),timeout=330);result=codec.decode(wire) if wire['rc']==0 else wire
  (run/'node-34.json').write_text(json.dumps(result,indent=2)+'\n');receipt['nodes']['34']=result['rc'];print('CPU_QUALIFICATION_CAPTURE_RC='+str(result['rc']),flush=True)
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
