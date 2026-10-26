"""One-shot owned CPU43 artifact recovery; no requests or remote mutations."""
from pathlib import Path
import ast,base64,gzip,hashlib,importlib.util,json,os,shlex,datetime
ROOT=Path(__file__).resolve().parents[1]
RID='20261018T-L09-interrupted-control-recovery-v2'
SETTINGS='config/C32-packed-normal-runtime-v1/formal-C32-L04-after-control-v1.json'
SHA='d6a9bbba88a63258e5d4c97f3afa28756251c07bac48a714364f283ed2bd1407'
HOST=r'''import pathlib,socket,subprocess,json,hashlib,gzip,base64,datetime
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L09-sglang-formal-C32-L04-after-control-v1')
assert root.is_dir() and root.resolve()==root
expected=EXPECTED
s=json.loads((root/'host-settings.json').read_text())
for k in ['project_id','framework','concurrency','namespace','run_id','container_name']:assert s[k]==expected[k]
for n,h in expected['files_sha256'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
out={'project_id':'glm53','node':43,'run_id':s['run_id'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_requests':0,'SDK_init_calls':0,'remote_mutations':False,'files':{},'artifacts':{}}
for p in sorted(root.rglob('*')):
 if p.is_file():out['files'][str(p.relative_to(root))]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
for n in ['controller-status.json','host-settings.json','output/formal-result.json','output/formal-progress.json','output/need-cache-gate.json','output/cache-gate.json']:
 p=root/n
 if p.exists():
  assert p.stat().st_size<=64*1024**2,('Artifact exceeds reviewed cap; retain without truncation',n)
  raw=p.read_bytes();out['artifacts'][n]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'gzip_base64':base64.b64encode(gzip.compress(raw,mtime=0)).decode()}
for n in ['client-stdout.txt','client-stderr.txt']:
 p=root/n
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-65536));tail=f.read()
  out[n]={'bytes':p.stat().st_size,'bounded_tail_utf8':tail.decode(errors='replace'),'tail_limit':65536,'original_complete_remote_file_preserved':True}
r=subprocess.run(['docker','inspect',s['container_name'],'--format','{{.Id}} {{.Name}} {{.Image}} {{json .State}}'],capture_output=True,text=True,timeout=30)
out['owned_client_container_inspect']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
p=root/'controller-status.json'
if p.exists():
 st=json.loads(p.read_text());pid=st.get('host_docker_client_pid')
 if isinstance(pid,int) and pid>1:
  r=subprocess.run(['ps','-p',str(pid),'-o','pid=,ppid=,stat=,comm='],capture_output=True,text=True,timeout=15);out['owned_docker_client_process']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(out))
'''
def main():
 s=ROOT/SETTINGS;assert hashlib.sha256(s.read_bytes()).hexdigest()==SHA;cfg=json.loads(s.read_text())
 p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
 spec=importlib.util.spec_from_file_location('inventory',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);fd=os.open(run/'execution-once.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
 expected={k:cfg[k] for k in ['project_id','framework','concurrency','namespace','run_id']};expected['container_name']=cfg['client_container_name'];expected['files_sha256']=cfg['client_files_sha256']
 code=HOST.replace('EXPECTED',repr(expected));ast.parse(code);(run/'host-source.py').write_text(code)
 pod=m.NODES[43][0]
 mapping=m.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);(run/'mapping.json').write_text(json.dumps(mapping,indent=2)+'\n');assert mapping['rc']==0 and mapping['stdout']=='172.16.12.43'
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 out=m.invoke([command],code,timeout=180);(run/'node-43.json').write_text(json.dumps(out,indent=2)+'\n');assert out['rc']==0
 d=json.loads(out['stdout']);assert d['project_id']=='glm53' and d['run_id']==cfg['run_id']
 for n,item in d['artifacts'].items():
  raw=gzip.decompress(base64.b64decode(item['gzip_base64'],validate=True));assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];dest=run/'recovered'/n;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(raw)
 report=run/'recovered/output/formal-result.json';v=json.loads(report.read_text()) if report.exists() else {}
 result={'project_id':'glm53','pass':True,'as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':cfg['run_id'],'formal_result_exists':report.exists(),'phase':v.get('phase'),'baseline_accepted':v.get('baseline_accepted'),'backend_HTTP_attempts':v.get('backend_HTTP_attempts'),'warmup_count':len(v.get('warmup',[])),'warmup_passes':sum(x.get('pass') is True for x in v.get('warmup',[])),'formal_present':v.get('formal') is not None,'error':v.get('error'),'owned_client_inspect_rc':d['owned_client_container_inspect']['rc'],'model_requests':0,'remote_mutations':False}
 (run/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
