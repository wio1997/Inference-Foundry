"""Read complete critical lines from the frozen native Meta log copy; no SDK/HTTP."""
import hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve();source=P.with_name('start_vllm_shared_pool_v3.py');assert hashlib.sha256(source.read_bytes()).hexdigest()=='f47203d5735ad6ce81c943a214fcf261d3179e30f566d5b037b1d7605bb17beb'
s=importlib.util.spec_from_file_location('pool',source);base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
RUN=base.b.ROOT/'runs/20261017T1759-vllm-shared-meta-mounts-v1'
HOST=r'''import pathlib,subprocess,json,hashlib,re
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID
out={'project_id':'glm53','node':30,'meta_container_id':CID,'files':[],'model_requests':0,'SDK_init_calls':0}
for item in CAPTURED:
 p=pathlib.Path(item['path']);assert str(p).startswith(REMOTE+'/live-meta-memcache-logs/') and p.is_file() and p.stat().st_size<=20*1024**2
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256'] and (not raw or raw.endswith(b'\n'))
 lines=[l for l in raw.decode().splitlines() if re.search(r'Mount bm on loc|Unmount|UnMount|ock\.mmc\.meta\.lease_ttl_ms\s*=|ock\.mmc\.evict_threshold_(?:high|low)\s*=',l)]
 assert len(lines)<=100
 out['files'].append({'path':str(p),'sha256':item['sha256'],'bytes':len(raw),'critical_lines_complete':lines})
print(json.dumps(out))
'''
def main():
 receipt=json.loads((base.RUN/'receipt.json').read_text());assert receipt['pool_kept_running']
 captured=json.loads(json.loads((base.RUN/'live-meta-snapshot.json').read_text())['stdout'])['files']
 meta=json.loads(json.loads((base.RUN/'meta-start.json').read_text())['stdout'])
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 r=base.invoke(30,base.replace(HOST,{'CID':meta['container_id'],'NAME':meta['name'],'IMAGE':base.IMAGE,'RID':base.RID,'REMOTE':base.REMOTE,'CAPTURED':captured}),100)
 (RUN/'node-30.json').write_text(json.dumps(r,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0},indent=2)+'\n')
if __name__=='__main__':main()
