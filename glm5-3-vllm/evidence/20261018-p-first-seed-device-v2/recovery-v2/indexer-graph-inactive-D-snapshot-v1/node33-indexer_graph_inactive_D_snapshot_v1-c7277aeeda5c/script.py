from pathlib import Path
import json,socket,subprocess,shutil,gzip,hashlib
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live1-20261019/node33');cid='3caaa2a7856b78e27a692a909505d9d4a34969b68217c64869e5f4397bc25b5b'
plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['Id']==json.loads((root/'launch.json').read_text())['container_id'] and d['Name']=='/'+plan['container']['name'] and d['State']['Running']
assert not(root/'indexer-k-graph-overlap-probe/enable').exists()
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False);files=[]
for p in root.rglob('*'):
 if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.jsonl','.py'] and p.relative_to(root).parts[0]!='profile':files.append(p)
assert sum(p.stat().st_size for p in files)<256*1024**2
for p in files:
 q=out/'model-root'/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
for patch in plan['runtime_patches']:
 b=Path(patch['source']).read_bytes();assert hashlib.sha256(b).hexdigest()==patch['sha256'];q=out/'overlay-sources'/Path(patch['source']).name;q.parent.mkdir(exist_ok=True);q.write_bytes(b)
q=subprocess.run(['docker','logs','--timestamps',cid],capture_output=True,timeout=30)
with gzip.open(out/'model-console.txt.gz','wb') as f:f.write(q.stdout+q.stderr)
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'completed':True,'CID':cid,'files':len(files),'bytes':sum(p.stat().st_size for p in files),'state':d['State'],'scope':'Immutable source, functional native-fallback evidence, all16 inactive logs and model console before exact owned D replacement. No quality/formal started.'})+'\nGLM53_RESULT_JSON_END')
