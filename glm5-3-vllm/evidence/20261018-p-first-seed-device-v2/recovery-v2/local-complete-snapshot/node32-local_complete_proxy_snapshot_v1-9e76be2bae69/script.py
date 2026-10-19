import json,socket,subprocess,pathlib,hashlib,gzip,shutil,urllib.request
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
root=pathlib.Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live3-20261018/node32');local=root/'local-complete-proxy-v1';launch=json.loads((local/'launch.json').read_text());cid=launch['CID'];assert cid=='cfa1eade67ed5dda23ab25e599a5f6c86dfcc2f0b179770886f812ce7b1a7579'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and not d['HostConfig'].get('Devices') and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
with urllib.request.urlopen('http://172.16.12.32:29882/healthcheck',timeout=5) as response:health=json.load(response)
assert health['request_num']==0
out=pathlib.Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
for name in ['proxy.py','launch.json','result.json']:
 shutil.copy2(local/name,out/name)
assert hashlib.sha256((out/'proxy.py').read_bytes()).hexdigest()==launch['source_sha256']
logs=subprocess.check_output(['docker','logs','--timestamps',cid],stderr=subprocess.STDOUT,timeout=20);assert len(logs)<32*1024**2
with gzip.open(out/'console.txt.gz','wb') as f:f.write(logs)
code="""import json,pathlib,hashlib
files=[]
paths=list(pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/worker').glob('*.py'))+list(pathlib.Path('/vllm-workspace/vllm/vllm/profiler').rglob('*.py'))+[pathlib.Path('/vllm-workspace/vllm/vllm/config/profiler.py')]
for p in paths:
 if p.is_file() and 'model_runner' not in p.name:
  text=p.read_text()
  if 'def profile(' in text or 'torch_profiler_record_shapes' in text or 'def start(' in text and 'profiler' in str(p):
   assert len(text)<200000
   files.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':text})
print(json.dumps({'files':files}))
"""
source=json.loads(subprocess.check_output(['docker','exec','-i','4cb8a338da590f0d0fd624892b2af7a20884029c7418c05ada75f5dbf3bcb157','python3','-'],input=code.encode(),timeout=25));(out/'native-profiler-source.json').write_text(json.dumps(source,indent=2)+'\n')
report={'local_proxy_CID':cid,'source_sha256':launch['source_sha256'],'model_kept_running':True,'proxy_kept_running':True,'health':health,'state':d['State'],'console_bytes':len(logs),'native_profiler_source_files':[(f['path'],f['sha256']) for f in source['files']],'scope':'Own finite CPU proxy artifact snapshot and read-only native profiler source. No model, operator, trace settings or other workload changed.'}
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
