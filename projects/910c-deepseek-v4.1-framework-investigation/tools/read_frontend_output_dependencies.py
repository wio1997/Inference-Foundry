#!/usr/bin/env python3
"""Read exact ASGI/decoder ABI sources and model tokenizer metadata only."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from read_inactive_dp_profile_probe import ROOT

REMOTE=r'''
from pathlib import Path
from hashlib import sha256
import json,subprocess,sys
name='wio-vllm-v41-repro171-20261102';item,=json.loads(subprocess.check_output(['docker','inspect',name]));assert item['State']['Running'] and item['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner="""from pathlib import Path
from hashlib import sha256
import inspect,json,sys,sysconfig
roots=list(dict.fromkeys([Path(sysconfig.get_path('purelib')),*Path('/usr/local/lib').glob('python*/site-packages')]))
paths=[Path('/vllm-workspace/vllm/vllm/entrypoints/openai/completion/api_router.py'),Path('/vllm-workspace/vllm/vllm/entrypoints/serve/utils/sse_keep_alive.py')]
for rel in ['starlette/responses.py','uvicorn/protocols/http/httptools_impl.py','pydantic/main.py','tokenizers/decoders/__init__.py']:
 found=[r/rel for r in roots if (r/rel).is_file()];assert found,rel
 paths.append(found[0])
rows=[]
for p in paths:
 b=p.read_bytes();rows.append(dict(runtime_path=str(p),sha256=sha256(b).hexdigest(),content=b.decode()))
versions={}
for pkg in ['tokenizers','starlette','uvicorn','pydantic','pydantic_core']:
 for r in roots:
  ms=list(r.glob(pkg+'-*.dist-info/METADATA'))
  if ms:
   lines=ms[0].read_text().splitlines();versions[pkg]={k:next((s[len(k)+2:] for s in lines if s.startswith(k+': ')),None) for k in ['Name','Version']};break
p=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/tokenizer.json');assert p.is_relative_to('/data/tiankuan/wio');raw=p.read_bytes();d=json.loads(raw)
metadata=dict(path=str(p),sha256=sha256(raw).hexdigest(),bytes=len(raw),decoder=d.get('decoder'),normalizer=d.get('normalizer'),pre_tokenizer=d.get('pre_tokenizer'),model_type=d.get('model',{}).get('type'),added_special_token_count=sum(t.get('special',False) for t in d.get('added_tokens',[])),vocab_items=len(d.get('model',{}).get('vocab',{})))
root=next(r for r in roots if (r/'tokenizers/__init__.py').is_file());sys.path.insert(0,str(root))
from tokenizers.decoders import DecodeStream
try:sig=str(inspect.signature(DecodeStream))
except (ValueError,TypeError):sig=None
abi=dict(signature=sig,doc=DecodeStream.__doc__,class_methods=[k for k in dir(DecodeStream) if not k.startswith('__')])
print(json.dumps(dict(sources=rows,versions=versions,tokenizer_metadata=metadata,DecodeStream_ABI=abi,tokenizer_model_loaded=False)))
"""
d=json.loads(subprocess.check_output(['docker','exec',name,'python3','-B','-I','-S','-c',inner]))
root=Path(sys.argv[1]);assert root.is_relative_to('/data/tiankuan/wio')
d['completed_profile_candidate_files']=[str(p) for p in root.iterdir() if any(s in p.name.lower() for s in ['input','capture','client','request'])]
d.update(container_id=item['Id'],image=item['Image'],new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[])
print(json.dumps(d))
'''


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,ROOT],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout);out=Path('audit/execution-search')
    for row in d['sources']:
        content=row.pop('content');rp=row['runtime_path']
        rel=rp.removeprefix('/vllm-workspace/') if rp.startswith('/vllm-workspace/') else 'installed/'+rp.split('/site-packages/',1)[1]
        p=out/'frontend-output-runtime-source'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
        assert sha256(p.read_bytes()).hexdigest()==row['sha256'];row['local_path']=str(p)
    d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'frontend-output-extra-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in d.items() if k not in ['sources']},ensure_ascii=False))


if __name__=='__main__':main()
