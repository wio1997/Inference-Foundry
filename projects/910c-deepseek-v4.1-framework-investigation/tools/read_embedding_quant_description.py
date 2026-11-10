#!/usr/bin/env python3
"""Read only descriptor JSON and emit embedding method selectors, never weights."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
ROOT=Path('audit/execution-search')
REMOTE=r'''
import io,json,subprocess,tarfile
from hashlib import sha256
name='wio-vllm-v41-repro171-20261102';c,=json.loads(subprocess.check_output(['docker','inspect',name]));assert c['Id']=='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb' and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
path='/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/quant_model_description.json';p=subprocess.run(['docker','cp',name+':'+path,'-'],capture_output=True,timeout=30);assert p.returncode==0
with tarfile.open(fileobj=io.BytesIO(p.stdout)) as t:
 a=[x for x in t.getmembers() if x.isfile()];assert len(a)==1 and a[0].size<20000000;b=t.extractfile(a[0]).read()
d=json.loads(b);selected={k:v for k,v in d.items() if 'embed_tokens' in k or k.startswith('embed.') or '.embed.' in k or k in {'quantization_type','model_quant_type','kv_cache_type'}}
assert all(isinstance(v,(str,int,bool,float,type(None))) for v in selected.values())
print(json.dumps(dict(container_id=c['Id'],image=c['Image'],path=path,sha256=sha256(b).hexdigest(),bytes=len(b),total_keys=len(d),selected_description=selected,has_hc_head_fn='hc_head_fn' in d,qualification='Descriptor metadata only; no checkpoint tensor, imported framework or loaded-object evidence.',new_NPU_work=0,service_requests=0,remote_writes=[])))
'''
def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=65)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout);d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();(ROOT/'embedding-quant-description.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sha256=d['sha256'],bytes=d['bytes'],selected=d['selected_description'],new_NPU_work=0)))
if __name__=='__main__':main()
