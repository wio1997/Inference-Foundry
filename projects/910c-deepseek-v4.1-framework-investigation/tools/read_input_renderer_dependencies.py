#!/usr/bin/env python3
"""Read exact renderer/HF/executor dependencies and tokenizer metadata only."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import json,subprocess,sys,shlex
from hashlib import sha256
from pathlib import Path
name='wio-vllm-v41-repro171-20261102';item,=json.loads(subprocess.check_output(['docker','inspect',name]))
assert item['State']['Running'] and item['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner=r"""import json,sysconfig
from pathlib import Path
from hashlib import sha256
root=Path('/vllm-workspace/vllm/vllm');site=Path(sysconfig.get_path('purelib'))
paths=[root/x for x in ['renderers/hf.py','renderers/registry.py','tokenizers/registry.py','tokenizers/__init__.py','config/model.py','utils/async_utils.py']]
paths.extend(site/x for x in ['transformers/tokenization_utils_tokenizers.py','transformers/tokenization_utils_base.py'])
out=[]
for p in paths:
 if not p.is_file():out.append(dict(runtime_path=str(p),missing=True));continue
 raw=p.read_bytes();out.append(dict(runtime_path=str(p),sha256=sha256(raw).hexdigest(),content=raw.decode()))
model=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8');metadata=[]
allow=['tokenizer_class','add_bos_token','add_eos_token','bos_token','eos_token','pad_token','unk_token','model_max_length','clean_up_tokenization_spaces','split_special_tokens','tokenize_chinese_chars','do_lower_case','strip_accents','truncation_side','padding_side','legacy','errors','add_prefix_space']
for p in [model/'tokenizer_config.json',model/'tokenizer.json']:
 raw=p.read_bytes();d=json.loads(raw)
 selected={k:d[k] for k in allow if k in d}
 if p.name=='tokenizer.json':selected={k:d.get(k) for k in ['post_processor','truncation','padding','decoder','normalizer']}
 metadata.append(dict(path=str(p),sha256=sha256(raw).hexdigest(),bytes=len(raw),selected=selected))
versions={}
for pkg in ['transformers','tokenizers']:
 p,=list(site.glob(pkg+'-*.dist-info/METADATA'));lines=p.read_text().splitlines();versions[pkg]={k:next(s[len(k)+2:] for s in lines if s.startswith(k+': ')) for k in ['Name','Version']}
print(json.dumps(dict(sources=out,versions=versions,tokenizer_metadata=metadata)))
"""
r=subprocess.run(['docker','exec',name,'python3','-B','-I','-S','-c',inner],capture_output=True,text=True)
if r.returncode:print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
x=json.loads(r.stdout)
sp=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/scripts/serve.sh');b=sp.read_bytes();ts=shlex.split(b.decode().replace('\\\n',' '),comments=True)
opts={}
for i,s in enumerate(ts):
 if s.split('=')[0] in ['--renderer-num-workers','--tokenizer-mode','--enable-prompt-embeds','--tokenizer']:
  opts[s.split('=')[0]]=s.split('=',1)[1] if '=' in s else (ts[i+1] if i+1<len(ts) and not ts[i+1].startswith('--') else True)
exports={k:v for s in ts for k,sep,v in [s.partition('=')] if k in ['TOKENIZERS_PARALLELISM','RAYON_NUM_THREADS','OMP_NUM_THREADS']}
x.update(container_id=item['Id'],image=item['Image'],saved_launch=dict(path=str(sp),sha256=sha256(b).hexdigest(),selected_flags=opts,selected_exports=exports,renderer_worker_flag_present='--renderer-num-workers' in opts),selected_container_env={k:v for s in item['Config'].get('Env',[]) for k,sep,v in [s.partition('=')] if k in ['TOKENIZERS_PARALLELISM','RAYON_NUM_THREADS','OMP_NUM_THREADS']},CPU_tokenizer_loads=0,model_weights_loads=0,new_NPU_work=0,service_requests=0,new_profile_collection=0,remote_writes=[])
print(json.dumps(x))
'''


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
    x=json.loads(r.stdout);out=Path('audit/execution-search')
    for row in x['sources']:
        raw=row.pop('content',None)
        if raw is None:continue
        rp=row['runtime_path'];rel=rp.removeprefix('/vllm-workspace/') if rp.startswith('/vllm-workspace/') else 'installed/'+rp.split('/site-packages/',1)[1]
        p=out/'input-renderer-runtime-source'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(raw)
        assert sha256(p.read_bytes()).hexdigest()==row['sha256'];row['local_path']=str(p)
    x['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'input-renderer-extra-dependencies.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(x['sources']),missing=[r['runtime_path'] for r in x['sources'] if r.get('missing')],versions=x['versions'],metadata=x['tokenizer_metadata'],selected_container_env=x['selected_container_env'],new_NPU_work=0,remote_writes=[])))


if __name__=='__main__':main()
