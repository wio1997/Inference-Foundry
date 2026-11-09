#!/usr/bin/env python3
"""CPU-only replay of original renderer methods and actual installed HF calls."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

INNER=r'''
from pathlib import Path
from hashlib import sha256
import sys,sysconfig,json,ast,copy,contextlib,queue,dataclasses,typing,os
from types import SimpleNamespace as S
sys.path.insert(0,sysconfig.get_path('purelib'))
# Isolated process only. Potential library caches remain inside the owned root.
cache='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/cpu-input-renderer-cache'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=cache+'/hf',XDG_CACHE_HOME=cache+'/xdg',USE_TORCH='0',USE_TF='0',USE_FLAX='0',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
before={str(p) for p in Path(cache).rglob('*')} if Path(cache).exists() else set()
data=json.load(sys.stdin);raw=data['content'].encode();assert sha256(raw).hexdigest()=='7863f18d0f0ac87fd158b773f181b9abb81a528245ed3e1d8a1dfcbcdf4dc404'
rows=[json.loads(s) for s in raw.decode().splitlines()];assert len(rows)==8
from transformers import TokenizersBackend
from tokenizers import Tokenizer
model=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8')
config=json.loads((model/'tokenizer_config.json').read_bytes());config.pop('tokenizer_class',None)
config=TokenizersBackend.convert_added_tokens(config,save=False)
# The fixed config names the generic fast backend; build that backend directly
# to avoid AutoConfig traversing model/NPU integrations. This does not replay
# AutoTokenizer class resolution or claim loaded-service object identity.
hf=TokenizersBackend(tokenizer_file=str(model/'tokenizer.json'),**config)
assert isinstance(hf,TokenizersBackend)
native=Tokenizer.from_file(str(model/'tokenizer.json'))
pins=[]
def original(rel,names):
 p=Path('/vllm-workspace/vllm/vllm')/rel;b=p.read_bytes();pins.append(dict(runtime_path=str(p),sha256=sha256(b).hexdigest()))
 tree=ast.parse(b);ns=tree.body
 chosen=[]
 for cls,name in names:
  scope=ns if cls is None else next(n.body for n in ns if isinstance(n,ast.ClassDef) and n.name==cls)
  n=copy.deepcopy(next(n for n in scope if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name))
  if isinstance(n,ast.FunctionDef):n.decorator_list=[]
  chosen.append(n)
 return compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*chosen],type_ignores=[])),str(p),'exec')
env=dict(copy=copy,contextlib=contextlib,queue=queue,TokenizersBackend=TokenizersBackend)
exec(original('tokenizers/hf.py',[(None,n) for n in ['ThreadSafeHFTokenizerMixin','get_cached_tokenizer','maybe_make_thread_pool']]),env)
cached=env['get_cached_tokenizer'](hf);pooled=env['maybe_make_thread_pool'](cached,2)
base=Path('/vllm-workspace/vllm/vllm/renderers/params.py');b=base.read_bytes();tree=ast.parse(b);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TokenizeParams')
constants=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_PARALLEL_TO_PROMPT_TOKENS' for t in n.targets)]
class FixtureValidationError(ValueError):
 def __init__(self,message,**kwargs):super().__init__(message)
ev=dict(dataclass=dataclasses.dataclass,cast=typing.cast,VLLMValidationError=FixtureValidationError,logger=S(warning=lambda *a:None))
mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*constants,cls],type_ignores=[])
exec(compile(ast.fix_missing_locations(mod),str(base),'exec'),ev);pins.append(dict(runtime_path=str(base),sha256=sha256(b).hexdigest()))
exec(original('entrypoints/openai/completion/protocol.py',[('CompletionRequest','build_tok_params')]),ev)
ev['TokensPrompt']=dict
exec(original('renderers/base.py',[('BaseRenderer','_build_tokens_prompt'),('BaseRenderer','_tokenize_prompt')]),ev)
renderer=S(get_tokenizer=lambda:pooled,_wants_offsets=lambda *a:False,_build_tokens_prompt=ev['_build_tokens_prompt'])
result=[];groups={};all_ids=[]
for i,r in enumerate(rows):
 text=r.get('question',r.get('prompt'));assert isinstance(text,str)
 req=S(max_tokens=r.get('max_tokens',1024),truncate_prompt_tokens=None,truncation_side=None,add_special_tokens=True,echo=False,return_token_ids=False,return_token_offsets=False)
 params=ev['build_tok_params'](req,S(max_model_len=34816))
 prompt=params.apply_pre_tokenization(pooled,dict(prompt=text));out=ev['_tokenize_prompt'](renderer,prompt,params);out=params.apply_post_tokenization(pooled,out)
 ids=out['prompt_token_ids'];nids=native.encode(text,add_special_tokens=True).ids
 groups.setdefault(str(r.get('dp_rank','all')),[]).append(i);all_ids.append(ids)
 result.append(dict(index=i,dp_rank=r.get('dp_rank'),text_chars=len(text),IDs=len(ids),IDs_equal_direct_native=ids==nids,IDs_sha256=sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
   kwargs=params.get_encode_kwargs(),pre_text_unchanged=prompt['prompt']==text,post_ids_unchanged=len(ids)==len(nids),return_offsets=False,needs_detokenization=params.needs_detokenization))
# Decision-changing cache metadata only; no proposed replacement is executed.
meta=json.loads((model/'tokenizer.json').read_bytes());added=meta['added_tokens'];contents=[a['content'] for a in added];partials={c[:j] for c in contents for j in range(1,len(c))}
groups['global_all8']=list(range(8));prefixes=[]
for group,ii in groups.items():
 texts=[rows[i].get('question',rows[i].get('prompt')) for i in ii];prefix=os.path.commonprefix(texts)
 valid=[]
 for k in range(3,len(prefix)+1):
  tail=prefix[k-3:k]
  if not (tail.isascii() and tail.isdecimal() and (k==3 or prefix[k-4].isascii() and not prefix[k-4].isdigit())):continue
  if any(prefix[max(0,k-j):k] in partials for j in range(1,min(k,max(map(len,contents))))):continue
  valid.append(k)
 cut=valid[-1] if valid else None
 prefixes.append(dict(group=group,indices=ii,shared_chars=len(prefix),cut_numeric_ASCII3=cut,
  suffix_chars=None if cut is None else [len(s)-cut for s in texts],prefix_sha256=sha256(prefix.encode()).hexdigest()))
after={str(p) for p in Path(cache).rglob('*')} if Path(cache).exists() else set()
assert 'torch_npu' not in sys.modules
print('INPUT_RENDERER_RESULT='+json.dumps(dict(status='PASS_SOURCE_REPLAY' if all(r['IDs_equal_direct_native'] for r in result) else 'MISMATCH',
 rows=result,prefix_metadata=prefixes,added_token_flags={k:sum(bool(a.get(k)) for a in added) for k in ['lstrip','rstrip','single_word','normalized']},
 added_tokens=len(added),BPE_dropout=meta['model'].get('dropout'),pre_tokenizer=meta['pre_tokenizer'],
 source_pins=pins,HF_loaded_type=type(hf).__name__,pooled_type=type(pooled).__name__,
 scope='Original renderer methods plus installed generic HF backend constructed from fixed tokenizer.json/config, not AutoTokenizer model-class resolution; validation-error class inert and no errors asserted. Actual serving process IDs/bytecode not captured. Prefix metadata is not a cache implementation or equality proof.',
 CPU_tokenizer_original_loads=2,native_deep_copies_in_pool=2,torch_imported='torch' in sys.modules,torch_npu_imported=False,model_weights_loads=0,new_NPU_work=0,service_requests=0,new_profile_collection=0,remote_cache_created=sorted(after-before),service_changes=[])))
'''

REMOTE=r'''
from pathlib import Path
from hashlib import sha256
import json,subprocess,sys
name='wio-vllm-v41-repro171-20261102';item,=json.loads(subprocess.check_output(['docker','inspect',name]));assert item['State']['Running'] and item['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
ip,=list(Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/benchmark-20261028/workload/frozen/measure1').rglob('*.requests.jsonl'));b=ip.read_bytes();assert sha256(b).hexdigest()=='7863f18d0f0ac87fd158b773f181b9abb81a528245ed3e1d8a1dfcbcdf4dc404'
r=subprocess.run(['docker','exec','-i',name,'python3','-B','-I','-S','-c',sys.argv[1]],input=json.dumps(dict(path=str(ip),content=b.decode())),capture_output=True,text=True)
if r.returncode:print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
lines=r.stdout.splitlines();result=[s.removeprefix('INPUT_RENDERER_RESULT=') for s in lines if s.startswith('INPUT_RENDERER_RESULT=')];assert len(result)==1
x=json.loads(result[0]);extra='\n'.join(s for s in lines if not s.startswith('INPUT_RENDERER_RESULT='));x.update(container_id=item['Id'],image=item['Image'],library_extra_stdout_lines=len(lines)-1,library_extra_stdout_sha256=sha256(extra.encode()).hexdigest());print(json.dumps(x))
'''


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr[-3500:])
    x=json.loads(r.stdout);x['probe_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    Path('audit/execution-search/input-renderer-ID-replay.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in x.items() if k not in ['pre_tokenizer','source_pins']},ensure_ascii=False))


if __name__=='__main__':main()
