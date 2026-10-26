"""CPU-only original renderer vs bounded input-token cache semantic gate."""
import concurrent.futures,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,'/data/wio/Inference_Foundry')
import vllm_ascend.patch.platform.patch_deepseek_v4_thinking
from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.base import BaseRenderer
from vllm.renderers.params import TokenizeParams
from experiments.renderer_cache.adapter import cached_tokenize_prompt
from experiments.renderer_cache.token_cache import TextTokenCache
root=Path('/data/wio/Inference_Foundry');data=Path('/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl')
tok=DeepseekV4Tokenizer.from_pretrained('/data/yxy/DeepSeek-V4-Flash-0731-w4a8')
r=DeepseekV4Renderer.__new__(DeepseekV4Renderer);r.tokenizer=tok
params=TokenizeParams(max_total_tokens=1048576,add_special_tokens=False)
prompts=[]
for line in data.read_text().splitlines()[:48]:
 m=[{'role':'user','content':json.loads(line)['question']}]
 prompts.append({'prompt':tok.apply_chat_template(messages=m,conversation=m,tokenize=False)})
original=BaseRenderer._tokenize_prompt
ref=[original(r,p,params) for p in prompts]
assert min(len(x['prompt_token_ids']) for x in ref)==32851
st=time.perf_counter();cold=[cached_tokenize_prompt(original,r,p,params) for p in prompts];cold_s=time.perf_counter()-st
assert cold==ref
st=time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 hot=list(ex.map(lambda p:cached_tokenize_prompt(original,r,p,params),prompts*4))
hot_s=time.perf_counter()-st
assert hot==ref*4
hot[0]['prompt_token_ids'][0]=-999
assert cached_tokenize_prompt(original,r,prompts[0],params)==ref[0]
extra=dict(prompts[0],cache_salt='per-request-test');out=cached_tokenize_prompt(original,r,extra,params);assert out['cache_salt']=='per-request-test' and out['prompt_token_ids']==ref[0]['prompt_token_ids']
# Actual changed encoder options have independent keys and original parity.
for special in (False,True):
 p=TokenizeParams(max_total_tokens=1048576,add_special_tokens=special)
 assert cached_tokenize_prompt(original,r,prompts[0],p)==original(r,prompts[0],p)
for n in (32,64):
 kw={'add_special_tokens':False,'truncation':True,'max_length':n}
 assert r._foundry_text_token_cache.encode(prompts[0]['prompt'],kw,tok)==tuple(tok(prompts[0]['prompt'],**kw)['input_ids'])
# Eviction and oversized entries recompute instead of reusing stale IDs.
calls=[]
def fake(text,**kw): calls.append(text);return {'input_ids':[ord(x) for x in text]}
c=TextTokenCache(max_entries=1,max_bytes=4096)
for t in ('a','b','a'):assert c.encode(t,{},fake)==(ord(t),)
assert calls==['a','b','a']
calls.clear();c=TextTokenCache(max_entries=1,max_bytes=1)
c.encode('a',{},fake);c.encode('a',{},fake);assert calls==['a','a'] and not c._rows
# The adapter must retain the complete offset path, not just IDs.
class OffsetRenderer:
 def _wants_offsets(self,p,q):return True
sentinel={'prompt_token_ids':[7],'prompt_token_offsets':[(0,1)]}
assert cached_tokenize_prompt(lambda *a:sentinel,OffsetRenderer(),{},None) is sentinel
cache=r._foundry_text_token_cache
assert cache._bytes<=cache.max_bytes and len(cache._rows)<=cache.max_entries
result={'status':'pass','scope':'CPU original renderer encoding and per-request object parity; not serving performance or model output parity','dataset_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'template_function':tok.apply_chat_template.__code__.co_filename,'all48_prompt_token_hashes':[hashlib.sha256(json.dumps(x['prompt_token_ids'],separators=(',',':')).encode()).hexdigest() for x in ref],'cold48_s':cold_s,'hot192_s':hot_s,'cache_entries':len(cache._rows),'estimated_cache_bytes':cache._bytes,'gates':['all48 cold exact','all192 threaded hits exact','caller mutation isolation','request extras fresh','encoding options exact','LRU eviction','oversize bypass','complete offset bypass']}
(root/'evidence/20260929_loop081_bound/run671/cpu_gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
