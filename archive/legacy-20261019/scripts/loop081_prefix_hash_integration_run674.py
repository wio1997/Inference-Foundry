"""CPU-only initial block-hash cost screen; no serving source changes."""
import json,time,statistics,hashlib,socket
from pathlib import Path
from types import SimpleNamespace

def not_serving():
 s=socket.socket();s.settimeout(1);v=s.connect_ex(('127.0.0.1',8080));s.close()
 if v==0:raise RuntimeError('service ready: refuse competing CPU screen')
not_serving()
import vllm_ascend.patch.platform.patch_deepseek_v4_thinking
from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
from vllm.v1.core.kv_cache_utils import get_request_block_hasher,init_none_hash
from vllm.utils.hashing import sha256
init_none_hash(sha256)
tok=DeepseekV4Tokenizer.from_pretrained('/data/yxy/DeepSeek-V4-Flash-0731-w4a8')
p=Path('/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl')
ids=[]
for line in p.read_text().splitlines()[:48]:
 m=[{'role':'user','content':json.loads(line)['question']}]
 text=tok.apply_chat_template(messages=m,conversation=m,tokenize=False)
 ids.append(tok(text,add_special_tokens=False)['input_ids'])

import sys
sys.path.insert(0,'/data/wio/Inference_Foundry')
from experiments.prefix_hash_cache.cache import InitialPrefixHashMemo
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams
import concurrent.futures
from vllm.v1.core import kv_cache_utils as _kv
assert hasattr(_kv, "_foundry_original_hasher_factory")
original=_kv._foundry_original_hasher_factory(2,sha256)
memo=get_request_block_hasher(2,sha256)
assert isinstance(memo,InitialPrefixHashMemo)
def request(tokens,hasher,salt=None):
 return Request('gate',tokens,SamplingParams(max_tokens=1024),None,block_hasher=hasher,cache_salt=salt)
not_serving();start=time.perf_counter();ref=[request(x,original).block_hashes for x in ids];ref_s=time.perf_counter()-start
not_serving();start=time.perf_counter();cold=[request(x,memo).block_hashes for x in ids];cold_s=time.perf_counter()-start
assert cold==ref
not_serving();start=time.perf_counter();hot=[request(x,memo).block_hashes for x in ids];hot_s=time.perf_counter()-start
assert hot==ref
hot[0][0]=b'caller mutation'
assert request(ids[0],memo).block_hashes==ref[0]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 hits=list(ex.map(lambda x:request(x,memo).block_hashes,ids))
assert hits==ref
# Incremental continuation must preserve exact chained hashes, including partial prompt tail.
for x in (ids[0],ids[-1]):
 a=request(x,original);b=request(x,memo)
 for token in (7,13,17,19):
  a.append_output_token_ids(token);b.append_output_token_ids(token)
  assert a.block_hashes==b.block_hashes
for salt in ('a','b'):
 assert request(ids[0],memo,salt).block_hashes==request(ids[0],original,salt).block_hashes
typed=InitialPrefixHashMemo(original)
for tokens in ([1,2],[True,2],[1.0,2]):
 assert request(tokens,typed).block_hashes==request(tokens,original).block_hashes
# Small prompts exercise fresh lists, changed input, eviction and capacity bypass.
ev=InitialPrefixHashMemo(original,max_entries=1)
for x in ([1,2],[3,4],[1,2]):assert request(x,ev).block_hashes==request(x,original).block_hashes
assert len(ev.rows)==1
small=InitialPrefixHashMemo(original,max_bytes=1);assert request(ids[0],small).block_hashes==ref[0] and not small.rows
# Bypass metadata must pass the complete request through unchanged.
for field,value in (('prompt_embeds',object()),('mm_features',[object()]),('lora_request',object()),('cache_salt','salt')):
 r=request([1,2],None);setattr(r,field,value);seen=[];sentinel=[b'fallback']
 f=InitialPrefixHashMemo(lambda q:seen.append(q) or sentinel)
 assert f(r) is sentinel and seen==[r] and not f.rows
seed=_kv.NONE_HASH
try:
 _kv.NONE_HASH=b'changed seed guard test'
 assert request(ids[0],memo).block_hashes==request(ids[0],original).block_hashes
finally:
 _kv.NONE_HASH=seed
assert request(ids[0],memo).block_hashes==ref[0]
assert memo.bytes<=memo.max_bytes and len(memo.rows)<=memo.max_entries
# Installed factory router: OFF -> ON -> OFF, immutable code and exact hashes.
import tempfile,os
with tempfile.TemporaryDirectory() as tmp:
 mode_path=Path(tmp)/'mode.json'
 def mode(phase,enabled,version):
  f=mode_path.with_suffix('.tmp');f.write_text(json.dumps(dict(phase=phase,enabled=enabled,version=version)));f.replace(mode_path)
 os.environ['EXTREME_PREFIX_HASH_MODE_FILE']=str(mode_path)
 routed=get_request_block_hasher(2,sha256)
 del os.environ['EXTREME_PREFIX_HASH_MODE_FILE']
 for phase,enabled,version in [('off_a',False,1),('on',True,2),('off_b',False,3)]:
  mode(phase,enabled,version)
  for _ in range(2):
   assert [request(x,routed).block_hashes for x in ids]==ref
  if phase=='off_a':assert routed.memo.hits==routed.memo.misses==0
  if phase=='on':assert routed.memo.hits>=48 and routed.memo.misses<=48
  if phase=='off_b':assert routed.memo.hits==48 and routed.memo.misses==48
 # A continuation never reads the control file, even if it is missing.
 a=request(ids[0],routed);b=request(ids[0],original);mode_path.unlink()
 a.append_output_token_ids(7);b.append_output_token_ids(7);assert a.block_hashes==b.block_hashes
 try:request(ids[0],routed)
 except FileNotFoundError:pass
 else:raise AssertionError('missing mode must fail')
 mode_path.write_text('{}')
 try:request(ids[0],routed)
 except RuntimeError:pass
 else:raise AssertionError('invalid mode must fail')
out={'status':'pass','scope':'CPU initial hash memo semantics and cost, not live removable wall','block_size':2,'reference48_s':ref_s,'cold48_s':cold_s,'hot48_s':hot_s,'entries':len(memo.rows),'estimated_bytes':memo.bytes,'gates':['48 original exact','48 cold exact','48 hit exact','concurrent hits','caller mutation isolation','continuation and partial tail exact','salt fallback exact','changed tokens and LRU eviction','oversize bypass','complete embeds/MM/LoRA/salt fallback','hash seed change fallback','bool/float token fallback','installed OFF ON OFF router exact','observed route and hit counts','continuation skips mode read','missing/invalid mode fails']}
Path('/data/wio/Inference_Foundry/evidence/20260929_loop081_bound/run674/preflight/hash_integration.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
