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
rows=[]
for bs in (2,):
 f=get_request_block_hasher(bs,sha256);samples=[]
 for rep in range(3):
  not_serving();start=time.perf_counter()
  output=[f(SimpleNamespace(block_hashes=[],num_tokens=len(x),all_token_ids=x,mm_features=[],lora_request=None,cache_salt=None,prompt_embeds=None)) for x in ids]
  samples.append(time.perf_counter()-start)
  if rep==0:reference=output
  else:assert output==reference
 rows.append({'hash_block_size':bs,'full48_s':samples,'median_s':statistics.median(samples),'blocks_per_request':[len(x) for x in output]})
result={'scope':'CPU original text-only initial prefix hashing; not live exposed time; actual live hash granularity not established here','all48_token_hashes':[hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest() for x in ids],'rows':rows}
out=Path('/data/wio/Inference_Foundry/evidence/20260929_loop081_bound/run673/prefix_hash_block2_screen.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
