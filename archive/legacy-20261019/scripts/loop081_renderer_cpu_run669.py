"""CPU-only frozen text preprocessing concurrency screen; no E2E claim."""
import asyncio,concurrent.futures,hashlib,json,time,os
from pathlib import Path
from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
DATA=Path('/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl')
tok=DeepseekV4Tokenizer.from_pretrained('/data/yxy/DeepSeek-V4-Flash-0731-w4a8')
messages=[[{'role':'user','content':json.loads(x)['question']}] for x in DATA.read_text().splitlines()[:48]]
def template(m): return tok.apply_chat_template(messages=m,conversation=m,tokenize=False)
def encode(s): return tok(s,add_special_tokens=False)['input_ids']
reference=[encode(template(m)) for m in messages]
async def batch(ms,workers):
 records=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
  loop=asyncio.get_running_loop();t0=time.perf_counter_ns()
  def call(i,stage,fn,arg):
   a=time.perf_counter_ns();o=fn(arg);b=time.perf_counter_ns();records.append({'i':i,'stage':stage,'start_ns':a,'end_ns':b});return o
  async def one(i,m):
   text=await loop.run_in_executor(ex,call,i,'template',template,m)
   ids=await loop.run_in_executor(ex,call,i,'tokenize',encode,text)
   return ids,time.perf_counter_ns()
  outputs=await asyncio.gather(*(one(i,m) for i,m in enumerate(ms)))
  return outputs,{'workers':workers,'wall_ms':(time.perf_counter_ns()-t0)/1e6,'completion_ms':[(t-t0)/1e6 for _,t in outputs],'calls':records}
rows=[]
for repeat in range(3):
 for workers in ([1,4,12,1] if repeat%2==0 else [1,12,4,1]):
  for cohort in range(4):
   outputs,row=asyncio.run(batch(messages[cohort*12:(cohort+1)*12],workers))
   assert [x for x,_ in outputs]==reference[cohort*12:(cohort+1)*12]
   row.update(repeat=repeat,cohort=cohort);rows.append(row)
result={'scope':'CPU-only exact frozen prompt template/tokenization concurrency screen. Excludes HTTP validation, Core hash/admission and NPU contention.','dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'token_counts':[len(x) for x in reference],'token_sha256':[hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest() for x in reference],'all_token_ids_exact':True,'rows':rows,'pid':os.getpid()}
p=Path('/data/wio/Inference_Foundry/evidence/20260929_loop081_bound/run669/renderer_cpu.json');p.write_text(json.dumps(result,indent=2)+'\n')
import statistics
print(json.dumps({w:{'median_batch_ms':statistics.median(r['wall_ms'] for r in rows if r['workers']==w)} for w in (1,4,12)}));print('PASS exact token IDs all48 × all repetitions')
