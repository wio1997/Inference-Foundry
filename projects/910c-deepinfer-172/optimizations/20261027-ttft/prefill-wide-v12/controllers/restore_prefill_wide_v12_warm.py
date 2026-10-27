"""Restore prefix/graph warm state only; no AISBench or formal measurement."""
from pathlib import Path
import concurrent.futures,json,time,urllib.request,traceback
p=Path('/data/tiankuan/wio/910c-deepinfer-172');dest=p/'diagnostics/restore-wide-v12';dest.mkdir(exist_ok=True);status=p/'runs/prefill-wide-v12-restore-warm-status.json';state={'state':'WARMING','purpose':'restore accepted prefix and verification graph warm state after RMS evaluation; no new benchmark scores','started':time.time()}
def save():status.write_text(json.dumps(state,indent=2))
def complete(row):
 body={'model':'dsv41','prompt':row['prompt'],'max_tokens':row['max_tokens'],'temperature':0,'ignore_eos':True,'stream':False};req=urllib.request.Request('http://127.0.0.1:8172/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=180) as f:return json.load(f)
def parallel(rows):
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:return list(pool.map(complete,rows))
save()
try:
 for phase in ['measure1','measure2']:
  folder=p/'workload/frozen'/phase;seeds=json.loads((folder/'warmup.json').read_text());responses=parallel([{'prompt':x['prompt'],'max_tokens':1} for x in seeds]);(dest/(phase+'-prefix-prime.json')).write_text(json.dumps(responses,ensure_ascii=False,indent=2))
 folder=p/'workload/frozen/measure2';data=[json.loads(x) for x in next(folder.rglob('*.requests.jsonl')).read_text().splitlines()];refs={int(x['data_id']):x['prediction'] for x in map(json.loads,(folder/'frozen-vllm-details.jsonl').read_text().splitlines())}
 responses=parallel([{'prompt':x['question'],'max_tokens':32} for x in data]);(dest/'graph-prime.json').write_text(json.dumps(responses,ensure_ascii=False,indent=2))
 assert len(responses)==8
 for i,x in enumerate(responses):
  assert refs[i].startswith(x['choices'][0]['text']) and x['usage']['prompt_tokens']==32768 and x['usage']['completion_tokens']==32
 assert sum(x['usage']['prompt_tokens_details']['cached_tokens'] for x in responses)==235904
 state.update(state='PASS',primed_prefix_phases=2,graph_warm_requests=8,graph_warm_output_tokens_each=32,actual_cached_tokens=235904,formal_benchmark_rounds_added=0,ended=time.time());save()
except BaseException as e:state.update(state='FAIL',error=repr(e));save();traceback.print_exc();raise
