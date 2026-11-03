from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, socket, time, urllib.request

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030')
P=ROOT/'analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72'
assert json.loads((P/'controller-finished.json').read_text())['state']=='CAPTURED_AND_RESTORED'
OUT=P/'analysis/frozen-input-token-identity-v12.json';assert not OUT.exists()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
base='http://127.0.0.1:30075'
with opener.open(base+'/openapi.json',timeout=15) as r:
    raw=r.read(8*1024**2);api=json.loads(raw)
assert '/tokenize' in api['paths']
schemas=api['components']['schemas']
completion=schemas['CompletionRequest']['properties']
tokenize=schemas['TokenizeCompletionRequest']['properties']
assert 'add_special_tokens' in completion and 'add_special_tokens' in tokenize
special=completion['add_special_tokens'].get('default')
assert isinstance(special,bool)
source=ROOT/'runtime/benchmark-20261028/workload/frozen/measure1'
paths=list(source.rglob('*.requests.jsonl'));assert len(paths)==1
rows=[json.loads(line) for line in paths[0].read_text().splitlines()]
assert len(rows)==8
inputs=[r['question'] for r in rows]
assert hashlib.sha256(json.dumps(inputs,ensure_ascii=False).encode()).hexdigest()=='755b823003b4d200a4590424a245cdfb484d9b8c3d0b5225a7a058790b87e8c8'
seeds=json.loads((source/'warmup.json').read_text());assert len(seeds)==2
results=[];start=time.monotonic()
for index,(kind,row) in enumerate([('formal',r) for r in rows]+[('seed',r) for r in seeds]):
    assert time.monotonic()-start<120,'finite CPU tokenization budget'
    prompt=row['question'] if kind=='formal' else row['prompt']
    body=dict(model='dsv41',prompt=prompt,add_special_tokens=special)
    request=urllib.request.Request(base+'/tokenize',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    t0=time.monotonic_ns()
    with opener.open(request,timeout=20) as response:
        data=json.loads(response.read(4*1024**2))
    tokens=data['tokens'];assert all(type(x)is int for x in tokens)
    expected=32768 if kind=='formal' else 29696
    assert data['count']==len(tokens)==expected,(kind,index,data['count'],len(tokens))
    results.append(dict(kind=kind,index=index if kind=='formal' else index-8,dp_rank=row.get('dp_rank'),token_ids=tokens,
      token_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
      count=len(tokens),HTTP_elapsed_ns=time.monotonic_ns()-t0,elapsed_is_not_hash_CPU_time=True))
def lcp(a,b):
    count=0
    for x,y in zip(a,b):
        if x!=y:break
        count+=1
    return count
formal=[x for x in results if x['kind']=='formal'];seed=[x for x in results if x['kind']=='seed']
pairwise=[[lcp(a['token_ids'],b['token_ids']) for b in formal] for a in formal]
seed_lcp=[[lcp(a['token_ids'],b['token_ids']) for b in seed] for a in formal]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),formal_requests=8,seed_requests=2,
    formal_token_counts=[r['count'] for r in formal],seed_token_counts=[r['count'] for r in seed],
    distinct_full_formal_token_hashes=len({r['token_sha256'] for r in formal}),all8_common_prefix_tokens=min(pairwise[i][j] for i in range(8) for j in range(8)),
    pairwise_formal_token_prefix_lengths=pairwise,formal_to_seed_token_prefix_lengths=seed_lcp,
    completion_add_special_tokens_default=special,tokenize_declared_default=tokenize['add_special_tokens'].get('default'),explicit_tokenize_add_special_tokens=special,
    input_sha256='755b823003b4d200a4590424a245cdfb484d9b8c3d0b5225a7a058790b87e8c8',
    scope='Current166same-model/tokenizer HTTP tokenization of exact server-resident frozen prompts;not attestation of historical169buffer values.',
    no_inference_endpoint_called=True,no_hash_function_CPU_cost_measured=True,resolved_hash_block_size='UNKNOWN',
    cache_hash_extra_namespace_and_live_reuse_not_proved=True,whole_modules_closed=0,overall_complete=False)
payload=json.dumps(dict(summary=summary,tokenization_results=results,API_contract=dict(openapi_sha256=hashlib.sha256(raw).hexdigest(),completion_add_special_tokens=completion['add_special_tokens'],tokenize_add_special_tokens=tokenize['add_special_tokens'])),ensure_ascii=False,separators=(',',':'))+'\n'
OUT.write_text(payload)
print(json.dumps(dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
