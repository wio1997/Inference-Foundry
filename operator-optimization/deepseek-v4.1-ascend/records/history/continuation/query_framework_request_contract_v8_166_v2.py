from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, socket
assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
P = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030')
B = P/'runtime/benchmark-20261028/workload/frozen/measure1'
files = list(B.rglob('*.requests.jsonl'))
assert len(files) == 1, ('input_file_count', len(files))
pins = {}
def read(p):
    a=p.stat(); b=p.read_bytes(); z=p.stat()
    assert (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
    pins[str(p)] = hashlib.sha256(b).hexdigest()
    return b
rows = [json.loads(x) for x in read(files[0]).decode().splitlines()]
seeds = json.loads(read(B/'warmup.json'))
assert len(rows)==8 and 1<=len(seeds)<=8
def sha(v):
    return hashlib.sha256(json.dumps(v,ensure_ascii=False).encode()).hexdigest()
def kind(v):
    if isinstance(v,str): return 'text'
    if isinstance(v,list) and all(isinstance(t,int) and not isinstance(t,bool) for t in v): return 'token_ids'
    return type(v).__name__
def prefix(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y: break
        n+=1
    return n
prompts=[r['question'] for r in rows]
seed_prompts=[r['prompt'] for r in seeds]
input_hash=sha(prompts)
framework_result_path=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/runtime/logs/framework-20261030T091949Z-3d9bcf87/result.json')
fr=json.loads(read(framework_result_path))
assert pins[str(framework_result_path)]=='f6b3f4ad79503ffa143765884d36a1712546b15b104ed71edc267b1fee4206fb'
client_path=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official_profile_client.py')
read(client_path)
out=dict(generated_utc=datetime.now(timezone.utc).isoformat(),source_sha256=pins,source_stats_unchanged=True,
 input_file=str(files[0]),input_question_list_json_sha256=input_hash,framework_result_prompt_sha256=fr.get('prompt_sha256'),seed_fixture_declared_fields=[{k:r.get(k) for k in ['dp_rank','group_id','input_tokens','max_tokens','shared_prefix_tokens','included_in_formal_statistics']} for r in seeds],
 hash_equal_to_framework_result=input_hash==fr.get('prompt_sha256'),
 request_rows=[dict(index=i,kind=kind(p),length=len(p),prompt_sha256=sha(p),dp_rank=r.get('dp_rank'),metadata_keys=sorted(r.keys())) for i,(r,p) in enumerate(zip(rows,prompts))],
 seed_rows=[dict(index=i,kind=kind(p),length=len(p),prompt_sha256=sha(p),max_tokens=r.get('max_tokens'),dp_rank=r.get('dp_rank')) for i,(r,p) in enumerate(zip(seeds,seed_prompts))],
 question_prefix_matrix=[[prefix(a,b) for b in prompts] for a in prompts],
 seed_to_question_prefix_matrix=[[prefix(a,b) for b in prompts] for a in seed_prompts],
 no_prompt_or_token_values_exported=True,no_inference_or_device_work=True,
 limits=['Prefix length is in token IDs only if kind is token_ids; otherwise characters.',
 'Identical token prefix does not prove identical cache hash extra keys, cache salt, resolved H or runtime identity.',
 'Client request construction is source contract, not per-function runtime timing.'])
target=P/'analysis/whole-scope-continuation-20261030/profile-readonly-20261031/framework-request-contract-v8.json'
assert not target.exists(); payload=json.dumps(out,ensure_ascii=False,indent=2)+'\n';target.write_text(payload)
print(json.dumps(dict(result=out,full_result=str(target),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
