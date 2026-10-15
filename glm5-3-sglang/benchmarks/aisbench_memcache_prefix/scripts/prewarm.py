#!/usr/bin/env python3
"""Warm only each unique request's own 74,400-token prefix, real PD router, four client slots."""
from bench_profile import PROFILE
from bench_profile import benchmark_locks as acquire_benchmark_locks
import argparse
import array
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request
import urllib.error

p=argparse.ArgumentParser()
p.add_argument('--root',default='/work/aisbench-glm53-run')
p.add_argument('--cc',required=True)
p.add_argument('--execute',action='store_true')
p.add_argument('--manifest',default='dataset-manifest.jsonl')
a=p.parse_args();r=Path(a.root)
assert a.execute,'Explicit execution flag required'
benchmark_locks=[]
benchmark_locks=acquire_benchmark_locks(r)
records=[json.loads(x)for x in (r/a.manifest).read_text().splitlines()]
cc=int(a.cc)if a.cc.isdigit()else a.cc
rows=[x for x in records if x['cc']==cc]
assert len(rows)==(64 if isinstance(cc,int)else 1)
(r/'warmups').mkdir(exist_ok=True)
lock=(r/'warmups'/f'cc{a.cc}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=r/'warmups'/f'cc{a.cc}.jsonl';result=r/'warmups'/f'cc{a.cc}.complete.json'
assert not out.exists()and not result.exists(),'Existing warmup trace requires review, do not repeat silently'
guard=threading.Lock();start=time.perf_counter();epoch=time.time()

def one_dp(dp):
    for rec in rows:
        if rec['row']%4!=dp:continue
        raw=Path(rec['prefix_ids_file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==rec['prefix_ids_sha256']
        ids=array.array('I');ids.frombytes(raw);assert len(ids)==74400
        body=dict(input_ids=list(ids),sampling_params=dict(max_new_tokens=1,ignore_eos=True,temperature=0),stream=False,return_cached_tokens_details=True)
        request=urllib.request.Request(PROFILE['router_url'].rstrip('/')+'/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        t=time.perf_counter();row=dict(cc=cc,row=rec['row'],client_worker_slot=dp,prompt_sha256=rec['prompt_sha256'],
            prefix_ids_sha256=rec['prefix_ids_sha256'],input_tokens=74400,start_epoch=time.time())
        try:
            with urllib.request.urlopen(request,timeout=7200)as response:data=json.load(response)
            meta=data['meta_info'];assert meta['prompt_tokens']==74400 and meta['completion_tokens']==1,meta
            assert (meta.get('finish_reason')or{}).get('type')=='length',data
            row.update(success=True,usage=meta,response=data,e2e_s=time.perf_counter()-t)
        except Exception as e:
            row.update(success=False,error=repr(e),e2e_s=time.perf_counter()-t)
            if isinstance(e,urllib.error.HTTPError):row['http_status']=e.code;row['http_body']=e.read().decode(errors='replace')
            with guard,out.open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
            raise
        with guard,out.open('a')as f:f.write(json.dumps(row)+'\n');f.flush()

with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
    futures=[pool.submit(one_dp,i)for i in range(4)]
    for future in futures:future.result()
data=[json.loads(x)for x in out.read_text().splitlines()]
assert len(data)==len(rows)and all(x['success']for x in data)
elapsed=time.perf_counter()-start
report=dict(phase='PREFIX_PREFILL_DONE_STORAGE_AUDIT_REQUIRED',cc=cc,requests=len(rows),
            prefix_tokens_each=74400,total_prefill_tokens=len(rows)*74400,elapsed_s=elapsed,
            prefill_tokens_per_second=len(rows)*74400/elapsed,start_epoch=epoch,end_epoch=time.time(),
            transport='real PD router native/generate token-ID input, four client slots; no fake bootstrap and no explicit DP override',
            note='Real PD prefix warmup including transfer and one output token elapsed time excluded from formal AISBench metrics. Completion does not assert all asynchronous storage writes are finished.')
result.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
