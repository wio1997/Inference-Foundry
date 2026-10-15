#!/usr/bin/env python3
"""Independently verify every native page's K/V/index presence before flushing L1/L2."""
from bench_profile import PROFILE
import argparse
import array
import hashlib
import json
from pathlib import Path
import time
from memcache_hybrid import DistributedObjectStore, LocalConfig
from sglang.srt.mem_cache.utils import get_hash_str

p=argparse.ArgumentParser()
p.add_argument('--cc',required=True)
p.add_argument('--start-row',type=int,default=0)
p.add_argument('--audit-label',required=True)
p.add_argument('--timeout',type=int,default=600)
p.add_argument('--manifest',default='dataset-manifest.jsonl')
a=p.parse_args()
r=Path(PROFILE['p_audit_root'])
br=r/'bench-audit'
manifest=[json.loads(x)for x in (br/a.manifest).read_text().splitlines()]
cc=int(a.cc)if a.cc.isdigit()else a.cc
rows=[x for x in manifest if x['cc']==cc and (not isinstance(cc,int) or a.start_row<=x['row']<a.start_row+64)]
assert len(rows)==(64 if isinstance(cc,int)else 1)
extra=json.loads((r/'conf/model-memcache.json').read_text())
config=LocalConfig()
for k,v in extra.items():
    if hasattr(config,k):setattr(config,k,v)
store=DistributedObjectStore()
assert store.setup(config)==0
assert store.init(0,False)==0  # Only metadata queries, no new provider memory or BM data path.
try:
    expected={}
    for row in rows:
        f=br/'prefixes'/f'cc{a.cc}'/Path(row['prefix_ids_file']).name
        data=f.read_bytes();assert hashlib.sha256(data).hexdigest()==row['prefix_ids_sha256']
        ids=array.array('I');ids.frombytes(data);assert len(ids)==74400
        hashes=get_hash_str(list(ids[:74368]),page_size=128)
        assert len(hashes)==581
        # Native BF16 MLA page_first_kv_split stores all three components.
        expected[row['row']]=[f"{extra['extra_backend_tag']}_{h}__{suffix}"for h in hashes for suffix in ['k','v','index_k']]
    pending=set(expected);deadline=time.monotonic()+a.timeout;checks={};start=time.perf_counter()
    while pending:
        for rid in list(pending):
            keys=expected[rid];states=[]
            for offset in range(0,len(keys),512):states+=list(store.batch_is_exist(keys[offset:offset+512]))
            assert len(states)==len(keys)
            misses=[keys[i]for i,v in enumerate(states)if v!=1]
            checks[rid]=dict(objects=len(keys),present=len(keys)-len(misses),missing_count=len(misses),missing_examples=misses[:4])
            if not misses:pending.remove(rid)
        report=dict(phase='STORAGE_PREFIX_AUDIT_PASS'if not pending else 'WAITING_ASYNC_STORAGE_BACKUP',
                    cc=cc,source_start_row=a.start_row,source_rows=[x['row']for x in rows],audit_label=a.audit_label,requests=len(rows),pending_rows=sorted(pending),checks=checks,
                    cached_tokens_each=74368,elapsed_s=time.perf_counter()-start)
        (r/f'evidence/prefix-storage-official13-fourbatch-{a.audit_label}.json').write_text(json.dumps(report,indent=2)+'\n')
        if not pending:print(json.dumps(report),flush=True);break
        if time.monotonic()>=deadline:raise RuntimeError(f'Incomplete K/V/index storage; see prefix-storage-official13-fourbatch-cc{a.cc}.json')
        time.sleep(20)
finally:
    rc=store.close();assert rc==0,rc
