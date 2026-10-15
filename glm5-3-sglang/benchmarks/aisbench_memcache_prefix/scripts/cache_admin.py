#!/usr/bin/env python3
"""Idle-only flush/clear for the exact new native MemCache namespace."""
from bench_profile import PROFILE
from bench_profile import benchmark_locks as acquire_benchmark_locks
import argparse
import fcntl
import json
from pathlib import Path
import time
from cold_run_matrix import get, load_counts, LEADERS, write

p=argparse.ArgumentParser();p.add_argument('operation',choices=['flush_l1_l2','clear_l1_l2_l3'])
p.add_argument('--label',required=True);p.add_argument('--root',default='/work/aisbench-glm53-run')
a=p.parse_args();r=Path(a.root);locks=[]
locks=acquire_benchmark_locks(r)
receipt={'operation':a.operation,'epoch':time.time(),'heads':{},'responses':{}}
for role,url,tp,dp,ep in LEADERS:
    raw=get(url+'/get_server_info');info=raw.get('server_args')or raw
    assert info['tp_size']==tp and info['attn_dp_size']==dp and info['ep_size']==ep
    if role=='prefill':
        assert info['enable_hierarchical_cache'] and info['hicache_storage_backend']=='npu_memcache'
        extra=info['hicache_storage_backend_extra_config']
        if isinstance(extra,str):extra=json.loads(extra)
        assert 'host_io_mode' not in extra
        assert extra['meta_service_url']==PROFILE['meta_service_url']
        assert extra['extra_backend_tag']==PROFILE['extra_backend_tag']
    else:assert not info['enable_hierarchical_cache'] and info['mem_fraction_static']==.90
    for _ in range(60):
        loads=get(url+'/v1/loads');counts=[v for _,v in load_counts(loads)]
        assert counts,'Unrecognized actual load schema'
        if all(v==0 for v in counts):break
        time.sleep(5)
    else:raise RuntimeError('Service busy; refusing cache administration')
    receipt['heads'][role]={'server_info':raw,'loads':loads}
for role,url,*_ in LEADERS:
    receipt['responses'][role]=get(url+'/flush_cache?timeout=120',method='POST',timeout=150)
assert all(v is not False for v in receipt['responses'].values()),receipt
if a.operation=='clear_l1_l2_l3':
    receipt['responses']['dedicated_l3']=get(LEADERS[0][1]+'/hicache/storage-backend/clear',method='POST',timeout=180)
    assert receipt['responses']['dedicated_l3'] is not False,receipt
write(r/'logs'/f'cache-admin-{a.label}.json',receipt)
print(json.dumps({'operation':a.operation,'label':a.label,'status':'HTTP_ADMIN_PASS'}),flush=True)
