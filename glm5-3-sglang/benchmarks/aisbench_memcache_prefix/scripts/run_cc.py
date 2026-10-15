#!/usr/bin/env python3
"""Run one formal distinct-data cc only after prefill and complete storage audit."""
from bench_profile import PROFILE
from bench_profile import benchmark_locks as acquire_benchmark_locks
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from cold_run_matrix import get,load_counts,monitor,write,LEADERS

p=argparse.ArgumentParser();p.add_argument('--cc',type=int,choices=[16,32,64],required=True)
p.add_argument('--root',default='/work/aisbench-glm53-run');p.add_argument('--execute',action='store_true')
a=p.parse_args();assert a.execute;r=Path(a.root);cc=a.cc;locks=[]
for f in ['logs','runs','reports']:(r/f).mkdir(exist_ok=True)
locks=acquire_benchmark_locks(r)
progress=r/f'logs/formal-cc{cc}.json'
assert not progress.exists() and not(r/f'runs/cc{cc}').exists(),'Existing formal trace requires explicit audit'
def phase(name,**kw):write(progress,dict(phase=name,cc=cc,pid=os.getpid(),epoch=time.time(),**kw))
try:
    phase('FORMAL_PREFLIGHT')
    audit=json.loads((r/'dataset-audit.json').read_text());dataset=r/f'datasets/cc{cc}/test.jsonl'
    assert audit['formal_requests']==64 and audit['unique_prompts']==64 and audit['per_cc']==64
    assert hashlib.sha256(dataset.read_bytes()).hexdigest()==audit['datasets'][str(cc)]['file_sha256']
    warm=json.loads((r/f'warmups/cc{cc}.complete.json').read_text());assert warm['requests']==64
    storage=json.loads((r/f'logs/prefix-storage-cc{cc}.json').read_text())
    assert storage['phase']=='STORAGE_PREFIX_AUDIT_PASS' and storage['requests']==64 and not storage['pending_rows']
    flush=json.loads((r/f'logs/cache-admin-after-warm-cc{cc}.json').read_text())
    assert flush['operation']=='flush_l1_l2' and flush['epoch']>warm['end_epoch']
    for role,url,tp,dp,ep in LEADERS:
        info=get(url+'/get_server_info');info=info.get('server_args')or info
        assert info['tp_size']==tp and info['attn_dp_size']==dp and info['ep_size']==ep
        assert info['enable_cache_report'] and info['mem_fraction_static']==(.75 if role=='prefill'else .90)
        assert info['enable_hierarchical_cache']==(role=='prefill')
        if role=='prefill':
            extra=info['hicache_storage_backend_extra_config'];extra=json.loads(extra)if isinstance(extra,str)else extra
            assert 'host_io_mode' not in extra
        c=[v for _,v in load_counts(get(url+'/v1/loads'))];assert c and all(v==0 for v in c)
    subprocess.run([sys.executable,str(r/'scripts/build_config.py'),'--root',str(r),'--cc',str(cc)],check=True)
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',AIS_BENCH_REQUEST_TIMEOUT='7200',
        NO_PROXY='*',no_proxy='*',PYTHONPATH=str(r/'scripts')+os.pathsep+env.get('PYTHONPATH',''))
    for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY']:env.pop(k,None)
    cfg=r/f'configs/cc{cc}.py';runroot=r/f'runs/cc{cc}'
    cmd=['ais_bench',str(cfg),'--mode','perf','--num-warmups','0','--num-prompts','64','--work-dir',str(runroot)]
    write(r/f'logs/command-cc{cc}.json',dict(command=cmd,config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),
         client_concurrency=cc,requests=64,total_attempts_per_request=1,expected_input_tokens=80000,
         expected_output_tokens=600,expected_cached_tokens=74368,prewarm=warm,storage_audit=storage))
    phase('BENCHMARKING',command=cmd)
    stop=threading.Event();thread=threading.Thread(target=monitor,args=(stop,r/f'logs/load-cc{cc}.jsonl'),daemon=True);thread.start()
    try:
        with(r/f'logs/aisbench-cc{cc}.log').open('w')as out:
            worker=subprocess.Popen(cmd,cwd=PROFILE['benchmark_cwd'],env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=worker.wait(timeout=4*3600)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGTERM);worker.wait(timeout=60);raise
    finally:stop.set();thread.join(timeout=10)
    dirs=[d for d in runroot.iterdir()if d.is_dir()];assert len(dirs)==1,dirs
    phase('REPORTING',cli_exit=rc,run_dir=str(dirs[0]))
    subprocess.run([sys.executable,str(r/'scripts/summarize.py'),'--root',str(r),'--cc',str(cc),
                    '--run-dir',str(dirs[0])],env=env,check=True)
    assert rc==0,rc
    result=json.loads((r/f'reports/cc{cc}/result.json').read_text());assert result['valid']
    phase('COMPLETE',report=str(r/f'reports/cc{cc}/result.json'))
except Exception as e:phase('FAILED',error=str(e));raise
