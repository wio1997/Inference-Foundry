# SPDX-License-Identifier: Apache-2.0
"""Qualification metadata after native target/draft graph replay only.
No tensor access/copy/hash/synchronization. Never enabled for formal timing.
"""
import os,json,pathlib,time,threading
_LOCK=threading.Lock()

def graph_replayed(mode,batch,is_draft):
    if os.environ.get('FLASH_NATIVE_GRAPH_OBSERVER','0')!='1':raise ValueError('Graph qualification observer called while disabled')
    root=pathlib.Path('/run-evidence');phase=None
    p=root/'current-case.json'
    if p.exists():
        v=json.loads(p.read_text());phase={k:v[k] for k in ['case_id','phase','global_dp_rank'] if k in v and type(v[k]) in (str,int)}
    row={'project_id':'glm53-flash','kind':'native_graph_replay_after_branch','pid':os.getpid(),'monotonic_ns':time.monotonic_ns(),'runtime_mode':str(mode),'is_draft':bool(is_draft),'global_dp_rank_env':os.environ.get('FLASH_GLOBAL_DP_RANK'),'request_phase':phase,'scope':'Actual native replay invoked; correlate same-node active requestinterval andselectedDP rank. Startupreplay alone not functional proof.'}
    for name in ['num_tokens','num_reqs','uniform','has_lora']:
        value=getattr(batch,name,None);row[name]=value if type(value) in (bool,int) else None
    target=root/'qualification-metadata';target.mkdir(exist_ok=True);f=target/('graph-'+str(os.getpid())+'.jsonl')
    with _LOCK:
        if f.exists() and f.stat().st_size>8*1024*1024:raise RuntimeError('Qualification metadata bound exceeded')
        with f.open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n')
