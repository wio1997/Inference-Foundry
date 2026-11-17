#!/usr/bin/env python3
"""Bounded, separate diagnostic capture; never a scored benchmark runner."""
import argparse, datetime as dt, hashlib, json, os, signal, subprocess, time, urllib.request
from pathlib import Path

def atomic(p, x):
    p=Path(p); q=p.with_name(p.name+'.tmp'); q.write_text(json.dumps(x,indent=2)); q.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(g):
    now=dt.datetime.now(dt.timezone.utc)
    assert now < dt.datetime.fromisoformat(g['expires_at'].replace('Z','+00:00')), 'expired gate'
    assert g['scored_pipeline_idle'] is True and g['active_benchmark_jobs']==[], 'scored workload active/unknown'
    assert g['diagnostic_only'] is True and 1 <= g['max_seconds'] <= 600
    assert isinstance(g['workload_argv'],list) and g['workload_argv']
    assert set(g['roles'])=={'p','d'}
    for role,x in g['roles'].items():
        assert all(x.get(k) for k in ['endpoint','container_id','image_id','started_at','profiler_dir','launch_argv_sha256'])
        assert x['endpoint'].startswith('http://') and x['profiler_dir'].startswith('/')
        c=x['profiler_config']; assert c['profiler']=='torch' and c['torch_profiler_dir']==x['profiler_dir']
        assert c.get('ignore_frontend') is True and 0 < c.get('max_iterations',0) <= 64
        assert c.get('warmup_iterations',0)==0, 'Ascend wrapper has no schedule support'
    assert g['roles']['p']['profiler_dir']!=g['roles']['d']['profiler_dir']
def call(endpoint, action, out):
    t=time.time(); record={'url':endpoint.rstrip('/')+'/'+action,'utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        req=urllib.request.Request(record['url'],data=b'',method='POST')
        with urllib.request.urlopen(req,timeout=120) as r: record.update(status=r.status,body=r.read(65536).decode(errors='replace'))
        assert record['status']==200
    except Exception as e: record['error']=repr(e); raise
    finally: record['elapsed_s']=time.time()-t; atomic(out,record)
def metrics(endpoint,out):
    record={'url':endpoint.rstrip('/')+'/metrics','utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(record['url'],timeout=10) as r: record.update(status=r.status,body=r.read(8*1024*1024).decode(errors='replace'))
    except Exception as e:record['error']=repr(e)
    atomic(out,record)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gate',required=True);ap.add_argument('--out',required=True);ap.add_argument('--execute',action='store_true'); a=ap.parse_args()
    g=json.loads(Path(a.gate).read_text());validate(g)
    if not a.execute: print(json.dumps({'validated':True,'gate_sha256':sha(a.gate),'execute':False},indent=2)); return
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    atomic(out/'gate.json',g); state={'gate_sha256':sha(a.gate),'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'phase':'starting','diagnostic_only':True}; atomic(out/'state.json',state)
    started=[]; proc=None; failure=None
    try:
        # Direct role endpoints, never the PD proxy; starting P/D is sequential and timestamps are retained.
        for role,x in g['roles'].items():
            metrics(x['endpoint'],out/f'{role}_metrics_before.json')
            started.append(role) # even an HTTP timeout can have started profiling; always try cleanup
            call(x['endpoint'],'start_profile',out/f'{role}_start.json')
        state['phase']='workload';atomic(out/'state.json',state)
        with (out/'workload.log').open('wb') as log:
            proc=subprocess.Popen(g['workload_argv'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=g['workload_cwd'])
            try: rc=proc.wait(timeout=g['max_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
                raise RuntimeError('diagnostic workload exceeded bound')
        state['workload_rc']=rc
        if rc: raise RuntimeError(f'workload rc={rc}')
    except Exception as e: failure=repr(e)
    finally:
        state['stop_errors']={}
        for role in reversed(started):
            try:call(g['roles'][role]['endpoint'],'stop_profile',out/f'{role}_stop.json')
            except Exception as e:state['stop_errors'][role]=repr(e)
            metrics(g['roles'][role]['endpoint'],out/f'{role}_metrics_after.json')
        state.update(phase='terminal',ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),error=failure,profile_success_verified=False)
        atomic(out/'state.json',state)
    if failure or state['stop_errors']: raise SystemExit(1)
if __name__=='__main__':main()
