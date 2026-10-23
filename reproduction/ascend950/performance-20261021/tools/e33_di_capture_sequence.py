#!/usr/bin/env python3
"""Frozen DI C32 full-cohort U/P/U sequence; profile stop requires global drain."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from e33_vllm_capture_sequence import ROOT,CONTAINER_ROOT,save,stamp,wait_file,container_path

CONTAINER='performance-20261021-deepinfer-runtime'
BASE='http://127.0.0.1:18121'
def rpc(path,body=None,timeout=30):
    request=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:return json.load(response)
def idle(path):
    subprocess.run([sys.executable,str(ROOT/'tools/e26_wait_idle.py'),'--framework','di','--base-url',BASE,'--timeout','600','--json-out',str(path)],check=True,stdout=subprocess.DEVNULL,timeout=620)
def actors(control):
    status=rpc('/e33/status');rows=status['ranks'];assert len(rows)==8 and all(x['worker_alive'] for x in rows)
    result=[dict(pid=x['parent_worker_pid'],role='worker',rank=x['rank']) for x in rows]
    code='''from pathlib import Path
import json,os
workers=json.loads(os.environ['E33_WORKERS']);parents=set()
for p in workers:
 text=Path('/proc')/str(p)/'stat';s=text.read_text();parents.add(int(s[s.rfind(')')+2:].split()[1]))
assert len(parents)==1 and next(iter(parents))>1
print(json.dumps(sorted(parents)))'''
    p=subprocess.run(['docker','exec','-e','E33_WORKERS='+json.dumps([x['pid'] for x in result]),CONTAINER,'python3','-c',code],check=True,capture_output=True,text=True)
    result.extend(dict(pid=pid,role='api_parent') for pid in json.loads(p.stdout))
    save(control/'actors.json',result)
    return status
def return_artifacts(outdir):
    code='''from pathlib import Path
import json,os
root=Path('/work/performance-20261021');p=Path(os.environ['E33_OUT']);assert p.is_relative_to(root/'evidence-e33/di-detailed')
owner=(root/'evidence-e33').stat();rows=[]
for leaf in sorted(p.rglob('*_ascend_pt')):
 files=[f for f in leaf.rglob('*') if f.is_file()]
 rows.append(dict(path=str(leaf),files=len(files),bytes=sum(f.stat().st_size for f in files)))
for current,dirs,names in os.walk(p):
 os.chown(current,owner.st_uid,owner.st_gid)
 for name in names:os.chown(str(Path(current)/name),owner.st_uid,owner.st_gid)
print(json.dumps(rows))'''
    p=subprocess.run(['docker','exec','-e','E33_OUT='+container_path(outdir),CONTAINER,'python3','-c',code],check=True,capture_output=True,text=True,timeout=90)
    return json.loads(p.stdout)
def run(record):
    rid=record['run_id'];control=ROOT/'evidence-e33/control'/rid;control.mkdir(parents=True,exist_ok=False)
    outdir=ROOT/'evidence-e33/di-detailed'/rid
    context=dict(run_id=rid,case_id='L32K-C32-'+record['kind'].upper(),phase=record['kind'],out_dir=container_path(outdir))
    status=dict(run_id=rid,profiled=record['profiled'],phase='idle_gate',started_utc=stamp(),qualified=None)
    save(control/'status.json',status);save(control/'context.json',context);idle(control/'preflight.json')
    profiled=False;sample=control/'scheduler.jsonl';stop=control/'scheduler.stop'
    try:
        if record['kind'] in ('normal','hot'):
            save(control/'before-di-status.json',actors(control))
            subprocess.run(['docker','exec','-d',CONTAINER,'python3',str(CONTAINER_ROOT/'tools/e33_process_sampler.py'),'--framework','di','--run-id',rid,'--out',container_path(sample),'--stop-file',container_path(stop),'--actors-file',container_path(control/'actors.json'),'--all-threads-interval','5'],check=True)
            ready=wait_file(sample.with_suffix('.ready.json'));assert len([x for x in ready['actors'] if x['role']=='worker'])==8
        if record['profiled']:
            status['phase']='profiler_start';save(control/'status.json',status)
            start=rpc('/e33/profile/start',context,1000);profiled=True;save(control/'start-rpc.json',start)
            assert start['ack_count']==8 and start['global_idle_at_start']['ok']
            assert {x['physical_device_id'] for x in start['ranks']}==set(range(8))
            assert all(not x['profiler']['errors'] and x['profiler']['api']['record_shapes'] and x['profiler']['api']['experimental_config_requested']['profiler_level']=='Level1' for x in start['ranks'])
        status['phase']='workload';save(control/'status.json',status)
        subprocess.run([sys.executable,str(ROOT/'tools/e33_run_window.py'),'--framework','di','--run-id',rid,'--timeout','1200'],check=True,timeout=1300)
        idle(control/'drained.json')
        if sample.exists():stop.touch();wait_file(sample.with_suffix('.done.json'),30)
        if profiled:
            status['phase']='profiler_export_after_drain';save(control/'status.json',status)
            ended=rpc('/e33/profile/stop',context,3700);profiled=False;save(control/'stop-rpc.json',ended)
            raw=return_artifacts(outdir);save(control/'exported-traces.json',dict(raw_exports=raw,raw_count=len(raw),qualified=None))
            assert ended['ack_count']==8 and not ended['missing_ranks'] and not ended['errors']
            assert len(raw)==8 and all(x['files']>10 and x['bytes']>0 for x in raw)
        subprocess.run([sys.executable,str(ROOT/'tools/e33_review_window.py'),'--framework','di','--run-id',rid,'--corpus',record['corpus'],'--requests',str(record['requests']),'--concurrency',str(record['concurrency']),'--output',str(record['output_tokens']),'--cached',str(record['cached_tokens'])],check=True,timeout=120)
        jit=json.loads((ROOT/'evidence-e33/windows'/rid/'native-jit.json').read_text())
        status.update(phase='completed',completed_utc=stamp(),workload_review_passed=True,changed_jit_metadata=jit['changed_metadata_count'])
        save(control/'status.json',status);print(json.dumps(status),flush=True)
    except BaseException as exc:
        status.update(phase='failed',error=repr(exc),failed_utc=stamp());save(control/'status.json',status)
        if sample.exists():stop.touch()
        if profiled:
            try:idle(control/'failure-drain.json');save(control/'failure-stop.json',rpc('/e33/profile/stop',context,3700));return_artifacts(outdir)
            except BaseException as stop_exc:save(control/'failure-profiler-state.json',dict(error=repr(stop_exc),state='unknown; inspect before more work'))
        raise
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['normal','hot'],required=True);parser.add_argument('--only');a=parser.parse_args()
    records=json.loads((ROOT/'data/E33-detailed-window-manifest.json').read_text())['records'];lookup={x['run_id']:x for x in records}
    names=['NORMAL-U1','NORMAL-P','NORMAL-U2'] if a.mode=='normal' else ['HOT-PATH-W1','HOT-PATH-W2','HOT-U1-PREFIX','HOT-U1','HOT-P-PREFIX','HOT-P','HOT-U2-PREFIX','HOT-U2']
    if a.only:assert a.only in names;names=[a.only]
    for name in names:run(lookup['E33-DI-L32K-C32-D1-'+name])
if __name__=='__main__':main()
