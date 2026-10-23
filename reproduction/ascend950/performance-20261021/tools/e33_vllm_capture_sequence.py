#!/usr/bin/env python3
"""Run the frozen vLLM U1/P/U2 cohort; export only after native drain.

Presence of eight exported traces is a transport check, not qualification.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
CONTAINER='performance-20261021-vllm-official-ds41'
CONTAINER_ROOT=Path('/work/performance-20261021')
DIAG=ROOT/'evidence-e33/vllm'

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def container_path(path):return str(CONTAINER_ROOT/path.relative_to(ROOT))
def idle(out):
    subprocess.run([sys.executable,str(ROOT/'tools/e26_wait_idle.py'),'--framework','vllm','--base-url','http://127.0.0.1:18122','--timeout','600','--json-out',str(out)],check=True,stdout=subprocess.DEVNULL,timeout=620)
def rpc(name,out):
    before={'wall_ns':time.time_ns(),'perf_counter_ns':time.perf_counter_ns()}
    request=urllib.request.Request('http://127.0.0.1:18122/'+name,data=b'',method='POST')
    with urllib.request.urlopen(request,timeout=1200) as response:
        body=response.read().decode();code=response.status
    save(out,dict(before=before,after={'wall_ns':time.time_ns(),'perf_counter_ns':time.perf_counter_ns()},status=code,body=body))
    assert code==200
def wait_file(path,timeout=30):
    limit=time.monotonic()+timeout
    while time.monotonic()<limit:
        if path.exists():return json.loads(path.read_text())
        time.sleep(.2)
    raise TimeoutError(str(path))
def traces():
    return {str(p) for pattern in ('trace_view.json','*.pt.trace.json') for p in (DIAG/'profiles').rglob(pattern)}
def raw_exports(old_dirs):
    # Workers are daemon processes: native collection is valid but automatic
    # postprocessing cannot fork there. Parse later from a separate process.
    code='''from pathlib import Path
import json,os
r=Path('/work/performance-20261021');base=Path(os.environ['E33_RAW_ROOT']);old=set(json.loads(os.environ['E33_OLD_DIRS']))
owner=(r/'evidence-e33').stat();rows=[]
for p in sorted(base.iterdir()):
 if p.name in old:continue
 files=[f for f in p.rglob('*') if f.is_file()]
 rows.append(dict(path=str(p),files=len(files),bytes=sum(f.stat().st_size for f in files),info=[str(f) for f in files if f.name.startswith('profiler_info')]))
 for current,dirs,names in os.walk(p):
  os.chown(current,owner.st_uid,owner.st_gid)
  for name in names:os.chown(str(Path(current)/name),owner.st_uid,owner.st_gid)
print(json.dumps(rows))'''
    result=subprocess.run(['docker','exec','-e','E33_OLD_DIRS='+json.dumps(sorted(old_dirs)),'-e','E33_RAW_ROOT='+container_path(DIAG/'profiles'),CONTAINER,'python3','-c',code],capture_output=True,text=True,check=True,timeout=60)
    return json.loads(result.stdout)
def run(record):
    rid=record['run_id'];control=ROOT/'evidence-e33/control'/rid
    control.mkdir(parents=True,exist_ok=False)
    status={'run_id':rid,'profiled':record['profiled'],'started_utc':stamp(),'phase':'idle_gate','qualified':None}
    save(control/'status.json',status)
    idle(control/'preflight.json')
    save(DIAG/'window-context.json',{'run_id':rid,'changed_utc':stamp()});time.sleep(.3)
    profiled=False;sample_out=control/'scheduler.jsonl';stop=control/'scheduler.stop'
    old_traces=traces();old_dirs={p.name for p in (DIAG/'profiles').iterdir() if p.is_dir()}
    try:
        if record['kind'] in ('normal','hot'):
            cmd=['docker','exec','-d',CONTAINER,'python3',str(CONTAINER_ROOT/'tools/e33_process_sampler.py'),'--framework','vllm','--run-id',rid,'--out',container_path(sample_out),'--stop-file',container_path(stop),'--all-threads-interval','5']
            subprocess.run(cmd,check=True)
            ready=wait_file(sample_out.with_suffix('.ready.json'))
            assert len([x for x in ready['actors'] if x['role']=='worker'])==8,ready
        if record['profiled']:
            status['phase']='profiler_start';save(control/'status.json',status)
            rpc('start_profile',control/'start-rpc.json');profiled=True
        status['phase']='workload';save(control/'status.json',status)
        subprocess.run([sys.executable,str(ROOT/'tools/e33_run_window.py'),'--framework','vllm','--run-id',rid,'--timeout','1200'],check=True,timeout=1300)
        idle(control/'drained.json')
        if sample_out.exists():
            stop.touch();wait_file(sample_out.with_suffix('.done.json'),30)
        if profiled:
            status['phase']='profiler_export_after_drain';save(control/'status.json',status)
            rpc('stop_profile',control/'stop-rpc.json');profiled=False
            exported=sorted(traces()-old_traces)
            raw=raw_exports(old_dirs)
            save(control/'exported-traces.json',{'traces':exported,'count':len(exported),'raw_exports':raw,'raw_count':len(raw),'qualified':None,'offline_parse_required':len(exported)!=8,'note':'Raw presence only; clock, eight physical devices, complete cohort and dropped records need independent qualification.'})
            assert len(raw)==8 and all(x['files']>10 and x['bytes']>0 and x['info'] for x in raw),('Expected eight nonempty native worker exports',raw)
        args=[sys.executable,str(ROOT/'tools/e33_review_window.py'),'--framework','vllm','--run-id',rid,'--corpus',record['corpus'],'--requests',str(record['requests']),'--concurrency',str(record['concurrency']),'--output',str(record['output_tokens']),'--cached',str(record['cached_tokens'])]
        subprocess.run(args,check=True,timeout=120)
        status.update(phase='completed',completed_utc=stamp(),workload_review_passed=True)
        save(control/'status.json',status);print(json.dumps(status),flush=True)
    except BaseException as exc:
        status.update(phase='failed',error=repr(exc),failed_utc=stamp());save(control/'status.json',status)
        if sample_out.exists():stop.touch()
        if profiled:
            try:
                idle(control/'failure-drain.json');rpc('stop_profile',control/'failure-stop-rpc.json')
            except BaseException as stop_exc:
                save(control/'failure-profiler-state.json',{'error':repr(stop_exc),'state':'unknown; inspect before more work'})
        raise

def main():
    global DIAG
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['normal','hot'],required=True);parser.add_argument('--only',help='Run one not-yet-executed frozen window suffix, e.g. NORMAL-U2');parser.add_argument('--revision',choices=['','D1'],default='');args=parser.parse_args()
    manifest='data/E33-detailed-window-manifest.json' if args.revision else 'data/E33-window-manifest.json'
    if args.revision:DIAG=ROOT/'evidence-e33/vllm-detailed'
    records=json.loads((ROOT/manifest).read_text())['records']
    byid={r['run_id']:r for r in records}
    prefix='E33-V-L32K-C32-'+(args.revision+'-' if args.revision else '')
    names=['NORMAL-U1','NORMAL-P','NORMAL-U2'] if args.mode=='normal' else ['HOT-PATH-W1','HOT-PATH-W2','HOT-U1-PREFIX','HOT-U1','HOT-P-PREFIX','HOT-P','HOT-U2-PREFIX','HOT-U2']
    if args.only:
        assert args.only in names
        names=[args.only]
    for name in names:run(byid[prefix+name])
if __name__=='__main__':main()
