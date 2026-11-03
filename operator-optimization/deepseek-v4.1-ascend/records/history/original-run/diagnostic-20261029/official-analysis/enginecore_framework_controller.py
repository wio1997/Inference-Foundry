"""Single reviewable launcher/controller for 169 EngineCore CPU sampling.

Only --execute submits work. It runs frozen warmup/seed and C8 P90 32K/1K
against the user's baseline. No server reset, restart, code injection or profiler
configuration is performed. py-spy and procfs are external to target processes.
"""
import base64
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

BASE='wio-vllm-v41-repro169-20261028'
BASE_ID='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
STAGE='wio-vllm-v41-repro169-stage-20261028'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
APP='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'
BIN=APP+'/tools/py-spy-0.4.1-aarch64-972918b5'
BIN_SHA='972918b5d6af92649ec086ffdd89cefa0d725fd14844434c766f58494a1c4f8e'
EXPECTED_PROMPT_SHA='755b823003b4d200a4590424a245cdfb484d9b8c3d0b5225a7a058790b87e8c8'

def run(args,timeout=60):
    p=subprocess.run(args,check=False,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if p.returncode: raise RuntimeError('command failed ('+str(p.returncode)+'): '+p.stderr[-3000:])
    return p.stdout

def host_guard():
    b=json.loads(run(['docker','inspect',BASE]))[0]
    if b['Id']!=BASE_ID or not b['State']['Running']: raise RuntimeError('baseline identity/running mismatch')
    own={int(x.strip()) for x in run(['docker','top',BASE,'-eo','pid']).splitlines() if x.strip().isdigit()}
    raw=run(['docker','exec',STAGE,'npu-smi','info'],timeout=30)
    if 'Process id' not in raw: raise RuntimeError('npu-smi process table unavailable')
    pids=set()
    for line in raw.splitlines():
        c=[x.strip() for x in line.split('|')]
        if len(c)>4 and re.fullmatch(r'\d+\s+\d+',c[1]) and c[2].isdigit(): pids.add(int(c[2]))
    if len(pids)<16 or not pids<=own: raise RuntimeError('NPU owner mismatch '+repr(sorted(pids-own)))
    return {'npu_pids':sorted(pids),'baseline_id':b['Id']}

def host_health_idle():
    code='''import urllib.request,json
o=urllib.request.build_opener(urllib.request.ProxyHandler({}));u='http://127.0.0.1:8172'
assert o.open(u+'/health',timeout=5).status==200
s=o.open(u+'/metrics',timeout=5).read().decode()
rows=[x for x in s.splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
assert len(rows)>=4 and all(float(x.rsplit(' ',1)[1])==0 for x in rows),rows
print(json.dumps(rows))'''
    return json.loads(run(['docker','exec',BASE,'python3','-c',code],timeout=20))

def inside_exec(self_b64,proc_b64,mode,runid):
    source='import base64;exec(compile(base64.b64decode('+repr(self_b64)+'),"enginecore_framework_controller.py","exec"))'
    return ['docker','exec',BASE,'python3','-c',source,'--inside',mode,runid,proc_b64]

def abort_inside(runid,hard=False):
    path=APP+'/logs/'+runid+'/control.json'
    code='''import json,os,signal,pathlib,sys
p=pathlib.Path(sys.argv[1])
sig=signal.SIGKILL if sys.argv[3]=='hard' else signal.SIGTERM
if p.is_file():
 d=json.loads(p.read_text());pid=d['pid'];raw=pathlib.Path('/proc/'+str(pid)+'/stat').read_text();f=raw[raw.rfind(')')+2:].split()
 cmd=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes()
 if int(f[19])==d['starttime_ticks'] and sys.argv[2].encode() in cmd:
  for child in d.get('children',[]):
   try:
    raw=pathlib.Path('/proc/'+str(child['pid'])+'/stat').read_text();c=raw[raw.rfind(')')+2:].split()
    if int(c[19])==child['starttime_ticks'] and int(c[1])==pid:os.kill(child['pid'],signal.SIGTERM)
   except (FileNotFoundError,ProcessLookupError):pass
  os.kill(pid,sig)
else:
 matches=[]
 for x in pathlib.Path('/proc').iterdir():
  if not x.name.isdigit():continue
  try: args=(x/'cmdline').read_bytes().split(b'\\0')
  except OSError:continue
  if b'--inside' in args and (b'measure' in args or b'warmup' in args) and sys.argv[2].encode() in args:matches.append(int(x.name))
 if len(matches)==1:os.kill(matches[0],sig)'''
    subprocess.run(['docker','exec',BASE,'python3','-c',code,path,runid,'hard' if hard else 'soft'],capture_output=True,text=True,timeout=15)

def guarded_exec(self_b64,proc_b64,runid,mode):
    proc=subprocess.Popen(inside_exec(self_b64,proc_b64,mode,runid),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    failure=None
    try:
        deadline=time.monotonic()+240
        while proc.poll() is None:
            if time.monotonic()>deadline: failure=mode+' deadline';break
            time.sleep(2)
            try: host_guard()
            except Exception as exc: failure='ownership check: '+repr(exc);break
        if failure:
            abort_inside(runid)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                abort_inside(runid,hard=True)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.terminate()
            raise RuntimeError(failure)
        out,err=proc.communicate(timeout=10)
        print(out,flush=True)
        if proc.returncode: raise RuntimeError(mode+' failed: '+err[-4000:])
    finally:
        if proc.poll() is None:
            abort_inside(runid)
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: proc.terminate()

def host(self_b64,proc_b64,runid):
    try:
        first=host_guard();idle=host_health_idle()
        print(json.dumps({'state':'WARMUP','run_id':runid,'owner':first,'idle':idle}),flush=True)
        guarded_exec(self_b64,proc_b64,runid,'warmup')
        second=host_guard();idle=host_health_idle()
        print(json.dumps({'state':'MEASUREMENT','owner':second,'idle':idle}),flush=True)
        guarded_exec(self_b64,proc_b64,runid,'measure')
    finally:
        final={'state':'FINAL_HEALTH','run_id':runid,'at_wall_ns':time.time_ns()}
        try: final['owner']=host_guard()
        except Exception as exc: final['owner_error']=repr(exc)
        deadline=time.monotonic()+60
        while True:
            try: final['health_idle']=host_health_idle();break
            except Exception as exc:
                final['health_error']=repr(exc)
                if time.monotonic()>=deadline: break
                time.sleep(3)
        if 'health_idle' in final:final.pop('health_error',None)
        print(json.dumps(final),flush=True)
        source='''import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
(p/'FINAL_HEALTH.json').write_text(sys.argv[2])'''
        try: run(['docker','exec',BASE,'python3','-c',source,APP+'/logs/'+runid,json.dumps(final,indent=2)],timeout=20)
        except Exception as exc: print(json.dumps({'final_health_save_error':repr(exc)}),flush=True)

def opener(): return urllib.request.build_opener(urllib.request.ProxyHandler({}))
def request(path,body=None,headers=None,timeout=900):
    data=None if body is None else json.dumps(body).encode()
    return opener().open(urllib.request.Request('http://127.0.0.1:8172'+path,data=data,
                    headers={'Content-Type':'application/json',**(headers or {})}),timeout=timeout)
def metrics():
    with request('/metrics',timeout=15) as r: return r.read().decode()
def metric_by_engine(raw,name):
    values={'0':[],'1':[]}
    for line in raw.splitlines():
        if not line.startswith('vllm:'+name+'{'): continue
        m=re.search(r'(?:\{|,)engine="([01])"(?:,|\})',line)
        if m: values[m.group(1)].append(float(line.rsplit(' ',1)[1]))
    return {k:sum(v) if v else None for k,v in values.items()}
NAMES=['prefix_cache_queries_total','prefix_cache_hits_total','prompt_tokens_total',
       'generation_tokens_total','request_success_total','num_preemptions_total',
       'spec_decode_num_draft_tokens_total','spec_decode_num_accepted_tokens_total']
def delta(before,after):
    a={n:metric_by_engine(before,n) for n in NAMES};b={n:metric_by_engine(after,n) for n in NAMES}
    per={n:{k:(b[n][k]-a[n][k] if a[n][k] is not None and b[n][k] is not None else None)
             for k in ('0','1')} for n in NAMES}
    total={n:(sum(per[n].values()) if all(x is not None for x in per[n].values()) else None) for n in NAMES}
    return per,total

def complete(row,phase,index,stream,barrier):
    body={'model':'dsv41','prompt':row['prompt'],'max_tokens':row['max_tokens'],
          'temperature':0,'ignore_eos':True,'stream':stream,'request_id':phase+'-'+str(index)}
    if stream: body['stream_options']={'include_usage':True}
    headers={'X-data-parallel-rank':str(row['dp_rank'])} if row.get('dp_rank') is not None else {}
    barrier.wait(timeout=120)
    result={'start_ns':time.monotonic_ns(),'start_wall_ns':time.time_ns(),'chunks':[],'text':''}
    with request('/v1/completions',body,headers) as r:
        if not stream:
            data=json.load(r);result.update(usage=data['usage'],text=data['choices'][0]['text'])
        else:
            for raw in r:
                line=raw.decode()
                if not line.startswith('data:'): continue
                line=line[5:].strip()
                if line=='[DONE]': continue
                data=json.loads(line);now=time.monotonic_ns()
                text=''.join(x.get('text','') for x in data.get('choices',[]))
                result['chunks'].append({'at_ns':now,'nonempty':bool(text)})
                if text and 'first_nonempty_ns' not in result: result['first_nonempty_ns']=now
                result['text']+=text
                if data.get('usage'):result['usage']=data['usage']
    result.update(end_ns=time.monotonic_ns(),end_wall_ns=time.time_ns())
    return result

def parallel(rows,phase,stream=False):
    barrier=threading.Barrier(len(rows))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:
        return list(pool.map(lambda x:complete(x[1],phase,x[0],stream,barrier),enumerate(rows)))
def input_rows(folder):
    matches=sorted(folder.rglob('*.requests.jsonl'))
    if len(matches)!=1: raise RuntimeError('unexpected frozen request files '+repr(matches))
    rows=[json.loads(x) for x in matches[0].read_text().splitlines()]
    if len(rows)!=8: raise RuntimeError('expected eight requests')
    return rows
def seed(folder,label,out):
    rows=json.loads((folder/'warmup.json').read_text())
    results=parallel(rows,label+'-seed')
    if not all(x['usage']['prompt_tokens']==29696 and x['usage']['completion_tokens']==1 for x in results):
        raise RuntimeError('invalid seed token counts')
    (out/(label+'-seed.json')).write_text(json.dumps(results,ensure_ascii=False))
def assert_case(rows,results,total):
    ok=(len(rows)==len(results)==8 and all(x.get('usage',{}).get('prompt_tokens')==32768 and
        x.get('usage',{}).get('completion_tokens')==1024 for x in results) and
        total['prefix_cache_queries_total']==262144 and total['prefix_cache_hits_total']==235904 and
        total['prompt_tokens_total']==262144 and total['generation_tokens_total']==8192 and
        total['request_success_total']==8 and total['num_preemptions_total']==0)
    if not ok: raise RuntimeError('frozen case mismatch '+repr(total))
def request_rows(rows):
    return [{'prompt':r['question'],'max_tokens':1024,
             **({'dp_rank':r['dp_rank']} if r.get('dp_rank') is not None else {})} for r in rows]

def identify():
    found={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: name=(p/'cmdline').read_bytes().split(b'\0')[0].decode().strip()
        except (OSError,UnicodeDecodeError): continue
        if name in ('VLLM::EngineCore_DP0','VLLM::EngineCore_DP1'):
            if name in found: raise RuntimeError('duplicate engine '+name)
            raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
            found[name]={'pid':int(p.name),'starttime_ticks':int(fields[19])}
    if set(found)!={'VLLM::EngineCore_DP0','VLLM::EngineCore_DP1'}: raise RuntimeError('engine set '+repr(found))
    return found

def inside(mode,runid,proc_b64):
    if not re.fullmatch(r'framework-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}',runid): raise RuntimeError('bad run id')
    out=Path(APP+'/logs')/runid
    frozen=Path(APP+'/benchmark-20261028/workload/frozen')
    if mode=='warmup': out.mkdir(exist_ok=False)
    elif mode=='measure' and not (out/'warmup-valid.json').is_file(): raise RuntimeError('warmup incomplete')
    self_stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
    (out/'control.json').write_text(json.dumps({'pid':os.getpid(),'starttime_ticks':int(self_stat[19]),'run_id':runid,'children':[]}))
    if hashlib.sha256(Path(BIN).read_bytes()).hexdigest()!=BIN_SHA: raise RuntimeError('py-spy binary mismatch')
    with request('/health',timeout=10) as response:
        if response.status!=200: raise RuntimeError('unhealthy service')
    if mode=='warmup':
        folder=frozen/'warmup';seed(folder,'warmup',out);rows=input_rows(folder)
        before=metrics();(out/'warmup-metrics-before.txt').write_text(before)
        results=parallel(request_rows(rows),runid+'-warmup',False)
        after=metrics();(out/'warmup-metrics-after.txt').write_text(after)
        per,total=delta(before,after)
        (out/'warmup-results.json').write_text(json.dumps(results,ensure_ascii=False))
        (out/'warmup-counters.json').write_text(json.dumps({'per_engine':per,'total':total},indent=2))
        assert_case(rows,results,total)
        seed(frozen/'measure1','measure1',out)
        (out/'warmup-valid.json').write_text(json.dumps({'valid':True,'time_ns':time.time_ns()}))
        print(json.dumps({'state':'WARMUP_AND_MEASURE_SEED_COMPLETE','output':str(out)}));return
    if mode!='measure': raise RuntimeError('unknown mode')
    rows=input_rows(frozen/'measure1')
    prompt_sha=hashlib.sha256(json.dumps([r['question'] for r in rows],ensure_ascii=False).encode()).hexdigest()
    if prompt_sha!=EXPECTED_PROMPT_SHA: raise RuntimeError('frozen prompt hash changed')
    engine=identify();self_stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
    control={'pid':os.getpid(),'starttime_ticks':int(self_stat[19]),'run_id':runid,'children':[]}
    def save_control():
        temp=out/'control.json.tmp';temp.write_text(json.dumps(control));temp.replace(out/'control.json')
    save_control()
    samplers=[];threads=[];windows={};old_handler=signal.getsignal(signal.SIGTERM)
    def abort(signum,frame): raise RuntimeError('own measurement cancelled after owner/deadline check')
    signal.signal(signal.SIGTERM,abort)
    def watch(label,proc):
        code=proc.wait()
        windows[label].update(exit_code=code,ended_monotonic_ns=time.monotonic_ns(),ended_wall_ns=time.time_ns())
    def start(label,args,env=None):
        err=(out/(label+'.stderr')).open('wb');stdout=(out/(label+'.stdout')).open('wb')
        mark={'launch_monotonic_ns':time.monotonic_ns(),'launch_wall_ns':time.time_ns(),'argv':args}
        p=subprocess.Popen(args,stdout=stdout,stderr=err,env=env)
        mark.update(pid=p.pid,launched_monotonic_ns=time.monotonic_ns());windows[label]=mark
        samplers.append((p,stdout,err))
        raw=Path('/proc/'+str(p.pid)+'/stat').read_text();f=raw[raw.rfind(')')+2:].split()
        control['children'].append({'pid':p.pid,'starttime_ticks':int(f[19]),'label':label});save_control()
        t=threading.Thread(target=watch,args=(label,p),daemon=True);t.start();threads.append(t)
    try:
        before=metrics();(out/'measure-metrics-before.txt').write_text(before)
        for name in sorted(engine):
            suffix=name.rsplit('_',1)[-1];path=out/('pyspy-'+suffix+'.speedscope.json')
            start('pyspy-'+suffix,[BIN,'record','--nonblocking','--idle','--threads','--rate','50',
                                  '--duration','20','--format','speedscope','--pid',str(engine[name]['pid']),
                                  '--output',str(path)])
        env=dict(os.environ);env['WIO_PROC_OUT']=str(out/'enginecore-proc.json')
        proc_source=base64.b64decode(proc_b64).decode()
        start('proc',['python3','-c',proc_source],env)
        results=parallel(request_rows(rows),runid+'-measure',True)
        after=metrics();(out/'measure-metrics-after.txt').write_text(after)
        per,total=delta(before,after)
        (out/'measure-results.json').write_text(json.dumps(results,ensure_ascii=False))
        (out/'measure-counters.json').write_text(json.dumps({'per_engine':per,'total':total},indent=2))
        assert_case(rows,results,total)
        for p,_,_ in samplers: p.wait(timeout=35)
        for t in threads:t.join(timeout=5)
        if any(windows[k].get('exit_code')!=0 for k in windows): raise RuntimeError('one or more samplers failed')
        outputs={name:(out/('pyspy-'+name.rsplit('_',1)[-1]+'.speedscope.json')).is_file() for name in engine}
        if not all(outputs.values()): raise RuntimeError('missing speedscope files '+repr(outputs))
        (out/'result.json').write_text(json.dumps({'valid':True,'prompt_sha256':prompt_sha,'engine':engine,
            'sampler_windows':windows,'per_engine':per,'total':total,
            'limits':'nonblocking stacks may be partial; --idle includes waiting; samples are not CPU times',
            'profile_vs_baseline':'sampling overhead is present; do not treat this run as an uninstrumented baseline'},indent=2))
        print(json.dumps({'state':'COMPLETE','output':str(out),'windows':windows}),flush=True)
    finally:
        for p,stdout,err in samplers:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
            stdout.close();err.close()
        for t in threads:t.join(timeout=5)
        (out/'sampler-windows.json').write_text(json.dumps(windows,indent=2))
        signal.signal(signal.SIGTERM,old_handler)

def main():
    if sys.argv[1:]==['--execute']:
        source=Path(__file__).read_bytes()
        proc=Path(__file__).with_name('sample_engine_core_proc.py').read_bytes()
        compile(source,__file__,'exec');compile(proc,'sample_engine_core_proc.py','exec')
        self_b64=base64.b64encode(source).decode();proc_b64=base64.b64encode(proc).decode()
        runid='framework-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        code='import base64;exec(compile(base64.b64decode('+repr(self_b64)+'),"enginecore_framework_controller.py","exec"))'
        r=subprocess.run([SSH,shlex.join(['python3','-c',code,'--host',self_b64,proc_b64,runid])],timeout=3600)
        if r.returncode: raise SystemExit(r.returncode)
    elif len(sys.argv)==5 and sys.argv[1]=='--host': host(sys.argv[2],sys.argv[3],sys.argv[4])
    elif len(sys.argv)==5 and sys.argv[1]=='--inside': inside(sys.argv[2],sys.argv[3],sys.argv[4])
    else: raise SystemExit('Review first; use --execute to run frozen load and sample.')

if __name__=='__main__': main()
