"""169 detached controller. Invoked only via launch_official_profile.py.

Host activity is limited to Docker control. All persistent outputs are written
through the existing stage container into the wio/main task bind mount.
"""
import base64
import datetime
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback

BASE = 'wio-vllm-v41-repro169-20261028'
BASE_ID = 'b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
IMAGE = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
STAGE = 'wio-vllm-v41-repro169-stage-20261028'
ROOT = '/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028'
WORK = '/work/vllm-ascend-reproduce-172-on-169-20261028'
REL = '/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029'
APP = '/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'
SERVE_SHA = 'c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
RUN = os.environ['WIO_RUN_ID']
assert re.fullmatch(r'official-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}', RUN)
DIAG = 'wio-vllm-v41-'+RUN
HOST_RUN = ROOT+REL+'/'+RUN
STAGE_RUN = WORK+REL+'/'+RUN
CLIENT_B64 = os.environ['WIO_CLIENT_B64']
PROFILE = {'profiler':'torch','torch_profiler_dir':APP+'/official-traces',
           'torch_profiler_with_stack':False,'torch_profiler_with_memory':False,
           'torch_profiler_record_shapes':False,'ignore_frontend':False,
           'torch_profiler_dump_cuda_time_total':False,
           'max_iterations':12,'delay_iterations':0}
MAX_SECONDS = 3*60*60
DISK_RESERVE = 20*1024**3  # operational guard, not an estimate of trace size
TRACE_BUDGET = 32*1024**3  # bounded diagnostic budget, not a measured requirement

def cmd(args, timeout=60, check=True):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, check=check)

def inspect(name): return json.loads(cmd(['docker','inspect',name]).stdout)[0]
def stage_python(source, timeout=60):
    return cmd(['docker','exec',STAGE,'python3','-c',source], timeout=timeout).stdout
def stage_write(name, value):
    code = 'import pathlib,sys;p=pathlib.Path('+repr(STAGE_RUN+'/'+name)+');p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name("."+p.name+".upload");tmp.write_bytes(sys.stdin.buffer.read());tmp.replace(p)'
    subprocess.run(['docker','exec','-i',STAGE,'python3','-c',code],
                   input=json.dumps(value,ensure_ascii=False,indent=2)+'\n',text=True,
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90,check=True)
def status(state, **values):
    stage_write('controller-state.json', {'state':state,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'process_pid':os.getpid(),'run_id':RUN, **values})
def container_code(name, code, timeout=30):
    return cmd(['docker','exec',name,'python3','-c',code], timeout=timeout).stdout
def health(name, idle=False):
    code = '''import urllib.request,json
o=urllib.request.build_opener(urllib.request.ProxyHandler({}));u='http://127.0.0.1:8172'
s=o.open(u+'/health',timeout=5).status
m=o.open(u+'/metrics',timeout=5).read().decode()
rows=[x for x in m.splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
print(json.dumps({'health':s,'idle':len(rows)>=4 and all(float(x.rsplit(' ',1)[1])==0 for x in rows),'rows':rows}))'''
    result=json.loads(container_code(name,code))
    if result['health'] != 200 or (idle and not result['idle']): raise RuntimeError('service is not healthy/idle: '+repr(result))
    return result
def npu_snapshot():
    raw = cmd(['docker','exec',STAGE,'npu-smi','info'],timeout=30).stdout
    if 'Process id' not in raw: raise RuntimeError('npu-smi process table missing')
    pids=set()
    for row in raw.splitlines():
        col=[x.strip() for x in row.split('|')]
        if len(col)>4 and re.fullmatch(r'\d+\s+\d+',col[1]) and col[2].isdigit(): pids.add(int(col[2]))
    return pids,raw
def own_pids(name):
    data=cmd(['docker','top',name,'-eo','pid']).stdout
    return {int(x.strip()) for x in data.splitlines() if x.strip().isdigit()}
def owner_guard(name):
    pids,raw=npu_snapshot(); own=own_pids(name)
    if len(pids)<16 or not pids<=own: raise RuntimeError('NPU ownership/count mismatch: '+repr(sorted(pids-own)))
    return {'npu_pids':sorted(pids),'raw':raw}
def baseline_check():
    d=inspect(BASE)
    if d['Id']!=BASE_ID or d['Image']!=IMAGE: raise RuntimeError('baseline identity mismatch')
    return d
def serve_hash(name):
    return container_code(name,'import hashlib;print(hashlib.sha256(open('+repr(APP+'/scripts/serve.sh')+',"rb").read()).hexdigest())').strip()
def wait_health(name, timeout=2400):
    deadline=time.monotonic()+timeout
    last=None
    while time.monotonic()<deadline:
        if not inspect(name)['State']['Running']: raise RuntimeError(name+' exited while loading')
        try: return health(name,True)
        except Exception as exc: last=repr(exc)
        time.sleep(10)
    raise TimeoutError('health timeout '+str(last))
def free_guard():
    source='import shutil;print(shutil.disk_usage('+repr(STAGE_RUN)+').free)'
    free=int(stage_python(source).strip())
    if free < DISK_RESERVE: raise RuntimeError('task filesystem below 20 GiB operational reserve')
    return free

def prepare():
    d=baseline_check()
    if not d['State']['Running'] or serve_hash(BASE)!=SERVE_SHA: raise RuntimeError('baseline not running or frozen serve SHA mismatch')
    if any('instrumentation' in x or 'sitecustomize' in x for x in d['Config']['Env']):
        raise RuntimeError('baseline environment includes diagnostic instrumentation')
    guard={'health':health(BASE,True),'ownership':owner_guard(BASE),'inspect_id':d['Id'],'image':d['Image'],'serve_sha':SERVE_SHA}
    code='''from pathlib import Path
import base64,hashlib,json,shlex,shutil,subprocess
src=Path(WORK+'/runtime'); root=Path(TARGET); root.mkdir(parents=False,exist_ok=False)
dst=root/'runtime'
cache_size=int(subprocess.check_output(['du','-sb',str(src/'cache')]).split()[0]) if (src/'cache').exists() else 0
assert shutil.disk_usage(root).free >= cache_size+20*1024**3, 'insufficient disk for cache and operational reserve'
for n in ('scripts','config'): shutil.copytree(src/n,dst/n)
for n in ('cache','home','tmp','logs','official-traces','client-baseline','client-profile'): (dst/n).mkdir(parents=True,exist_ok=True)
if (src/'cache').exists(): shutil.copytree(src/'cache',dst/'cache',dirs_exist_ok=True,symlinks=True)
# client directories must not exist when client begins.
(dst/'client-baseline').rmdir();(dst/'client-profile').rmdir()
text=(dst/'scripts/serve.sh').read_text()
assert hashlib.sha256(text.encode()).hexdigest()==SHA
lines=text.splitlines();i=[i for i,x in enumerate(lines) if x.startswith('exec /usr/local/python3.12.13/bin/vllm serve ')]
assert len(i)==1
assert 'sitecustomize' not in text and '/instrumentation' not in text
lines[i[0]]+=' --profiler-config '+shlex.quote(json.dumps(PROFILE,separators=(',',':')))
lines.insert(i[0], 'export TMPDIR=/tmp')
(dst/'scripts/serve-official.sh').write_text('\\n'.join(lines)+'\\n')
(dst/'official_profile_client.py').write_bytes(base64.b64decode(CLIENT))
print(json.dumps({'script_sha':hashlib.sha256((dst/'scripts/serve-official.sh').read_bytes()).hexdigest()}))'''
    variables={'WORK':WORK,'TARGET':STAGE_RUN,'SHA':SERVE_SHA,'PROFILE':PROFILE,'CLIENT':CLIENT_B64}
    out=json.loads(stage_python('\n'.join(k+'='+repr(v) for k,v in variables.items())+'\n'+code, timeout=300))
    stage_write('preflight.json', {'baseline':guard,'prepared':out,'profile_config':PROFILE,
                                'disk_free_bytes':free_guard(), 'time':time.time_ns(),
                                'diagnostic_differences':['profiler configured; frontend CPU enabled',
                                   'VLLM_SERVER_DEV_MODE=1; VLLM_CUSTOM_SCOPES_FOR_PROFILING=1',
                                   'TMPDIR=/tmp isolated bind; temporary app/cache and bridge network',
                                   'observer runs npu-smi at 1 Hz'],
                                'coverage':'frontend CPU trace + worker CPU/NPU trace, client stages and per-engine metrics; scheduler is a separate engine process, so scheduler CPU scopes are not guaranteed captured'})
    cfg=d['HostConfig']
    args=['docker','create','--name',DIAG,'--label','wio.task='+RUN,'--privileged','--read-only',
          '--network','bridge','--ipc','private','--shm-size',str(cfg['ShmSize']),
          '--workdir',APP,'--entrypoint','bash']
    for env in d['Config']['Env']: args+=['--env',env]
    # Enable the development cache-reset route only in this isolated copy.
    args+=['--env','VLLM_SERVER_DEV_MODE=1']
    args+=['--env','VLLM_CUSTOM_SCOPES_FOR_PROFILING=1','--env','TMPDIR=/tmp']
    for u in cfg.get('Ulimits') or []: args+=['--ulimit',u['Name']+'='+str(u['Soft'])+':'+str(u['Hard'])]
    for cap in cfg.get('CapAdd') or []: args+=['--cap-add',cap]
    for opt in cfg.get('SecurityOpt') or []: args+=['--security-opt',opt]
    for m in d['Mounts']:
        if m['Type']!='bind': raise RuntimeError('unexpected non-bind mount')
        src=m['Source'];dest=m['Destination']
        if dest==APP: src=HOST_RUN+'/runtime'
        elif dest=='/tmp': continue
        elif m['RW']: raise RuntimeError('unexpected writable baseline mount '+dest)
        args+=['--mount','type=bind,src='+src+',dst='+dest+(',readonly' if not m['RW'] else '')]
    args+=['--mount','type=bind,src='+ROOT+'/runtime/benchmark-20261028,dst=/bench-inputs,readonly']
    args+=['--mount','type=bind,src='+HOST_RUN+'/runtime/tmp,dst=/tmp']
    for dev in cfg.get('Devices') or []:
        args+=['--device',dev['PathOnHost']+':'+dev['PathInContainer']+':'+dev['CgroupPermissions']]
    args += [IMAGE,'-c','exec bash '+APP+'/scripts/serve-official.sh >'+APP+'/logs/service.log 2>&1']
    diag_id=cmd(args,timeout=120).stdout.strip()
    stage_write('diagnostic-identity.json',{'name':DIAG,'id':diag_id,'label':RUN})
    created=inspect(DIAG)
    mounts=[m for m in created['Mounts'] if m['Destination']=='/tmp']
    if len(mounts)!=1 or mounts[0]['Source']!=HOST_RUN+'/runtime/tmp' or not mounts[0]['RW']:
        cmd(['docker','rm',DIAG],timeout=60)
        raise RuntimeError('diagnostic tmp bind is not the unique writable task bind')
    # Test the exact bind in a disposable CPU-only container before parking baseline.
    tmp_test=['docker','run','--rm','--label','wio.task='+RUN,'--read-only','--network','none','--entrypoint','python3',
              '--mount','type=bind,src='+HOST_RUN+'/runtime/tmp,dst=/tmp',
              '--env','TMPDIR=/tmp',IMAGE,'-c',
              'import os,tempfile; f=tempfile.NamedTemporaryFile(dir=os.environ["TMPDIR"]);f.write(b"ok");f.flush()']
    try: cmd(tmp_test,timeout=120)
    except BaseException:
        if inspect(DIAG)['Id']==diag_id: cmd(['docker','rm',DIAG],timeout=60)
        raise
    return diag_id

def stop_diag(diag_id):
    try: d=inspect(DIAG)
    except subprocess.CalledProcessError: return
    if d['Id']!=diag_id or d['Config']['Labels'].get('wio.task')!=RUN: raise RuntimeError('refuse to stop unrelated container')
    if d['State']['Running']: cmd(['docker','stop','--time','45',DIAG],timeout=100)

def restore(diag_id):
    stop_diag(diag_id)
    d=baseline_check()
    if not d['State']['Running']:
        pids,raw=npu_snapshot()
        if pids:
            status('RESTORE_BLOCKED_OTHER_NPU_PROCESSES',pids=sorted(pids),npu=raw)
            return False
        cmd(['docker','start',BASE],timeout=120)
    result=wait_health(BASE,timeout=2400)
    guard=owner_guard(BASE)
    sha=serve_hash(BASE)
    if sha!=SERVE_SHA: raise RuntimeError('restored serve SHA mismatch')
    status('RESTORED',health=result,ownership=guard,serve_sha=sha,baseline_id=BASE_ID,image=IMAGE,
           note='KV/prefix cache cold after restart; re-seed before benchmark')
    return True

def process_live(pid, start_marker):
    try:
        raw=open('/proc/'+str(pid)+'/stat').read()
        fields=raw[raw.rfind(')')+2:].split()
        return fields[0]!='Z' and fields[19]==start_marker
    except (FileNotFoundError,ProcessLookupError): return False

def watchdog(parent, parent_start, diag_id):
    deadline=time.monotonic()+MAX_SECONDS
    while time.monotonic()<deadline and process_live(parent,parent_start):
        time.sleep(5)
    if process_live(parent,parent_start):
        try: os.kill(parent,signal.SIGTERM)
        except ProcessLookupError: pass
        grace=time.monotonic()+30
        while time.monotonic()<grace and process_live(parent,parent_start): time.sleep(1)
        if process_live(parent,parent_start):
            try: os.kill(parent,signal.SIGKILL)
            except ProcessLookupError: pass
        grace=time.monotonic()+90
        while time.monotonic()<grace and process_live(parent,parent_start): time.sleep(1)
        if process_live(parent,parent_start):
            try: status('WATCHDOG_PARENT_STILL_ALIVE',parent_pid=parent)
            except BaseException: pass
            os._exit(1)
    # Never recover concurrently with the controller's finally block.
    try: restore(diag_id)
    except BaseException as exc:
        try: status('WATCHDOG_RESTORE_FAILED',error=repr(exc))
        except BaseException: pass
    os._exit(0)

def sample_loop(stop, samples):
    while not stop.is_set():
        ts=time.monotonic_ns();utc=time.time_ns()
        try:
            pids,raw=npu_snapshot();other=pids-own_pids(DIAG)
            samples.append({'monotonic_ns':ts,'epoch_ns':utc,'npu_smi':raw,'unknown_pids':sorted(other)})
            if other:
                if stop.is_set(): return
                os.kill(os.getpid(),signal.SIGUSR1)
                stage_write('external-npu-detected.json',{'pids':sorted(other),'npu_smi':raw,'monotonic_ns':ts})
                return
        except Exception as exc:
            samples.append({'monotonic_ns':ts,'epoch_ns':utc,'error':repr(exc)})
            if not stop.is_set():
                try: stage_write('npu-monitor-error.json',{'error':repr(exc),'monotonic_ns':ts})
                finally: os.kill(os.getpid(),signal.SIGUSR1)
            return
        stop.wait(max(0,1-(time.monotonic_ns()-ts)/1e9))

def route_check():
    code='''import json,urllib.request
o=urllib.request.build_opener(urllib.request.ProxyHandler({}))
p=json.load(o.open('http://127.0.0.1:8172/openapi.json',timeout=10))['paths']
assert '/start_profile' in p and '/stop_profile' in p and '/reset_prefix_cache' in p
print('ok')'''
    container_code(DIAG,code)
def runtime_versions():
    code='''import importlib.metadata,json
from pathlib import Path
names=['vllm','vllm-ascend','torch','torch-npu','ms_service_profiler']
out={}
for name in names:
 try: out[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError: out[name]=None
root=Path('/usr/local/Ascend/ascend-toolkit/latest')
info=root/'share/info/msserviceprofiler/version.info'
out['ascend_toolkit_link']=str(root)
out['ascend_toolkit_resolved']=str(root.resolve())
out['service_profiler_version_info_path']=str(info)
out['service_profiler_version_info']=info.read_text() if info.is_file() else None
print(json.dumps(out))'''
    return json.loads(container_code(DIAG,code))
def reset_prefix_cache():
    # DP broadcasts but returns the first result; case counters must validate.
    health(DIAG,True)
    code='''import json,urllib.request
o=urllib.request.build_opener(urllib.request.ProxyHandler({}))
req=urllib.request.Request('http://127.0.0.1:8172/reset_prefix_cache',data=b'',method='POST')
with o.open(req,timeout=120) as r: result=json.load(r)
assert result.get('success') is True,repr(result)
print(json.dumps(result))'''
    result=json.loads(container_code(DIAG,code,timeout=135))
    health(DIAG,True)
    stage_write('prefix-reset.json',{'result':result,'scope':'diagnostic only; only first DP return is exposed',
                                     'monotonic_ns':time.monotonic_ns()})
def client(phase):
    free_guard()
    args=['docker','exec','-e','WIO_PHASE='+phase,'-e','WIO_OUTPUT='+APP+'/client-'+phase,
          DIAG,'python3',APP+'/official_profile_client.py']
    proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    deadline=time.monotonic()+2400
    try:
        while proc.poll() is None:
            if time.monotonic()>deadline: raise TimeoutError('client timed out')
            free_guard()
            size=int(stage_python('import subprocess;print(int(subprocess.check_output(["du","-sb",'+repr(STAGE_RUN+'/runtime/official-traces')+']).split()[0]))').strip())
            if size>TRACE_BUDGET: raise RuntimeError('trace budget exceeded')
            time.sleep(5)
        out,err=proc.communicate(timeout=10)
        stage_write('client-'+phase+'-process.json',{'exit_code':proc.returncode,'stdout':out,'stderr':err})
        if proc.returncode: raise RuntimeError(phase+' client failed')
    finally:
        if proc.poll() is None: proc.terminate()

def wait_export():
    deadline=time.monotonic()+900; previous=None;stable=0;result={}
    code='''from pathlib import Path
import json
p=Path(TARGET)
roots=list(p.glob('*_ascend_pt'))
info=list(p.glob('**/profiler_info*.json'))
files=[x for x in p.rglob('*') if x.is_file()]
records=[]
for f in info:
 try:
  d=json.loads(f.read_text())
  records.append({'path':str(f),'rank_id':d.get('rank_id'),'cann_version':d.get('cann_version'),
                  'torch_npu_version':d.get('torch_npu_version'),'config':d.get('config')})
 except (OSError,ValueError) as e: records.append({'path':str(f),'read_error':repr(e)})
print(json.dumps({'roots':[str(x) for x in roots],'profiler_info':[str(x) for x in info],
 'profiler_info_fields':records,'file_count':len(files),'total_bytes':sum(x.stat().st_size for x in files)}))'''
    while time.monotonic()<deadline:
        free_guard()
        size=int(stage_python('import subprocess;print(int(subprocess.check_output(["du","-sb",'+repr(STAGE_RUN+'/runtime/official-traces')+']).split()[0]))').strip())
        if size>TRACE_BUDGET: raise RuntimeError('trace budget exceeded during export; data retained as incomplete')
        result=json.loads(stage_python('TARGET='+repr(STAGE_RUN+'/runtime/official-traces')+'\n'+code,timeout=120))
        marker=(len(result['roots']),len(result['profiler_info']),result['file_count'],result['total_bytes'])
        stable=stable+1 if marker==previous else 0
        if len(result['roots'])>=16 and len(result['profiler_info'])>=16 and stable>=2: break
        previous=marker;time.sleep(10)
    result['export_observation_complete'] = bool(len(result.get('roots',[]))>=16 and len(result.get('profiler_info',[]))>=16 and stable>=2)
    result['limitation']='File stability and 16 roots do not prove both windows or every rank/phase are covered.'
    stage_write('trace-inventory.json',result)

def validate_comparison():
    code='''from pathlib import Path
import json
p=Path(TARGET)/'runtime'
a=json.loads((p/'client-baseline/window-status.json').read_text())
b=json.loads((p/'client-profile/window-status.json').read_text())
assert a['valid'] and b['valid']
assert a['input_sha256']==b['input_sha256']
print(json.dumps({'identical_prompt_hash':a['input_sha256'],
                  'baseline_counters_by_engine':a['counter_deltas_by_engine'],
                  'profile_counters_by_engine':b['counter_deltas_by_engine']}))'''
    result=json.loads(stage_python('TARGET='+repr(STAGE_RUN)+'\n'+code))
    stage_write('comparison-validation.json',result)

def main():
    diag_id=prepare()
    parent=os.getpid()
    raw=open('/proc/'+str(parent)+'/stat').read()
    parent_start=raw[raw.rfind(')')+2:].split()[19]
    child=os.fork()
    if child==0: watchdog(parent,parent_start,diag_id)
    def external_occupant(signum,frame):
        raise RuntimeError('NPU ownership monitor requested stop; see monitor/occupant record')
    signal.signal(signal.SIGUSR1,external_occupant)
    samples=[];stop=threading.Event();sampler=threading.Thread(target=sample_loop,args=(stop,samples),daemon=True)
    try:
        status('PREPARED',diag_id=diag_id)
        # Repeat immediately before stopping only the explicitly identified baseline.
        baseline_check();health(BASE,True);owner_guard(BASE)
        cmd(['docker','stop','--time','45',BASE],timeout=100)
        pids,raw=npu_snapshot()
        if pids: raise RuntimeError('other processes appeared after parking baseline: '+repr(sorted(pids)))
        cmd(['docker','start',DIAG],timeout=120)
        sampler.start()
        status('DIAGNOSTIC_LOADING',diag_id=diag_id,watchdog_pid=child)
        wait_health(DIAG);owner_guard(DIAG);route_check()
        stage_write('runtime-versions.json',runtime_versions())
        status('WARMUP_LOAD_RUNNING')
        client('warmup')
        status('BASELINE_LOAD_RUNNING')
        client('baseline')
        # Reset only the temporary instance after both engines have drained.
        reset_prefix_cache()
        status('PROFILE_LOAD_RUNNING');client('profile')
        validate_comparison()
        status('EXPORT_WAIT')
        # Wait for rank exports, while treating observed completeness cautiously.
        wait_export()
    except BaseException as exc:
        try: stage_write('controller-error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        finally: raise
    finally:
        stop.set()
        if sampler.is_alive(): sampler.join(timeout=40)
        try: stage_write('npu-smi-samples.json',samples)
        finally: restore(diag_id)

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        try: stage_write('top-level-error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        except BaseException: pass
        raise
