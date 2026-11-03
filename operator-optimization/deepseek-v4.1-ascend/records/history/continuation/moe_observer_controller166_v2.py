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

BASE = 'wio-vllm-v41-repro166-20261030'
BASE_ID = '8757dcc7a2f8db672bbf6ddfc1ed29e6cff74d5dd89199770b7b6f3b27fd48e3'
IMAGE = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
STAGE = 'wio-vllm-v41-repro166-stage-20261030'
ROOT = '/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030'
WORK = '/work/vllm-ascend-reproduce-172-on-166-20261030'
REL = '/analysis/whole-scope-continuation-20261030'
APP = '/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'
SERVE_SHA = 'c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
RUN = os.environ['WIO_RUN_ID']
assert re.fullmatch(r'observer-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}', RUN)
DIAG = 'wio-vllm-v41-'+RUN
HOST_RUN = ROOT+REL+'/'+RUN
STAGE_RUN = WORK+REL+'/'+RUN
CLIENT_B64 = ''  # active controller copies existing server client directly
PROFILE = {'profiler':'torch','torch_profiler_dir':APP+'/official-traces',
           'torch_profiler_with_stack':False,'torch_profiler_with_memory':False,
           'torch_profiler_record_shapes':False,'ignore_frontend':False,
           'torch_profiler_dump_cuda_time_total':False,
           'max_iterations':12,'delay_iterations':0}
MAX_SECONDS = 60*60
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

def sample_loop(stop,samples):
    pending={}
    while not stop.is_set():
        ts=time.monotonic_ns()
        try:
            pids,raw=npu_snapshot();own=own_pids(DIAG);d=inspect(DIAG);diag_id=d['Id'];confirmed_other=[];unknown=[]
            for pid in pids-own:
                proc=Path('/proc')/str(pid)
                try:
                    cg=(proc/'cgroup').read_text()
                    if diag_id in cg:continue
                    confirmed_other.append(pid)
                except FileNotFoundError:unknown.append(pid)
            for pid in unknown:pending[pid]=pending.get(pid,0)+1
            for pid in list(pending):
                if pid not in unknown:pending.pop(pid,None)
            persistent=[pid for pid,n in pending.items() if n>=3]
            samples.append({'monotonic_ns':ts,'epoch_ns':time.time_ns(),'npu_smi':raw,'confirmed_other_pids':confirmed_other,'disappeared_or_unattributed_pids':unknown,'ownership_snapshot_not_atomic':True})
            if confirmed_other or persistent:
                if stop.is_set():return
                stage_write('ownership-stop.json',{'confirmed_other_pids':confirmed_other,'persistent_unattributed_pids':persistent,'scope':'No external job is killed; stop only own diagnostic','monotonic_ns':ts})
                os.kill(os.getpid(),signal.SIGUSR1);return
        except Exception as exc:
            samples.append({'monotonic_ns':ts,'error':repr(exc)})
            if not stop.is_set():
                stage_write('npu-monitor-error.json',{'error':repr(exc),'monotonic_ns':ts});os.kill(os.getpid(),signal.SIGUSR1)
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

# These definitions replace the legacy profiler preparation/client/main.
# Active path collects tensor values only, without a profiler.
import fcntl
import shutil
import tarfile
from pathlib import Path

OBSERVER_BUDGET=256*1024**2
DRAFT_ROOT=Path(ROOT+REL)/'observer-draft-v3'
BASELINE_RELEASE_REFERENCE=Path('/data/tiankuan/wio/migration-166-20261030/npu-after-stop.txt')

def json_write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def hbm_rows(raw):
    out={};board=None
    for row in raw.splitlines():
        a=[x.strip() for x in row.split('|')]
        if len(a)>3 and re.fullmatch(r'\d+\s+Ascend910',a[1]):board=int(a[1].split()[0])
        if len(a)>3 and board is not None and re.fullmatch(r'\d+\s+\d+',a[1]) and ':' in a[2]:
            pairs=re.findall(r'(\d+)\s*/\s*(\d+)',row)
            if not pairs:raise RuntimeError('Chip HBM field missing')
            chip=int(a[1].split()[0]);used,total=map(int,pairs[-1]);out[(board,chip)]=used
    if len(out)!=16:raise RuntimeError('Could not parse complete 16-chip HBM table')
    return out

def wait_released(timeout=120):
    ref=BASELINE_RELEASE_REFERENCE.read_text();refp=set()
    for line in ref.splitlines():
        a=[x.strip() for x in line.split('|')]
        if len(a)>4 and re.fullmatch(r'\d+\s+\d+',a[1]) and a[2].isdigit():refp.add(int(a[2]))
    if refp:raise RuntimeError('Empirical idle reference contains device processes')
    refh=hbm_rows(ref);deadline=time.monotonic()+timeout;stable=0;last=None
    while time.monotonic()<deadline:
        pids,raw=npu_snapshot();cur=hbm_rows(raw)
        # Policy envelope, not a mathematical proof of every allocation owner.
        good=not pids and all(cur[k]<=refh[k]+1024 for k in refh)
        stable=stable+1 if good else 0;last={'pids':sorted(pids),'reported_hbm_MiB':{str(k):v for k,v in cur.items()},'policy':'two empty PID snapshots; per-chip HBM<=own previously observed idle+1024MiB; not universal ownership proof'}
        if stable>=2:return last
        if pids:raise RuntimeError('Other NPU processes appeared after stopping own instance: '+repr(sorted(pids)))
        time.sleep(2)
    raise RuntimeError('HBM release policy not met: '+repr(last))

def prepare():
    d=baseline_check();stage=inspect(STAGE)
    if stage['Id']!='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d' or stage['Image']!=IMAGE or not stage['State']['Running']:raise RuntimeError('Stage identity changed')
    if not d['State']['Running'] or serve_hash(BASE)!=SERVE_SHA:raise RuntimeError('Baseline unavailable or serve script changed')
    if dict(x.split('=',1) for x in d['Config']['Env']).get('CANN_VERSION')!='9.2.0':raise RuntimeError('CANN differs')
    guard={'health':health(BASE,True),'ownership':owner_guard(BASE),'baseline':{'id':d['Id'],'image':d['Image'],'state':d['State']}}
    reference=json.loads((Path(ROOT+REL)/'reference-latest.json').read_text())
    if reference['state']!='PASS_FIXED_C8_REFERENCE':raise RuntimeError('Full-case reference missing')
    draft=json.loads((DRAFT_ROOT/'PLAN.json').read_text());pre=json.loads((DRAFT_ROOT/'import-preflight.json').read_text())
    if pre['status']!='PASS_LAZY_IMPORT_ONLY':raise RuntimeError('Observer lazy import preflight missing')
    for name,digest in draft['code_hashes'].items():
        if hashlib.sha256((DRAFT_ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Observer source changed '+name)
    root=Path(HOST_RUN);root.mkdir(exist_ok=False);dst=root/'runtime';dst.mkdir()
    for n in ('scripts','config','cache','home'):
        src=Path(ROOT)/'runtime'/n
        if src.exists():shutil.copytree(src,dst/n,symlinks=True)
        else:(dst/n).mkdir()
    for n in ('tmp','logs','official-traces','observer','instrumentation'):(dst/n).mkdir()
    for n in ('sitecustomize.py','wio_moe_observer.py'):shutil.copy2(DRAFT_ROOT/n,dst/'instrumentation'/n)
    client_source=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official_profile_client.py')
    if hashlib.sha256(client_source.read_bytes()).hexdigest()!='42a6d9e54ea0a922843e8ab37f4096e52c54fb4dbcdd23b77bb60d69293f3131':raise RuntimeError('Frozen client source mismatch')
    shutil.copy2(client_source,dst/'official_profile_client.py')
    if hashlib.sha256((dst/'scripts/serve.sh').read_bytes()).hexdigest()!=SERVE_SHA:raise RuntimeError('Copied serve changed')
    # Original serve.sh remains byte-identical. Only the isolated process gets
    # the diagnostic import path, output path and private TMPDIR bind.
    script='#!/usr/bin/env bash\nset -euo pipefail\nexport PYTHONPATH='+shlex.quote(APP+'/instrumentation')+':"${PYTHONPATH:-}"\nexport WIO_OBSERVER_ROOT='+shlex.quote(APP+'/observer')+'\nexec bash '+shlex.quote(APP+'/scripts/serve.sh')+'\n'
    (dst/'scripts/serve-observer.sh').write_text(script)
    cfg=d['HostConfig'];args=['docker','create','--name',DIAG,'--label','wio.task='+RUN,'--runtime',cfg['Runtime'],'--privileged','--read-only','--network','bridge','--ipc','private','--shm-size',str(cfg['ShmSize']),'--workdir',APP,'--entrypoint','bash']
    for env in d['Config']['Env']:args+=['--env',env]
    for u in cfg.get('Ulimits') or []:args+=['--ulimit',f"{u['Name']}={u['Soft']}:{u['Hard']}"]
    for cap in cfg.get('CapAdd') or []:args+=['--cap-add',cap]
    for opt in cfg.get('SecurityOpt') or []:args+=['--security-opt',opt]
    for m in d['Mounts']:
        if m['Type']!='bind':raise RuntimeError('Unexpected non-bind mount')
        if m['Destination']=='/tmp':continue
        src=str(dst) if m['Destination']==APP else m['Source']
        if m['RW'] and m['Destination']!=APP:raise RuntimeError('Unexpected writable bind')
        args+=['--mount','type=bind,src='+src+',dst='+m['Destination']+(',readonly' if not m['RW'] else '')]
    args+=['--mount','type=bind,src='+str(dst/'tmp')+',dst=/tmp','--mount','type=bind,src='+ROOT+'/runtime/benchmark-20261028,dst=/bench-inputs,readonly']
    for dev in cfg.get('Devices') or []:args+=['--device',dev['PathOnHost']+':'+dev['PathInContainer']+':'+dev['CgroupPermissions']]
    args += [IMAGE,'-c','exec bash '+APP+'/scripts/serve-observer.sh >'+APP+'/logs/service.log 2>&1']
    diag_id=cmd(args,timeout=90).stdout.strip();created=inspect(DIAG)
    if created['Id']!=diag_id or created['Image']!=IMAGE:raise RuntimeError('Diagnostic identity failed')
    json_write(root/'diagnostic-identity.json',{'name':DIAG,'id':diag_id,'label':RUN,'baseline_id':BASE_ID,'baseline_prestate':d['State'],'image':IMAGE})
    json_write(root/'docker-create-argv.json',args)
    json_write(root/'preflight.json',{'guard':guard,'reference':reference,'observer_plan':draft,'source_serve_sha':SERVE_SHA,'differences':['isolated mounted runtime/cache/home/tmp/logs','same-call observer adds ordinal update and device-to-host snapshots; timing perturbed','no profiler; no optimization or returned tensor mutation'],'budget_observer_bytes':OBSERVER_BUDGET,'max_eager_samples_per_layer_shape':3,'all_module_coverage_remains_open':True})
    return diag_id

def observer_size():
    return sum(p.stat().st_size for p in (Path(HOST_RUN)/'runtime/observer').rglob('*') if p.is_file())

def client(phase):
    free_guard();proc=subprocess.Popen(['docker','exec','-e','WIO_PHASE='+phase,'-e','WIO_OUTPUT='+APP+'/client-'+phase,DIAG,'timeout','--kill-after=10s','600s','python3',APP+'/official_profile_client.py'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    deadline=time.monotonic()+625
    try:
        while proc.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('Finite observer client timeout')
            free_guard()
            if observer_size()>OBSERVER_BUDGET:raise RuntimeError('Observer disk budget exceeded')
            time.sleep(2)
        out,err=proc.communicate(timeout=10);stage_write('client-'+phase+'-process.json',{'exit_code':proc.returncode,'stdout':out,'stderr':err})
        if proc.returncode:raise RuntimeError(phase+' client failed '+err[-1500:])
        result=json.loads((Path(HOST_RUN)/'runtime'/('client-'+phase)/'window-status.json').read_text())
        if not result['valid']:raise RuntimeError('Frozen workload counter validation failed')
    finally:
        if proc.poll() is None:proc.terminate()

def arm_after_warmup():
    health(DIAG,True);root=Path(HOST_RUN)/'runtime/observer';time.sleep(2);files=sorted(root.rglob('*.pt'))
    if not files:raise RuntimeError('No warmup/captured tensor files; observer path not executed')
    if observer_size()>OBSERVER_BUDGET:raise RuntimeError('Observer warmup budget exceeded')
    with tarfile.open(Path(HOST_RUN)/'observer-warmup-snapshots.tar.gz','w:gz',compresslevel=1) as tf:
        for p in files:tf.add(p,arcname=str(p.relative_to(root)),recursive=False)
    # Captured file paths must remain valid. Preserve directories and metadata,
    # remove only old tensor snapshots before actual measured-phase requests.
    for p in files:p.unlink()
    json_write(root/'ARMED.json',{'phase':'fixed-measure1','armed_epoch_ns':time.time_ns(),'warmup_tensor_files_archived_and_removed':len(files),'capture_metadata_retained':True})
    stage_write('arm-record.json',json.loads((root/'ARMED.json').read_text()))

def inventory():
    root=Path(HOST_RUN)/'runtime/observer';previous=None;stable=0
    for _ in range(10):
        files=sorted(root.rglob('*.pt'));marker=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
        stable=stable+1 if marker==previous else 0;previous=marker
        if stable>=2:break
        time.sleep(2)
    if not files or stable<2:raise RuntimeError('Post-case snapshots missing/not stable')
    rows=[];rankdirs=set();layers=set()
    for p in files:
        stem=re.sub(r'_device_\d+_\d+$','',p.stem);meta=p.with_name(stem+'.json')
        if not meta.is_file():raise RuntimeError('Tensor has no binding metadata '+str(p))
        m=json.loads(meta.read_text());rankdirs.add(m['ep_rank']);layers.add((m['role'],m['layer_name']))
        rows.append({'tensor':str(p),'tensor_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'metadata':str(meta),'metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest(),'ep_rank':m['ep_rank'],'role':m['role'],'layer_name':m['layer_name'],'capturing':m['capturing'],'M':m['M'],'K':m['K'],'E_local':m['E_local']})
    json_write(Path(HOST_RUN)/'tensor-index.json',rows)
    reference=json.loads((Path(ROOT+REL)/'reference-latest.json').read_text())
    refstate=json.loads((Path(reference['root'])/'state.json').read_text())
    refclient=Path(refstate['result']['client_results']);diagclient=Path(HOST_RUN)/'runtime/client-baseline'
    a=json.loads((refclient/'responses.json').read_text());b=json.loads((diagclient/'responses.json').read_text());ra=json.loads((refclient/'window-status.json').read_text());rb=json.loads((diagclient/'window-status.json').read_text())
    if ra['input_sha256']!=rb['input_sha256']:raise RuntimeError('Reference/observer prompts differ')
    text_equal=[x['text']==y['text'] for x,y in zip(a,b)]
    json_write(Path(HOST_RUN)/'reference-output-comparison.json',{'identical_prompt_hash':ra['input_sha256'],'per_request_text_equal':text_equal,'all_text_equal':len(text_equal)==8 and all(text_equal),'scope':'visible text equality for this one round; scheduling-dependent numeric differences require review if unequal; no global correctness proof'})
    result={'status':'COLLECTED_VALUES_NOT_YET_SEMANTICALLY_VALIDATED','tensor_files':len(rows),'ep_ranks':sorted(rankdirs),'rank_count':len(rankdirs),'layer_surfaces':[list(x) for x in sorted(layers)],'observer_bytes':observer_size(),'all_case_counters_valid':True,'needs_next_review':['read CPU tensor values and observer ordinals','match all16 ranks by actual group/layer/ordinal, validate routing-count conservation','bind actual custom/CANN branch, capacity, mask and expert placement','capture-path overwrite and sampling gaps; not complete whole-module coverage'],'model_speedup_claimed':False,'reference_visible_text_equal':len(text_equal)==8 and all(text_equal),'overall_goal_complete':False}
    if len(rankdirs)!=16:raise RuntimeError('Missing EP ranks '+repr(sorted(rankdirs)))
    json_write(Path(HOST_RUN)/'capture-result.json',result)

def restore(diag_id):
    stop_diag(diag_id);d=baseline_check()
    if not d['State']['Running']:
        try:release=wait_released(timeout=120)
        except Exception as e:status('RESTORE_DEFERRED_DEVICE_NOT_RELEASED',error=repr(e));return False
        json_write(Path(HOST_RUN)/'release-before-restore.json',release)
        cmd(['docker','start',BASE_ID],timeout=120)
    result=wait_health(BASE,timeout=2400);guard=owner_guard(BASE)
    if serve_hash(BASE)!=SERVE_SHA:raise RuntimeError('Restored serve SHA mismatch')
    json_write(Path(HOST_RUN)/'restore-result.json',{'status':'PASS_ORIGINAL_BASELINE_RESTORED','id':BASE_ID,'image':IMAGE,'health':result,'all_npu_pids_owned':True,'npu_pids':guard['npu_pids'],'prefix_cache_cold_reseed_required':True,'source_code_or_serve_modified':False})
    status('RESTORED',health=result,ownership=guard,serve_sha=SERVE_SHA,baseline_id=BASE_ID,image=IMAGE,note='Original baseline restored; prefix cache cold, re-seed for next comparison')
    return True

def main():
    lock_path=Path(ROOT+REL)/'.device-diagnostic.lock';lock=open(lock_path,'a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    diag_id=prepare();parent=os.getpid();raw=Path('/proc',str(parent),'stat').read_text();parent_start=raw[raw.rfind(')')+2:].split()[19];child=os.fork()
    if child==0:watchdog(parent,parent_start,diag_id)
    def requested_stop(signum,frame):raise RuntimeError('Finite diagnostic interrupted/ownership monitor stop: '+str(signum))
    signal.signal(signal.SIGUSR1,requested_stop);signal.signal(signal.SIGTERM,requested_stop);signal.signal(signal.SIGINT,requested_stop)
    samples=[];stop=threading.Event();sampler=threading.Thread(target=sample_loop,args=(stop,samples),daemon=True);failed=None
    try:
        status('PREPARED',diag_id=diag_id,watchdog_pid=child,observer_only=True)
        baseline_check();health(BASE,True);owner_guard(BASE)
        cmd(['docker','stop','--time','45',BASE_ID],timeout=100)
        json_write(Path(HOST_RUN)/'release-before-diagnostic.json',wait_released(timeout=120))
        cmd(['docker','start',diag_id],timeout=120);sampler.start();status('DIAGNOSTIC_LOADING',diag_id=diag_id,watchdog_pid=child)
        wait_health(DIAG,timeout=2400);owner_guard(DIAG);stage_write('runtime-versions.json',runtime_versions())
        status('WARMUP_FIXED_CASE_RUNNING');client('warmup');arm_after_warmup()
        status('OBSERVED_FIXED_CASE_RUNNING');client('baseline');health(DIAG,True);inventory()
        status('VALUES_COLLECTED_PENDING_SEMANTIC_REVIEW')
    except BaseException as e:
        failed=repr(e);stage_write('controller-error.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
    finally:
        stop.set()
        if sampler.is_alive():sampler.join(timeout=40)
        stage_write('npu-smi-samples.json',samples)
        restored=restore(diag_id)
        if not restored:raise RuntimeError('Original baseline restore deferred; see explicit state')
        json_write(Path(HOST_RUN)/'controller-finished.json',{'state':'RESTORED_AFTER_ERROR' if failed else 'CAPTURED_AND_RESTORED','capture_error':failed,'watchdog_pid':child,'baseline_id':BASE_ID,'overall_goal_complete':False})
        # Child keeps the inherited flock until this parent exits and it verifies
        # restoration. That prevents another owner operation from racing recovery.
        lock.close()

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        try:stage_write('top-level-error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        except BaseException:pass
        raise
