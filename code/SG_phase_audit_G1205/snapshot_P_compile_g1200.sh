python3 - <<'PYREMOTE'
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import urllib.request

assert socket.gethostname()=='<ROLE_HOSTNAME>' and os.geteuid()==0
root=Path('/PRIVATE_TASK_ROOT')
run=root/'sg_pd_g1178'
handle=json.loads((run/'actual_launch_handle.json').read_text())
state=json.loads((run/'state.json').read_text())
assert state['pid']==handle['pid']==2240397 and state['generation']==handle['epoch']==1178
p=Path('/proc',str(state['pid']))
assert str(run/'node_controller.py').encode() in (p/'cmdline').read_bytes().split(b'\0')
guard=Path('/proc','2227468')
assert str(root/'workload_guard_g1120.py').encode() in (guard/'cmdline').read_bytes().split(b'\0')
assert any(fd.resolve()==Path('/PRIVATE_COORD_ROOT/deepseek-v41-p3d4-workload.lock') for fd in (guard/'fd').iterdir())
info=json.loads(subprocess.check_output(['docker','inspect',state['container_id']],timeout=10))[0]
assert info['Image']==handle['image'] and info['State']['Running']
assert info['Config']['Labels']['codex.controller_pid']==str(state['pid'])
assert info['Config']['Labels']['codex.run']==run.name
url='http://127.0.0.1:19093'
def get(path):
    with urllib.request.urlopen(url+path,timeout=6) as response:
        assert response.status==200
        return response.read()
get('/ready')
metrics=get('/metrics').decode()
idle={}
for key in ('num_running_reqs','num_queue_reqs'):
    v=[float(x) for x in re.findall(r'^sglang:'+key+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)',metrics,re.M)]
    assert v and sum(v)==0,(key,v)
    idle[key]=v
args=json.loads(get('/server_info'))
a=args.get('server_args',args)
assert a['disaggregation_mode']=='prefill' and a['tp_size']==a['ep_size']==8
assert a['speculative_algorithm']=='DSPARK' and a['speculative_num_draft_tokens']==6
assert a['chunked_prefill_size']==8192 and not a['disable_radix_cache']
D_COMPILE_TARGET={'host_run_root':str(run),'container_name':state['container'],'container_id':state['container_id']}

def compile_cache():
    assert D_COMPILE_TARGET is not None, 'Explicit actual decode container/mount required'
    base=Path(D_COMPILE_TARGET['host_run_root'])
    info=json.loads(subprocess.check_output(['docker','inspect',D_COMPILE_TARGET['container_name']]))[0]
    assert info['Id']==D_COMPILE_TARGET['container_id'] and info['State']['Running']
    assert info['Config']['Labels']['codex.run']==base.name
    mounts=sorted(info['Mounts'],key=lambda m:len(m['Destination']),reverse=True)
    assert any(m['Source']==str(base) and m['Destination']=='/task/'+base.name and m['RW'] for m in mounts)
    def host_path(value):
        for m in mounts:
            dest=m['Destination'].rstrip('/')
            if value==dest or value.startswith(dest+'/'):
                assert m['RW'],'Compiler root maps to readonly mount'
                p=Path(m['Source'])/value.removeprefix(dest).lstrip('/')
                assert p.resolve().is_relative_to(base.resolve()),('Compiler root escapes actual run',str(p))
                return p
        raise AssertionError(('No host mapping for compiler root',value))
    keys=['HOME','TMPDIR','TRITON_CACHE_DIR','ASCEND_CACHE_PATH','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR','TORCHAIR_CACHE_HOME','VLLM_CACHE_ROOT','XDG_CACHE_HOME','SGLANG_CACHE_DIR','SGLANG_JIT_CACHE_DIR','SGLANG_DG_CACHE_DIR','SGLANG_CUTE_AOT_CACHE_DIR','TILELANG_CACHE_DIR','HUMMING_CACHE_DIR']
    envs=[dict(x.split('=',1) for x in info['Config']['Env'] if '=' in x)]
    # Native launch adds compiler variables after docker creation. Inspect actual live container processes as well.
    pids=subprocess.check_output(['docker','top',info['Id'],'-eo','pid'],text=True).splitlines()[1:]
    for line in pids:
        pid=line.strip()
        if not pid.isdigit():continue
        path=Path('/proc')/pid/'environ'
        try:raw=path.read_bytes()
        except FileNotFoundError:continue
        envs.append({k:v for item in raw.decode(errors='replace').split('\0') if '=' in item for k,v in [item.split('=',1)] if k in keys})
    assert len(envs)>1,'Actual live decode process environment unavailable'
    roots={base/s for s in ['cache/triton','cache/ascend','cache/torch_extensions','cache/inductor','cache/torchair','home/.cache/vllm','home/.cache/torch','home/.cache/sglang','tmp/torchinductor_root']}
    for env in envs:
        for key in keys[2:]:
            if key in env and env[key]:roots.add(host_path(env[key]))
        if env.get('HOME'):
            roots.add(host_path(env['HOME']+'/.cache/vllm'));roots.add(host_path(env['HOME']+'/.cache/torch'));roots.add(host_path(env['HOME']+'/.cache/sglang'))
        if env.get('TMPDIR'):roots.add(host_path(env['TMPDIR']+'/torchinductor_root'))
    files={q for r in roots for q in r.rglob('*') if q.is_file() and not q.name.endswith('.lock')}
    return {str(q.relative_to(base)):{'bytes':q.stat().st_size,'mtime_ns':q.stat().st_mtime_ns} for q in sorted(files)}

files=compile_cache()
keys={'ASCEND_CACHE_PATH','ASCEND_WORK_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','SGLANG_CACHE_DIR','PYTHONPYCACHEPREFIX'}
env_sets={key:set() for key in keys}
procs=subprocess.check_output(['docker','top',info['Id'],'-eo','pid'],text=True).splitlines()[1:]
for row in procs:
    path=Path('/proc',row.strip(),'environ')
    try:
        pairs=path.read_bytes().split(b'\0')
    except FileNotFoundError:
        continue
    for raw in pairs:
        k,eq,v=raw.partition(b'=')
        key=k.decode(errors='replace')
        if eq and key in keys:
            env_sets[key].add(v.decode(errors='replace'))
mounts=sorted(info['Mounts'],key=lambda m:len(m['Destination']),reverse=True)
def resolve(value):
    for m in mounts:
        dest=Path(m['Destination'])
        val=Path(value)
        if val==dest or dest in val.parents:
            assert m['RW'],value
            return Path(m['Source'])/val.relative_to(dest)
    raise AssertionError(('unmapped selected variable',value))
extra={}
for value in sorted(env_sets['ASCEND_WORK_PATH']):
    base=resolve(value)
    assert base.resolve().is_relative_to(run.resolve())
    extra[str(base.relative_to(run))]={str(f.relative_to(run)):{'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns}
        for f in sorted(base.rglob('*')) if f.is_file() and not f.name.endswith('.lock')}
selector={'engine_type':'prefill','mode':'device_hit','model_name':'ds41flash','moe_ep_rank':'0','pp_rank':'0','tp_rank':'0'}
selected=[]
for line in metrics.splitlines():
    if not line.startswith('sglang:prefill_effective_tokens_total{'):
        continue
    left, val=line.rsplit(' ',1)
    if all(f'{k}="{v}"' in left for k,v in selector.items()):
        selected.append(float(val))
assert len(selected)==1,selected
assert shutil.disk_usage('/data').free>=shutil.disk_usage('/data1').free
out=root/'P_compile_snapshots_g1200'
out.mkdir(exist_ok=True)
summary={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'P_controller_PID':state['pid'],'P_container_ID':info['Id'],
        'image_ID':info['Image'],'idle':idle,'guard_FD_verified':True,'ready':200,
        'compiler_file_count':len(files),'separate_Ascend_work_file_counts':{k:len(v) for k,v in extra.items()},
        'selected_env':{k:sorted(v) for k,v in env_sets.items()},'prefill_device_hit_selector':selector,
        'prefill_device_hit_value':selected[0],'source_function_SHA256':'8596c62f1f63a491728c077954832137ac959148ea885e371fae8a25aa326cbe',
        'scope':'Actual idle P baseline before the G1178 PD workload. No inference submission. Repeat at the phase boundaries to prove stability.'}
report={'summary':summary,'compiler_files':files,'separate_Ascend_work_files':extra}
path=out/'before_G1178_PD_first_workload.json'
with path.open('x') as f:
    json.dump(report,f,indent=2)
    f.write('\n')
summary['raw_receipt_SHA256']=hashlib.sha256(path.read_bytes()).hexdigest()
summary['raw_snapshot']=str(path)
print('G1200_P_COMPILE_AND_CACHE_BASELINE_JSON '+json.dumps(summary))
PYREMOTE
