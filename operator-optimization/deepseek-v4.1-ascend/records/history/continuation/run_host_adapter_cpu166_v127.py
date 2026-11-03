"""Compile/test the adapter against a CPU fake runtime in the existing container."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,secrets,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,read,write

INNER=r'''
from pathlib import Path
import hashlib,json,os,platform,re,subprocess,time
assert platform.system()=='Linux' and platform.machine()=='aarch64'
base=Path('/tmp')/('wio-host-adapter-'+RUN_ID);base.mkdir(mode=0o700,exist_ok=False)
for name,source in SOURCES.items():
    assert name in ALLOWED and Path(name).name==name
    (base/name).write_text(source)
header=Path('/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h')
library=Path('/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so')
assert hashlib.sha256(header.read_bytes()).hexdigest()=='62d127fbb899b1476e054b1815ca6e0c71a362868c655f0e3da4616fb7598fb7'
assert hashlib.sha256(library.read_bytes()).hexdigest()=='1c06b9c4a8910cfb585eab4ed0066bc09e28a0cf273dfcb15458f7cc49a78cea'
sym=subprocess.run(['readelf','--dyn-syms','--wide',str(library)],capture_output=True,text=True,check=True,timeout=10).stdout
wanted=['aclrtLaunchKernelWithHostArgs','aclrtGetFunctionName','aclrtGetThreadLastTaskId','aclrtStreamGetId']
symbols={name:[line.strip()for line in sym.splitlines()if line.split()and line.split()[-1].split('@')[0]==name and ' UND 'not in line]for name in wanted}
cc=['/usr/bin/cc','-std=c11','-Wall','-Wextra','-Werror','-O2','-fPIC','-I',str(base)]
commands=[cc+['-shared',str(base/'fake_runtime.c'),'-Wl,-soname,libfake_runtime.so','-ldl','-o',str(base/'libfake_runtime.so')],
    cc+['-shared',str(base/'host_packet_snapshot.c'),str(base/'host_args_adapter.c'),'-ldl','-pthread','-o',str(base/'libhost_adapter.so')],
    cc+[str(base/'fake_driver.c'),'-L',str(base),'-lfake_runtime','-Wl,-rpath,$ORIGIN','-ldl','-pthread','-o',str(base/'fake_driver')],
    cc+['-shared',str(base/'fake_driver.c'),'-L',str(base),'-lfake_runtime','-Wl,-rpath,$ORIGIN','-ldl','-pthread','-o',str(base/'liblate_plugin.so')],
    cc+[str(base/'late_probe.c'),'-ldl','-o',str(base/'late_probe')],
    cc+['-shared','-fsanitize=address,undefined','-fno-omit-frame-pointer',str(base/'host_packet_snapshot.c'),str(base/'host_args_adapter.c'),'-ldl','-pthread','-o',str(base/'libhost_adapter_sanitized.so')]]
builds=[]
for command in commands:
    p=subprocess.run(command,capture_output=True,text=True,timeout=30)
    builds.append(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    if p.returncode:
        print(json.dumps(dict(status='CPU_BUILD_ERROR',base=str(base),builds=builds,new_device_work=False)));raise SystemExit(1)
asan=subprocess.run(['/usr/bin/cc','-print-file-name=libasan.so'],capture_output=True,text=True,check=True).stdout.strip()
assert Path(asan).is_file()
needed={name:subprocess.run(['readelf','-d',str(base/name)],capture_output=True,text=True,check=True,timeout=10).stdout for name in ['libhost_adapter.so','libfake_runtime.so','fake_driver','late_probe']}
assert all('libascendcl'not in s and 'libtorch'not in s for s in needed.values())
specs=[('disabled','one',{'WIO_HOST_TRACE_ENABLE':'0'}),('normal','one',{}),('no_ids','one',{'WIO_HOST_TRACE_QUERY_IDS':'0'}),
    ('failure','failure',{}),('non_target','non_target',{}),('name_failure','name_failure',{}),('oversize','oversize',{}),
    ('recursive','recursive',{}),('many','many',{'WIO_HOST_TRACE_MAX_RECORDS':'7'}),
    ('byte_budget','one',{'WIO_HOST_TRACE_MAX_BYTES':'1024'}),('zero_quota','one',{'WIO_HOST_TRACE_MAX_RECORDS':'0'}),
    ('unarmed','one',{'WIO_HOST_TRACE_ARM_FILE':str(base/'not-armed')}),('expired','expired',{'WIO_HOST_TRACE_MAX_SECONDS':'1'}),
    ('early_exit','early_exit',{}),('bad_dir','one',{'WIO_HOST_TRACE_DIR':str(base/'missing'/'directory')}),('fork','fork',{}),
    ('late_local','plugin',{}),('direct_lookup_bypass','direct',{}),('missing_backend','missing',{})]
specs += [('asan_'+mode,mode,{'SANITIZED':'1'})for mode in ['one','many','recursive','oversize']]
cases=[]
for name,mode,extra in specs:
    trace=base/name;trace.mkdir(mode=0o700)
    env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(base),LD_PRELOAD=str(base/'libhost_adapter.so'),WIO_HOST_TRACE_ENABLE='1',
        WIO_HOST_TRACE_DIR=str(trace),WIO_HOST_TRACE_RUN_ID=name,WIO_HOST_TRACE_QUERY_IDS='1',WIO_HOST_TRACE_REAL_LIBRARY=str(base/'libfake_runtime.so'))
    extra=dict(extra)
    if extra.pop('SANITIZED',None):
        env['LD_PRELOAD']=asan+':'+str(base/'libhost_adapter_sanitized.so');env['ASAN_OPTIONS']='detect_leaks=0:abort_on_error=1';env['UBSAN_OPTIONS']='halt_on_error=1'
    env.update(extra)
    if mode in ['plugin','direct','missing']:
        target=str(base/'liblate_plugin.so')if mode=='plugin'else str(base/'libfake_runtime.so')
        command=[str(base/'late_probe'),mode,target]
    else:command=[str(base/'fake_driver'),mode]
    p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=15)
    files=list(trace.glob('*.jsonl'));assert len(files)<=1
    records=[];file_index=[]
    for path in files:
        blob=path.read_bytes();assert len(blob)<=int(env.get('WIO_HOST_TRACE_MAX_BYTES','1048576'))
        records.extend(json.loads(line)for line in blob.splitlines())
        file_index.append(dict(path=str(path),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    parsed=None
    if p.stdout.strip():parsed=json.loads(p.stdout)
    cases.append(dict(name=name,mode=mode,returncode=p.returncode,stdout=parsed,stderr=p.stderr,records=records,files=file_index))
    if p.returncode:break
binaries=[]
for name in ['libhost_adapter.so','libhost_adapter_sanitized.so','libfake_runtime.so','fake_driver','late_probe','liblate_plugin.so']:
    path=base/name;blob=path.read_bytes();binaries.append(dict(path=str(path),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
print(json.dumps(dict(status='CPU_ADAPTER_TESTS_FINISHED',base=str(base),platform=dict(system=platform.system(),machine=platform.machine()),
    sources=[dict(name=name,bytes=len(source.encode()),sha256=hashlib.sha256(source.encode()).hexdigest())for name,source in SOURCES.items()],
    builds=builds,installed_library_symbols=symbols,installed_CANN_library_read_only=True,binary_dynamic_tables=needed,
    binaries=binaries,cases=cases,CANN_runtime_loaded_in_tests=False,Torch_or_NPU_context_created=False,
    real_CANN_interception_or_original_operand_recovery=False,original_container_or_service_changed=False,new_device_work=False)))
'''


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',choices=['b','c','d'],required=True);args=ap.parse_args()
    suffix='v127'+args.attempt
    receipt=N/('host-adapter-CPU166-'+suffix+'.json');assert not receipt.exists()
    allowed=['host_packet_snapshot.h','host_packet_snapshot.c','host_args_adapter.c','fake_runtime.c','fake_driver.c','late_probe.c']
    sources={name:(N/'eager-host-adapter-v127'/name).read_text()for name in allowed}
    run_id='cpu-'+suffix+'-'+secrets.token_hex(4)
    inner='RUN_ID='+repr(run_id)+'\nALLOWED='+repr(allowed)+'\nSOURCES='+repr(sources)+'\n'+INNER
    compile(inner,'<CPU-only-adapter-tests>','exec')
    host="""import socket,os,subprocess,json
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
cid='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'
before=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=10).stdout)[0]
assert before['Id']==cid and before['State']['Running'] and before['State']['Pid']==2032500
p=subprocess.run(['docker','exec',cid,'python3','-I','-S','-c',INNER_CODE],capture_output=True,text=True,timeout=120)
after=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=10).stdout)[0]
assert after['Id']==cid and after['State']['Running'] and after['State']['Pid']==before['State']['Pid']
print(json.dumps(dict(hostname=socket.gethostname(),container_id=cid,container_pid=after['State']['Pid'],returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
""".replace('INNER_CODE',repr(inner))
    control=N/'ssh166-adapter-v127.sock';assert control.exists()
    argv=['/usr/bin/ssh','-F','/dev/null','-S',str(control),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes',
        '-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(host)]
    p=subprocess.run(argv,capture_output=True,text=True,timeout=140)
    envelope=None
    for line in p.stdout.splitlines():
        try:
            r=json.loads(line)
            if r.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':envelope=r
        except (ValueError,AttributeError):pass
    record=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,envelope=envelope,stderr=p.stderr,
        source_payload_sha256=hashlib.sha256(inner.encode()).hexdigest(),new_device_work=False)
    write(receipt,record)
    assert envelope is not None and p.returncode==0,(p.returncode,p.stderr)
    result=json.loads(envelope['stdout']);write(O/('host-adapter-CPU-results-'+suffix+'.json'),result)
    assert envelope['returncode']==0,(envelope['returncode'],result)
    print(json.dumps(dict(status=result['status'],platform=result.get('platform'),base=result.get('base'),
        cases=len(result.get('cases',[])),failed_cases=[r['name']for r in result.get('cases',[])if r['returncode']],
        installed_symbols={k:bool(v)for k,v in result.get('installed_library_symbols',{}).items()},new_device_work=False)))


if __name__=='__main__':main()
