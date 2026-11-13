"""从实测 Docker 参数重建启动命令。默认 dry run；执行需指定共享锁目录。"""
from pathlib import Path
import argparse,fcntl,json,os,re,signal,socket,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
BRANCH='sglang-dsv4.1flash1017-910c'
IID='sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517'
PORTS=[18167,18280,14699,*range(18400,18407)]
def command(model,name,generation='g42'):
    argv=json.loads((ROOT/(f'deployment/{generation}/docker_command.json' if generation!='g30' else 'deployment/docker_command.json')).read_text())
    for i,v in enumerate(argv):
        if i and argv[i-1]=='--name':argv[i]=name
        elif i and argv[i-1]=='--label':argv[i]='archive.branch='+BRANCH
        elif i and argv[i-1]=='-v':
            host,rest=v.split(':',1)
            if host.startswith('/data/sglang-dsv41'):host=str(ROOT)+host[len('/data/sglang-dsv41'):]
            elif host.startswith('/data/models/'):host=str(model)
            argv[i]=host+':'+rest
    return argv
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--name',default='sglang-dsv41-910c');p.add_argument('--execute',action='store_true')
    p.add_argument('--generation',choices=['g30','g37','g42','g43'],default='g42')
    p.add_argument('--device-lock-dir',type=Path);p.add_argument('--port-lock-dir',type=Path)
    p.add_argument('--legacy-device-lock-dir',type=Path);p.add_argument('--pool-policy',type=Path)
    p.add_argument('--lifetime-seconds',type=int,default=21600)
    a=p.parse_args();argv=command(a.model_path.absolute(),a.name,a.generation)
    if not a.execute:print(json.dumps(argv,ensure_ascii=False,indent=2));return
    assert a.device_lock_dir and a.port_lock_dir,'Use fleet-wide existing shared device/port lock directories'
    assert 0<a.lifetime_seconds<=21600
    locks=[];groups=[(a.device_lock_dir,range(16),'{}'),(a.port_lock_dir,PORTS,'sglang-port-{}.lock')]
    def policy():
        if a.generation!='g30':
            assert a.pool_policy and a.pool_policy.is_file(),'Specify the existing authoritative pool policy'
            assert not json.loads(a.pool_policy.read_text()).get('leases'),'Existing leases; do not take over'
    if a.generation!='g30':
        assert a.legacy_device_lock_dir,'Also hold the existing legacy device locks'
        groups.append((a.legacy_device_lock_dir,range(16),'device-{}.lock'))
    policy()
    for root,items,pattern in groups:
        assert root.is_dir(),'Coordination directory must already exist'
        for i in items:
            f=(root/pattern.format(i)).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    policy()
    d=json.loads(subprocess.check_output(['docker','image','inspect',IID]))[0];assert d['Id']==IID and d['Architecture']=='arm64'
    assert a.model_path.is_dir() and (a.model_path/'optional/quarot.safetensors').is_file()
    desc=json.loads((a.model_path/'quant_model_description.json').read_text())
    qkv={k:v for k,v in desc.items() if k.startswith('layers.') and (k.endswith('.attn.wq_a.weight') or k.endswith('.attn.wkv.weight'))}
    assert len(qkv)==80 and set(qkv.values())=={'FLOAT'},'Native QKV fusion requires all 80 target projections FLOAT'
    vendor=ROOT/'compat/vendor_strict_host_g2';vm=json.loads((vendor/'strict_host_manifest.json').read_text())
    assert vm['payload_count']==20 and vm['preserved_payloads_exact']==17
    import hashlib
    if a.generation!='g30':
        provenance=[json.loads((ROOT/f'evidence/{g}/publication-provenance.json').read_text()) for g in (['g37'] if a.generation=='g37' else ['g37','g43'])]
        for item in [item for doc in provenance for item in doc['files']]:
            if item['path'].startswith('compat/runtime/'):
                assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['published_sha256'],item['path']
    for item in vm['payloads']:assert hashlib.sha256((vendor/item['path']).read_bytes()).hexdigest()==item['sha256']
    for key in ['compat/candidate_build_g8/lib','compat/rope_build_core5_g1/lib','compat/router_build_current_g2/lib']:
        assert len(list((ROOT/key).glob('*.so')))==1,'Build all three dependencies first'
    n=subprocess.check_output(['npu-smi','info'],text=True)
    assert all(f'No running processes found in NPU {i}' in n for i in range(8)),'Existing NPU workload; do not launch'
    hbm=[]
    for line in n.splitlines():
        fields=[x.strip() for x in line.split('|')[1:-1]]
        if len(fields)==3 and re.fullmatch(r'\d+\s+\d+',fields[0]) and ':' in fields[1]:
            vals=re.findall(r'(\d+)\s*/\s*(\d+)',fields[2])
            if vals:hbm.append(tuple(map(int,vals[-1])))
    assert len(hbm)==16 and all(total-used>=53*1024 for used,total in hbm),'Need at least 53GiB free HBM on every die'
    for port in PORTS:
        with socket.socket() as s:s.bind(('127.0.0.1',port))
    for i,v in enumerate(argv):
        if i and argv[i-1]=='-v' and v.endswith(':ro'):
            assert Path(v.split(':',1)[0]).exists(),'Missing bind source: '+v
    for path in ['serve_g30/home','serve_g30/tmp','serve_g30/logs','serve_g30/cache','serve_ascops37/home','serve_ascops37/tmp','serve_ascops37/logs','serve_ascops37/cache','compat/probe_g5/home/.cache/torch_extensions/py312_cpu']:(ROOT/path).mkdir(parents=True,exist_ok=True)
    for i,v in enumerate(argv):
        if i and argv[i-1]=='-e' and '=' in v:
            key,value=v.split('=',1)
            if key in {'HOME','TMPDIR','ASCEND_PROCESS_LOG_PATH','ASCEND_WORK_PATH'} and value.startswith('/task/'):
                (ROOT/value.removeprefix('/task/')).mkdir(parents=True,exist_ok=True)
    state=ROOT/'.work';state.mkdir(exist_ok=True)
    stop=False
    def on_signal(*_):nonlocal stop;stop=True
    signal.signal(signal.SIGINT,on_signal);signal.signal(signal.SIGTERM,on_signal)
    with (state/'service.log').open('x') as log:
        process=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            start=time.monotonic()
            while process.poll() is None and not stop and time.monotonic()-start<a.lifetime_seconds:time.sleep(2)
        finally:
            if process.poll() is None:
                d=json.loads(subprocess.check_output(['docker','inspect',a.name]))[0]
                assert d['Config']['Labels']['archive.branch']==BRANCH
                subprocess.run(['docker','stop','--time','30',a.name],check=True,timeout=60)
                process.wait(timeout=60)
        if not stop and time.monotonic()-start<a.lifetime_seconds and process.returncode:raise SystemExit(process.returncode)
if __name__=='__main__':main()
