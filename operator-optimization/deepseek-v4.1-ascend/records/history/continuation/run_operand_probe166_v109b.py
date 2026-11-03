"""One explicitly authorized,model-free diagnostic in the existing running container."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    receipt=N/'operand-probe166-v109b.json'
    assert not receipt.exists()and not(N/'operand-probe-submission-v109b.json').exists(),'never repeat a submitted probe'
    script=N/'diagnostic-operand-probe-v109/inspect_graph_params.py'
    original=read(N/'operand-probe166-v109.json');assert original['result']['attempt']['returncode']==1 and 'Read-only file system' in original['result']['attempt']['stderr'];assert original['result']['elapsed_seconds']<1
    expected='c42067e616ae46f6fc6784b3eac1c749559847596a273280a2d34eeb4e82889f';b=script.read_bytes()
    assert hashlib.sha256(b).hexdigest()==expected
    now=datetime.now(timezone.utc);out='/tmp/wio-operand-probe-v109-'+now.strftime('%Y%m%dT%H%M%SZ')
    # The fixed payload has already been reviewed;no dynamic model source is staged.
    code='''import os,socket,subprocess,json,hashlib,base64,tarfile,io,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
cid='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'
out=OUTPUT;payload=base64.b64decode(PAYLOAD);assert hashlib.sha256(payload).hexdigest()==EXPECTED
r=dict(hostname=socket.gethostname(),container=cid,output_dir=out,device=0,status='GUARD_ONLY',new_model_load=False,graph_replays=0,source_sha256=EXPECTED)
def cmd(args,timeout=20,**kw):
 p=subprocess.run(args,capture_output=True,timeout=timeout,**kw)
 return p,dict(returncode=p.returncode,stdout=p.stdout.decode(errors='replace')[-20000:],stderr=p.stderr.decode(errors='replace')[-6000:])
p,state=cmd(['docker','inspect','--format','{{.State.Status}} {{.State.Pid}}',cid]);r['container_guard']=state
assert p.returncode==0 and p.stdout.decode().startswith('running '),'do not start/restart a container'
preflight="""from pathlib import Path
import hashlib,json,platform,os
h=Path('/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h');lib=Path('/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so')
assert hashlib.sha256(h.read_bytes()).hexdigest()=='62d127fbb899b1476e054b1815ca6e0c71a362868c655f0e3da4616fb7598fb7'
assert platform.machine()=='aarch64'
assert not Path(OUT).exists()
print(json.dumps(dict(header_sha256=hashlib.sha256(h.read_bytes()).hexdigest(),library_path=str(lib.resolve()),library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),architecture=platform.machine(),output_dir_preexisting=False)))
""".replace('OUT',repr(out))
p,env=cmd(['docker','exec',cid,'python3','-I','-S','-c',preflight]);r['SDK_preflight']=env
assert p.returncode==0,'pinned SDK mismatch or unavailable Python/library;no capture'
p,guard=cmd(['npu-smi','info']);r['resource_guard']=guard
assert p.returncode==0
s=p.stdout.decode()
# Conservative guard: the physical/chip map is unchanged and every reported card
# is idle. No memory allocation,load or capture has started before this check.
assert all(('No running processes found in NPU '+str(i))in s for i in range(8)),'occupied or unknown card;skip'
device_rows=[line for line in s.splitlines()if '0000:'in line]
assert len(device_rows)==16 and all(line.split('|')[3].strip().split()[0]=='0'for line in device_rows),'non-idle cores or unknown inventory;skip'
assert any('0000:9D:00.0'in line and line.split('|')[1].split()==['0','0']for line in device_rows),'selected physical/logical device mapping changed'
r['status']='ONE_PROBE_SUBMITTED';r['submitted_monotonic']=time.monotonic()
p,attempt=cmd(['docker','exec','-i',cid,'timeout','--signal=TERM','--kill-after=5s','45s','python3','-',
 '--run','--device','0','--output',out,'--dump-own-graph'],timeout=58,input=payload)
r['attempt']=attempt;r['elapsed_seconds']=time.monotonic()-r['submitted_monotonic'];r['files']=[]
for name in ['probe-result.json','own-tiny-graph.json']:
 p,rec=cmd(['docker','cp',cid+':'+out+'/'+name,'-'])
 if p.returncode:
  r['files'].append(dict(name=name,status='NOT_CREATED_OR_UNAVAILABLE',returncode=p.returncode,stderr=rec['stderr']));continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as t:
  fs=[x for x in t.getmembers()if x.isfile()];assert len(fs)==1 and fs[0].size<=4*1024*1024
  raw=t.extractfile(fs[0]).read()
 r['files'].append(dict(name=name,status='READ',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),source_b64=base64.b64encode(raw).decode()))
p,after=cmd(['npu-smi','info']);r['resource_after']=after
r['status']='ONE_PROBE_ATTEMPT_FINISHED';print(json.dumps(r))
'''.replace('OUTPUT',repr(out)).replace('PAYLOAD',repr(base64.b64encode(b).decode())).replace('EXPECTED',repr(expected))
    control=N/'ssh166-probe-v109.sock';assert control.exists()
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(control),'-o','ControlMaster=no','-o','ProxyCommand=false',
        '-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn',
        'python3 -I -S -c '+shlex.quote(code)]
    # Record intent before submission so a client observation failure cannot trigger a retry.
    write(N/'operand-probe-submission-v109b.json',dict(authorized_by='User explicit reply:允许最小探针',
        source_sha256=expected,output_dir=out,device=0,submitted_utc=now.isoformat(),at_most_one_attempt=True))
    p=subprocess.run(args,capture_output=True,text=True,timeout=105);result=None
    for line in p.stdout.splitlines():
        try:
            v=json.loads(line)
            if isinstance(v,dict)and v.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=v
        except ValueError:pass
    write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,
        result=result,stdout_if_no_result=p.stdout[-8000:]if result is None else None,stderr=p.stderr[-6000:],at_most_one_attempt=True))
    assert result is not None,(p.returncode,p.stderr[-1500:])
    folder=N/'operand-probe-results-v109b';folder.mkdir(exist_ok=False)
    files=[]
    for r in result['files']:
        row={k:v for k,v in r.items()if k!='source_b64'}
        if r['status']=='READ':
            raw=base64.b64decode(r['source_b64']);assert hashlib.sha256(raw).hexdigest()==r['sha256']
            path=folder/r['name'];path.write_bytes(raw);row['local_path']=str(path)
        files.append(row)
    write(O/'operand-probe-result-index-v109b.json',dict(files=files,attempt_status=result['status'],
        output_dir=out,source_sha256=expected,device=0,at_most_one_attempt=True,
        no_model_load_or_graph_replay=True,saved_model_operands_correlated=False,overall_complete=False))
    observed=read(folder/'probe-result.json')if(folder/'probe-result.json').exists()else{}
    print(json.dumps(dict(attempt_status=result['status'],probe_status=observed.get('status'),
        error=observed.get('error'),returned_tasks=len(observed.get('tasks',[])),files=files,
        no_model_load_or_graph_replay=True)))

if __name__=='__main__':main()
