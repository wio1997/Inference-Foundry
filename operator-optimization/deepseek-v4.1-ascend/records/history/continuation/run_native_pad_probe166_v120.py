"""One newly authorized 44B native Pad capture, fresh guards and durable attempt record."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    receipt=N/'native-pad-probe166-v120.json';submission=N/'native-pad-submission-v120.json'
    assert not receipt.exists()and not submission.exists(),'never repeat a submitted diagnostic'
    auth=read(O/'native-pad-authorization-v119.json');assert auth['user_reply']=='允许本次 Pad 探针'and auth['status']=='AUTHORIZED_ONCE_NOT_YET_EXECUTED'
    script=N/'diagnostic-native-pad-probe-v111/inspect_pad_graph.py';b=script.read_bytes()
    expected='f48726e647c8bfb67b9b99fbc4cbec159d855af473ec4436964b7ca3850174ca';assert hashlib.sha256(b).hexdigest()==expected
    now=datetime.now(timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ');out='/tmp/wio-native-pad-v111-v120-'+stamp
    controller='/tmp/wio-native-pad-controller-v120-'+stamp+'.json'
    code='''import os,socket,subprocess,json,hashlib,base64,tarfile,io,time
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
cid='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'
out=OUTPUT;control=Path(CONTROLLER);payload=base64.b64decode(PAYLOAD)
assert hashlib.sha256(payload).hexdigest()==EXPECTED and not control.exists()
r=dict(hostname=socket.gethostname(),container=cid,output_dir=out,controller_receipt=str(control),device=0,
 status='GUARD_ONLY',capture_attempts=0,new_model_load=False,graph_replays=0,source_sha256=EXPECTED)
def save():control.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\\n')
def cmd(args,timeout=20,**kw):
 p=subprocess.run(args,capture_output=True,timeout=timeout,**kw)
 return p,dict(returncode=p.returncode,stdout=p.stdout.decode(errors='replace')[-24000:],stderr=p.stderr.decode(errors='replace')[-7000:])
save()
try:
 p,state=cmd(['docker','inspect','--format','{{.State.Status}} {{.State.Pid}}',cid]);r['container_guard']=state
 assert p.returncode==0 and p.stdout.decode().startswith('running '),'do not start/restart a container'
 preflight="""from pathlib import Path
import hashlib,json,platform,os
h=Path('/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h');lib=Path('/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so')
assert hashlib.sha256(h.read_bytes()).hexdigest()=='62d127fbb899b1476e054b1815ca6e0c71a362868c655f0e3da4616fb7598fb7'
assert hashlib.sha256(lib.read_bytes()).hexdigest()=='1c06b9c4a8910cfb585eab4ed0066bc09e28a0cf273dfcb15458f7cc49a78cea'
assert platform.machine()=='aarch64'and not Path(OUT).exists()
env={k:os.environ.get(k)for k in ['ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','ASCEND_DEVICE_ID']}
for k in ['ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES']:
 v=env[k]
 assert v in [None,'','all','All']or v.split(',')[0].strip()=='0','selectedlogical0mappingunknown'
assert env['ASCEND_DEVICE_ID']in[None,'','0'],'defaultdevicechanged'
print(json.dumps(dict(header_sha256=hashlib.sha256(h.read_bytes()).hexdigest(),library_path=str(lib.resolve()),library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),architecture=platform.machine(),visible_device_controls=env,output_dir_preexisting=False)))
""".replace('OUT',repr(out))
 p,env=cmd(['docker','exec',cid,'python3','-I','-S','-c',preflight]);r['SDK_preflight']=env
 assert p.returncode==0,'pinned SDK/device mapping mismatch;stop before library/device'
 p,guard=cmd(['npu-smi','info']);r['resource_guard']=guard;assert p.returncode==0
 s=p.stdout.decode();device_rows=[line for line in s.splitlines()if '0000:'in line]
 assert len(device_rows)==16,'unknown physical/chip inventory;skip'
 selected=[line for line in device_rows if line.split('|')[1].strip().split()[0]=='0']
 assert len(selected)==2 and all(line.split('|')[3].strip().split()[0]=='0'for line in selected),'card0occupiedorunknown;skip'
 assert 'No running processes found in NPU 0'in s,'card0processpresent;skip'
 assert any('0000:9D:00.0'in line and line.split('|')[1].split()==['0','0']for line in selected),'selectedphysical/logicalmappingchanged'
 r['status']='ONE_PROBE_SUBMITTED';r['capture_attempts']=1;r['submitted_monotonic']=time.monotonic();save()
 p,attempt=cmd(['docker','exec','-i',cid,'timeout','--signal=TERM','--kill-after=5s','45s','python3','-',
 '--run','--device','0','--output',out,'--dump-own-graph'],timeout=58,input=payload)
 r['attempt']=attempt;r['elapsed_seconds']=time.monotonic()-r['submitted_monotonic'];r['files']=[];save()
 for name in ['probe-result.json','own-native-pad-graph.json']:
  p,rec=cmd(['docker','cp',cid+':'+out+'/'+name,'-'])
  if p.returncode:
   r['files'].append(dict(name=name,status='NOT_CREATED_OR_UNAVAILABLE',returncode=p.returncode,stderr=rec['stderr']));continue
  with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as t:
   fs=[x for x in t.getmembers()if x.isfile()];assert len(fs)==1 and fs[0].size<=4*1024*1024;raw=t.extractfile(fs[0]).read()
  r['files'].append(dict(name=name,status='READ',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),source_b64=base64.b64encode(raw).decode()))
 r['status']='ONE_PROBE_ATTEMPT_FINISHED'
except Exception as e:
 r.update(status='GUARD_STOPPED'if r['capture_attempts']==0 else'ATTEMPT_STOPPED',error_type=type(e).__name__,error=str(e))
finally:
 try:
  p,after=cmd(['npu-smi','info']);r['resource_after']=after
  p,afterstate=cmd(['docker','inspect','--format','{{.State.Status}} {{.State.Pid}}',cid]);r['container_after']=afterstate
 except Exception as e:r['after_guard_error']=str(e)
 save();print(json.dumps(r))
'''.replace('OUTPUT',repr(out)).replace('CONTROLLER',repr(controller)).replace('PAYLOAD',repr(base64.b64encode(b).decode())).replace('EXPECTED',repr(expected))
    control=N/'ssh166-pad-v120.sock';assert control.exists()
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(control),'-o','ControlMaster=no','-o','ProxyCommand=false',
        '-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
    write(submission,dict(authorized_by='User explicit reply:允许本次 Pad 探针',source_sha256=expected,
        output_dir=out,controller_receipt=controller,device=0,submitted_utc=now.isoformat(),at_most_one_attempt=True))
    p=subprocess.run(args,capture_output=True,text=True,timeout=105);result=None
    for line in p.stdout.splitlines():
        try:
            item=json.loads(line)
            if isinstance(item,dict)and item.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=item
        except ValueError:pass
    write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,
        stdout_if_no_result=p.stdout[-8000:]if result is None else None,stderr=p.stderr[-5000:],at_most_one_attempt=True))
    assert result is not None,'Inspect durable remotecontrollerreceipt;never resubmit'
    folder=N/'native-pad-probe-results-v120';folder.mkdir(exist_ok=False);files=[]
    for item in result.get('files',[]):
        row={k:v for k,v in item.items()if k!='source_b64'}
        if item['status']=='READ':
            raw=base64.b64decode(item['source_b64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
            path=folder/item['name'];path.write_bytes(raw);row['local_path']=str(path)
        files.append(row)
    write(O/'native-pad-probe-result-index-v120.json',dict(files=files,attempt_status=result['status'],
        capture_attempts=result['capture_attempts'],output_dir=out,controller_receipt=controller,source_sha256=expected,
        device=0,at_most_one_attempt=True,no_model_load_or_graph_replay=True,saved_model_operands_correlated=False,overall_complete=False))
    observed=read(folder/'probe-result.json')if(folder/'probe-result.json').exists()else{}
    print(json.dumps(dict(attempt_status=result['status'],probe_status=observed.get('status'),error=result.get('error',observed.get('error')),
        returned_tasks=len(observed.get('tasks',[])),capture_attempts=result['capture_attempts'],no_model_load_or_graph_replay=True)))

if __name__=='__main__':main()
