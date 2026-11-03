"""Read/disassemble only three pinned native Pad/MemSet objects; CPU tool only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,read,write


def main():
    receipt=N/'native-static-disassembly166-v128.json';assert not receipt.exists()
    rows=read(O/'pad-provider-binary-index-v111.json')['files'];assert len(rows)==3
    controls=[dict(path=r['path'],bytes=r['bytes'],sha256=r['sha256'])for r in rows]
    inner='CONTROLS='+repr(controls)+'\n'+r'''
from pathlib import Path
import hashlib,json,subprocess
tool=Path('/usr/local/Ascend/ascend-toolkit/latest/bin/llvm-objdump')
assert tool.is_file()
blob=tool.read_bytes()
version=subprocess.run([str(tool),'--version'],capture_output=True,text=True,timeout=10)
records=[]
for row in CONTROLS:
    path=Path(row['path']);data=path.read_bytes()
    assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256']
    p=subprocess.run([str(tool),'-d',str(path)],capture_output=True,text=True,timeout=20)
    assert len(p.stdout)<2*1024*1024
    records.append(dict(**row,command=[str(tool),'-d',str(path)],returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
print(json.dumps(dict(status='STATIC_CPU_DISASSEMBLY_FINISHED',tool=dict(path=str(tool),resolved=str(tool.resolve()),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),version_returncode=version.returncode,version_stdout=version.stdout,version_stderr=version.stderr),objects=records,
    original_objects_modified=False,CANN_or_Torch_runtime_loaded=False,new_device_work=False)))
'''
    host=r'''
import socket,os,subprocess,json
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
cid='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'
before=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=10).stdout)[0]
assert before['Id']==cid and before['State']['Running']and before['State']['Pid']==2032500
p=subprocess.run(['docker','exec',cid,'python3','-I','-S','-c',INNER],capture_output=True,text=True,timeout=80)
after=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=10).stdout)[0]
assert after['State']['Running']and after['State']['Pid']==2032500
print(json.dumps(dict(hostname=socket.gethostname(),container_id=cid,container_pid=2032500,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
'''.replace('INNER',repr(inner))
    argv=['/usr/bin/ssh','-F','/dev/null','-S',str(N/'ssh166-native-static-v128.sock'),'-o','ControlMaster=no','-o','ProxyCommand=false',
        '-o','BatchMode=yes','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(host)]
    p=subprocess.run(argv,capture_output=True,text=True,timeout=100);envelope=None
    for line in p.stdout.splitlines():
        try:
            row=json.loads(line)
            if row.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':envelope=row
        except(ValueError,AttributeError):pass
    write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,envelope=envelope,stderr=p.stderr,new_device_work=False))
    assert p.returncode==0 and envelope and envelope['returncode']==0,(p.returncode,p.stderr)
    result=json.loads(envelope['stdout']);directory=N/'sdk-native-static-v128';directory.mkdir(exist_ok=False)
    extracted=[]
    for row in result['objects']:
        q=directory/(Path(row['path']).stem+'.disasm.txt');q.write_text(row['stdout'])
        extracted.append(dict(**{k:v for k,v in row.items()if k!='stdout'},local_disassembly=str(q),disassembly_bytes=q.stat().st_size,disassembly_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
    write(O/'native-static-disassembly-index-v128.json',dict(status=result['status'],tool=result['tool'],objects=extracted,
        original_objects_modified=False,CANN_or_Torch_runtime_loaded=False,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status=result['status'],objects=[dict(name=Path(r['path']).name,returncode=r['returncode'],disassembly_bytes=r['disassembly_bytes'],stderr=r['stderr'])for r in extracted],new_device_work=False)))


if __name__=='__main__':main()
