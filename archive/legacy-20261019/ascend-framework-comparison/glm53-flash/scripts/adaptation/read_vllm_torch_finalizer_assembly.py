# SPDX-License-Identifier: Apache-2.0
"""Read-only disassembly of actual R141 free caller via anonymous memfd.
No library execution/dlopen, no remote filesystem write, no NPU/model call.
"""
import pathlib,json,subprocess,shlex
P=pathlib.Path(__file__).resolve().parents[2];R=P/'runs/20261017-r130-vllm-native-loader-audit';assert not (R/'torch-finalizer-assembly.json').exists()
code=r'''import json,subprocess,socket,tarfile,io,struct,hashlib,os,shutil
assert socket.gethostname()=='A6-15U-Atlas900RCKA3-14';name='if-glm53-flash-r003-vllm14';v=json.loads(subprocess.check_output(['docker','inspect',name]))[0];assert not v['State']['Running'] and v['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83' and v['Config']['Labels']['project_id']=='glm53-flash';path='/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib/libtorch_python.so';q=subprocess.run(['docker','cp','-L',name+':'+path,'-'],capture_output=True,timeout=45);assert q.returncode==0
with tarfile.open(fileobj=io.BytesIO(q.stdout)) as t:
 f=[m for m in t if m.isfile()];assert len(f)==1;b=t.extractfile(f[0]).read()
shoff=struct.unpack_from('<Q',b,40)[0];ents,n,si=struct.unpack_from('<HHH',b,58);sections=[struct.unpack_from('<IIQQQQIIQQ',b,shoff+i*ents) for i in range(n)];target=0x6a3680;near=[]
for s in sections:
 if s[1] not in (2,11):continue
 st=sections[s[6]];strings=b[st[4]:st[4]+st[5]]
 for o in range(s[4],s[4]+s[5],s[9]):
  index,info,other,section,vv,ss=struct.unpack_from('<IBBHQQ',b,o)
  if vv<=target and vv+max(ss,1)>=target and index:near.append({'symbol':strings[index:strings.index(b'\0',index)].decode(errors='replace'),'RVA':vv,'size':ss})
value=target-256;size=512
objdump=shutil.which('objdump');assert objdump
fd=os.memfd_create('flash-dvpp-readonly-elf',flags=0);os.write(fd,b)
try:
 q=subprocess.run([objdump,'-d','-C','--start-address='+str(value),'--stop-address='+str(value+size),'/proc/self/fd/'+str(fd)],capture_output=True,text=True,timeout=20,pass_fds=(fd,));assert q.returncode==0
finally:os.close(fd)
assert len(q.stdout)<128*1024;print(json.dumps({'project_id':'glm53-flash','resource_mutations':0,'remote_filesystem_writes':0,'container_exec':False,'library_never_loaded':True,'source_SHA256':hashlib.sha256(b).hexdigest(),'symbol':'R149 torch firstfreecaller, notrootcauseproof','near_symbols':near,'symbol_RVA':value,'symbol_size':size,'actual_failure_return_offset':256,'objdump':objdump,'disassembly':q.stdout}))
'''
ssh=['ssh','-S','/Users/wio/work/sglang-glm53-pd/ssh-control','-p','60011','-oBatchMode=yes','-oConnectTimeout=15','root@61.241.77.34'];cmd=['kubectl','exec','-i','-n','default','zhixing-dockerctl-25xz2','-c','hostctl','--','nsenter','-t1','-m','--','python3','-'];q=subprocess.run(ssh+[shlex.join(cmd)],input=code,capture_output=True,text=True,timeout=90);assert q.returncode==0,q.stderr[-1500:];v=json.loads(q.stdout);(R/'torch-finalizer-assembly.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'sha256':v['source_SHA256'],'near_symbols':v['near_symbols'],'disassembly':v['disassembly']}))
