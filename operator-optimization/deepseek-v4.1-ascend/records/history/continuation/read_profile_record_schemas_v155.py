"""Read only record-format files from the stopped exact capture image, no container start/exec."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,base64,hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from cache_saved_raw_evidence_v155 import ssh_args


def main():
 output=O/'raw-record-schema-source-index-v155.json';assert not output.exists();av=read(O/'raw-decoder-availability-v155.json');target=av['container_metadata'][0];assert target['returncode']==0 and not target['running']and target['image']==av['archived_capture_expected_image'];container=target['id']
 code='''import os,socket,subprocess,tarfile,json,base64,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
container=%r
specs=[dict(label='torch_beans',path='/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_bean',selected=['_torch_op_bean.py','_op_mark_bean.py','_memory_use_bean.py']),dict(label='torch_tlv',path='/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_common_func/_tlv_decoder.py',selected=['_tlv_decoder.py']),dict(label='cann_profiler_analysis',path='/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis',selected=[])]
results=[]
for spec in specs:
 p=subprocess.Popen(['docker','cp','-L',container+':'+spec['path'],'-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);members=[];chosen=[];total=0;stopped=None;error=None;start=time.monotonic()
 try:
  with tarfile.open(fileobj=p.stdout,mode='r|')as tar:
   for m in tar:
    total+=m.size;members.append(dict(name=m.name,bytes=m.size,kind='file'if m.isfile()else'dir'if m.isdir()else'link'if m.issym()else'other',linkname=m.linkname))
    if total>64*1024**2 or len(members)>10000 or time.monotonic()-start>35:stopped='bounded_metadata_scan';p.terminate();break
    if m.isfile()and m.name.rsplit('/',1)[-1]in spec['selected']:
     assert m.size<256*1024;b=tar.extractfile(m).read();chosen.append(dict(member=m.name,payload=base64.b64encode(b).decode()))
 except (tarfile.TarError,OSError)as e:error=type(e).__name__+':'+str(e)
 try:rc=p.wait(timeout=5)
 except subprocess.TimeoutExpired:p.kill();rc=p.wait();stopped='local_docker_cp_process_deadline'
 err=p.stderr.read().decode(errors='replace')[-1200:];results.append(dict(spec=spec,returncode=rc,members=members,selected=chosen,scanned_member_bytes=total,stopped=stopped,error=error,stderr=err))
print(json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),container_id=container,expected_image=%r,results=results,operation='Read-only docker cp archive stream from stoppedcapturecontainer,onlyrecord-schemafilesselected;no start/exec/attach/Torch/CANN/device imports/model/profile.')))
'''%(container,target['image'])
 p=subprocess.run(ssh_args(code),capture_output=True,timeout=125);assert p.returncode==0,p.stderr[-800:];x=json.loads(p.stdout);cache=N/'raw-record-schema-source-v155';files=[]
 for r in x['results']:
  for z in r['selected']:
   b=base64.b64decode(z.pop('payload'));rel=Path(r['spec']['label'])/z['member'];assert'..'not in rel.parts;dest=cache/rel;assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b);z.update(local_path=str(dest),bytes=len(b),sha256=hashlib.sha256(b).hexdigest());files.append(dict(label=r['spec']['label'],**z))
 x.update(observed_utc=datetime.now(timezone.utc).isoformat(),files=files,new_device_work=False,overall_complete=False);write(output,x);print(json.dumps(dict(status='READ_COMPLETE',files=[Path(z['local_path']).name for z in files],directories=[dict(label=r['spec']['label'],returncode=r['returncode'],members=len(r['members']),scanned_bytes=r['scanned_member_bytes'],stopped=r['stopped'],error=r['error'])for r in x['results']],new_device_work=False)))


if __name__=='__main__':main()
