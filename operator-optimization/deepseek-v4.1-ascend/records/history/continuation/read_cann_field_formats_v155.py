"""Read the minimal record layouts/file routing and archived HCCL address fields."""
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, subprocess, zlib
from finalize_origin_coverage_v9 import N, O, read, write
from cache_saved_raw_evidence_v155 import ssh_args
from inspect_saved_raw_profile_v155 import BASE


def main():
    output=O/'raw-cann-field-format-index-v155.json';assert not output.exists()
    previous=read(O/'raw-record-schema-source-index-v155.json');container=previous['container_id']
    wanted=['data_struct_size_constant.py','ge_tensor_base_bean.py','struct_decoder.py','file_name_manager.py']
    members=next(r['members']for r in previous['results']if r['spec']['label']=='cann_profiler_analysis')
    selected=[m['name']for m in members if m['kind']=='file'and Path(m['name']).name in wanted];assert len(selected)==len(wanted)
    inv=read(O/'raw-profile-inventory-v155.json');plan=[z for z in inv['files']if 'additional.hccl_info.slice_'in Path(z['relative']).name and not z['relative'].endswith('.done')];assert len(plan)==64 and sum(z['bytes']for z in plan)==3037184
    packed=base64.b64encode(zlib.compress(json.dumps(plan).encode())).decode()
    code='''import os,socket,json,tarfile,subprocess,base64,hashlib,zlib
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
container=%r;selected=set(%r);plan=json.loads(zlib.decompress(base64.b64decode(%r)));base=Path(%r)
state=subprocess.run(['docker','inspect','--format','{{.State.Running}} {{.Image}}',container],capture_output=True,text=True,check=True).stdout.strip();assert state==%r
p=subprocess.Popen(['docker','cp','-L',container+':/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);schemas=[];total=0
with tarfile.open(fileobj=p.stdout,mode='r|')as tar:
 for m in tar:
  total+=m.size;assert total<64*1024**2
  if m.name in selected:
   assert m.isfile()and m.size<256*1024;b=tar.extractfile(m).read();schemas.append(dict(member=m.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),payload=base64.b64encode(b).decode()))
rc=p.wait(timeout=5);assert rc==0 and{z['member']for z in schemas}==selected
files=[]
for spec in plan:
 p=base/spec['relative'];assert not p.is_symlink()and p.resolve().is_relative_to(base.resolve());s=p.stat();assert(s.st_size,s.st_mtime_ns)==(spec['bytes'],spec['mtime_ns']);b=p.read_bytes();t=p.stat();assert(s.st_size,s.st_mtime_ns)==(t.st_size,t.st_mtime_ns);files.append(dict(relative=spec['relative'],bytes=len(b),source_mtime_ns=s.st_mtime_ns,sha256=hashlib.sha256(b).hexdigest(),source_stat_unchanged=True,payload=base64.b64encode(b).decode()))
print(json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),container_id=container,stopped_image_state=state,schemas=schemas,raw_hccl_files=files,read_only_stopped_filesystem_and_original_archive=True,new_device_work=False)))
'''%(container,selected,packed,BASE,'false '+previous['expected_image'])
    process=subprocess.run(ssh_args(code),capture_output=True,timeout=60);assert process.returncode==0,process.stderr.decode(errors='replace')[-1000:];data=json.loads(process.stdout)
    for key,root,relkey in [('schemas',N/'raw-record-schema-source-v155'/'cann_formats','member'),('raw_hccl_files',N/'saved-raw-evidence-v155'/'hccl','relative')]:
        for row in data[key]:
            b=base64.b64decode(row.pop('payload'));assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'];rel=Path(row[relkey]);assert not rel.is_absolute()and'..'not in rel.parts;path=root/rel;assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b);row['local_path']=str(path)
    data.update(observed_utc=datetime.now(timezone.utc).isoformat(),overall_complete=False);write(output,data)
    print(json.dumps(dict(status='PASS',schemas=len(data['schemas']),raw_files=len(data['raw_hccl_files']),raw_bytes=sum(z['bytes']for z in data['raw_hccl_files']),new_device_work=False)))


if __name__=='__main__':main()
