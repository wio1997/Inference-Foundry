"""Fetch bounded original archived field records, not a new profile/workload."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import subprocess,tarfile,json,hashlib,zlib,base64,io,shlex
from finalize_origin_coverage_v9 import N,O,read,write
from inspect_saved_raw_profile_v155 import BASE


def ssh_args(code):
 control=N/'ssh166-v155b-xinyufeng.sock';ident=read(N/'original166-xinyufeng-recheck-v155b.json');assert ident['application_status']=='REMOTE_ROOT_IDENTITY_RETURNED'and ident['remote_result']['hostname']=='bm-jn-zs-zone1-910C-64G-10-105'and ident['remote_result']['euid']==0;assert control.exists(),'Verified155bmaster absent;do not fallback.'
 return ['/usr/bin/ssh','-F','/dev/null','-S',str(control),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]


def main():
 cache=N/'saved-raw-evidence-v155';archive=cache/'raw-field-records-v155.tar.gz';out=O/'raw-field-cache-index-v155.json';receipt=N/'raw-field-cache-receipt-v155.json';assert not out.exists()and not receipt.exists()and not archive.exists();cache.mkdir(exist_ok=True);inv=read(O/'raw-profile-inventory-v155.json');plan=[]
 needles=['tensor_info','capture_op_info','task_track','hash_dic','type_info_dic','node_basic_info','context_id_info']
 for z in inv['files']:
  p=Path(z['relative']);name=p.name
  if name.endswith('.done'):continue
  pick=('/host/data/'in z['relative']and any(t in name for t in needles))or name in['info.json','sample.json','profiler_metadata.json']or name.startswith('info.json.')or name=='torch.op_mark'
  prefix=4096 if name=='torch.op_range'else None
  if pick or prefix:
   assert not z['is_symlink'];plan.append([z['relative'],z['bytes'],z['mtime_ns'],prefix])
 for rel in['runtime-versions.json','npu-smi-samples.json','diagnostic-identity.json','preflight.json']:plan.append([rel,None,None,None])
 estimated=sum(min(z[1],z[3])if z[3]is not None else(z[1]if z[1]is not None else 4*1024**2)for z in plan);assert estimated<225*1024**2 and len(plan)<1000
 write(O/'raw-field-cache-plan-v155.json',dict(original_profile_root=BASE,planned_files=len(plan),planned_payload_bytes_upper=estimated,plan=[dict(relative=z[0],source_bytes=z[1],source_mtime_ns=z[2],prefix_bytes=z[3])for z in plan],full_torch_op_range_copied=False,source_op_range_bytes_from_inventory=sum(z['bytes']for z in inv['files']if Path(z['relative']).name=='torch.op_range'),no_remote_or_device_mutation=True,overall_complete=False))
 packed=base64.b64encode(zlib.compress(json.dumps(plan,separators=(',',':')).encode())).decode()
 code='''import os,socket,json,time,tarfile,io,hashlib,zlib,base64,stat
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
base=Path(%r);plan=json.loads(zlib.decompress(base64.b64decode(%r)));manifest=[];start=time.monotonic()
class DigestReader:
 def __init__(self,f):self.f=f;self.h=hashlib.sha256()
 def read(self,n=-1):b=self.f.read(n);self.h.update(b);return b
with tarfile.open(fileobj=__import__('sys').stdout.buffer,mode='w|gz',compresslevel=1)as tar:
 for rel,expected_size,expected_mtime,prefix in plan:
  assert time.monotonic()-start<110
  rr=Path(rel);assert not rr.is_absolute()and'..'not in rr.parts;p=base/rr;assert p.resolve().is_relative_to(base.resolve())and not p.is_symlink();before=p.stat();assert p.is_file()
  if expected_size is not None:assert(before.st_size,before.st_mtime_ns)==(expected_size,expected_mtime)
  else:assert before.st_size<=4*1024**2
  size=before.st_size if prefix is None else min(prefix,before.st_size);assert size<=16*1024**2
  member='data/'+rel+('.prefix4096'if prefix is not None else'');info=tarfile.TarInfo(member);info.size=size
  if prefix is None:
   with p.open('rb')as f:
    reader=DigestReader(f);tar.addfile(info,reader);digest=reader.h.hexdigest()
  else:
   with p.open('rb')as f:b=f.read(size)
   tar.addfile(info,io.BytesIO(b));digest=hashlib.sha256(b).hexdigest()
  after=p.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  manifest.append(dict(relative=rel,archive_member=member,source_bytes=before.st_size,source_mtime_ns=before.st_mtime_ns,payload_bytes=size,payload_sha256=digest,payload_is_full_file=prefix is None,source_stat_unchanged=True))
 links=[];root=base/'runtime'/'official-traces'
 for parent,dirs,files in os.walk(root,followlinks=False):
  for name in dirs:
   p=Path(parent)/name
   if p.is_symlink():links.append(dict(relative=str(p.relative_to(base)),target=os.readlink(p)))
 payload=json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),original_root=str(base),files=manifest,symlink_directories_not_followed=links,elapsed_seconds=time.monotonic()-start,read_only_archived_inputs=True,new_device_model_or_capture=False)).encode();info=tarfile.TarInfo('manifest.json');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
'''%(BASE,packed)
 partial=archive.with_suffix('.partial')
 with partial.open('wb')as f:p=subprocess.run(ssh_args(code),stdout=f,stderr=subprocess.PIPE,timeout=125)
 rec=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,stderr=p.stderr.decode(errors='replace')[-1500:],archive_bytes=partial.stat().st_size,read_only_original_archive=True,new_device_work=False,overall_complete=False);write(receipt,rec);assert p.returncode==0,rec;partial.replace(archive)
 with tarfile.open(archive,'r:gz')as tar:
  members=tar.getmembers();assert all(m.isfile()and not m.issym()and not m.islnk()for m in members);m=tar.getmember('manifest.json');manifest=json.load(tar.extractfile(m));assert manifest['hostname']=='bm-jn-zs-zone1-910C-64G-10-105'and manifest['euid']==0;expected={z[0]:z for z in plan};assert len(manifest['files'])==len(expected)==len(plan);assert set(m.name for m in members)=={'manifest.json'}|{z['archive_member']for z in manifest['files']}
  result=[]
  for z in manifest['files']:
   pp=expected[z['relative']];assert z['source_stat_unchanged']and(pp[1]is None or(z['source_bytes'],z['source_mtime_ns'])==(pp[1],pp[2]));member=z['archive_member'];rel=Path(member);assert not rel.is_absolute()and'..'not in rel.parts;dest=cache/'files'/rel;assert not dest.exists();b=tar.extractfile(member).read();assert len(b)==z['payload_bytes']and hashlib.sha256(b).hexdigest()==z['payload_sha256'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b);result.append(dict(**z,local_path=str(dest),hash_matched=True))
 write(out,dict(generated_utc=datetime.now(timezone.utc).isoformat(),original_root=BASE,cache_root=str(cache),archive_path=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=result,file_count=len(result),payload_bytes=sum(z['payload_bytes']for z in result),full_source_bytes=sum(z['source_bytes']for z in result),symlink_directories_not_followed=manifest['symlink_directories_not_followed'],all_planned_payloads_hash_matched=True,all_original_profile_files_copied=False,torch_op_range_full_copied=False,new_remote_model_device_or_capture=False,overall_complete=False))
 print(json.dumps(dict(status='PASS',files=len(result),payload_bytes=sum(z['payload_bytes']for z in result),archive_bytes=archive.stat().st_size,full_files=sum(z['payload_is_full_file']for z in result),op_range_prefixes=sum(not z['payload_is_full_file']for z in result),symlink_dirs=len(manifest['symlink_directories_not_followed']),new_device_work=False)))


if __name__=='__main__':main()
