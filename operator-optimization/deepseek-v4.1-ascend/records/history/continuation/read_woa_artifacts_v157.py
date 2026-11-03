"""Read only operator-specific source and metadata headers, never model values."""
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, subprocess
from finalize_origin_coverage_v9 import N, O, read, write
from cache_saved_raw_evidence_v155 import ssh_args
from inspect_saved_raw_profile_v155 import BASE


def main():
    output=O/'WOA-saved-artifacts-v157.json';assert not output.exists()
    container=read(O/'raw-record-schema-source-index-v155.json')['container_id']
    code='''import os,socket,json,hashlib,subprocess,tarfile,base64,shlex,struct
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
base=Path(%r);container=%r
obj=json.loads(subprocess.run(['docker','inspect',container],capture_output=True,text=True,check=True).stdout)[0];assert obj['Id']==container and not obj['State']['Running']
serve=base/'runtime/scripts/serve.sh';b=serve.read_bytes();lines=[x for x in b.decode().splitlines()if x.startswith('exec /usr/local/python3.12.13/bin/vllm serve ')];assert len(lines)==1;argv=shlex.split(lines[0]);model_in_container=Path(argv[3]);assert model_in_container.is_absolute()
mounts=[m for m in obj['Mounts']if m['Type']=='bind'and(model_in_container==Path(m['Destination'])or Path(m['Destination'])in model_in_container.parents)];assert mounts;mount=max(mounts,key=lambda m:len(Path(m['Destination']).parts));model=Path(mount['Source'])/model_in_container.relative_to(mount['Destination']);assert model.is_dir()
def filemeta(path,payload=True,limit=16*1024**2):
 assert path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(model.resolve());before=path.stat();assert before.st_size<=limit;b=path.read_bytes();after=path.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 r=dict(relative=str(path.relative_to(model)),bytes=len(b),mtime_ns=before.st_mtime_ns,sha256=hashlib.sha256(b).hexdigest(),source_stat_unchanged=True)
 if payload:r['payload']=base64.b64encode(b).decode()
 return b,r
root_entries=[dict(name=p.name,is_file=p.is_file(),is_symlink=p.is_symlink(),bytes=p.lstat().st_size)for p in sorted(model.iterdir())];assert len(root_entries)<250
config,config_source=filemeta(model/'config.json');indexes=[p for p in model.iterdir()if p.is_file()and p.name.endswith('.safetensors.index.json')];assert len(indexes)==1
index_bytes,index_source=filemeta(indexes[0]);index=json.loads(index_bytes);selected={k:v for k,v in index['weight_map'].items()if'.wo_a.'in k or k.startswith('wo_a.')};assert selected
quant=[]
for name in['quant_model_description.json','quantization_config.json']:
 if(model/name).is_file():
  qb,qs=filemeta(model/name);quant.append(qs)
headers=[];total=0
for shard in sorted(set(selected.values())):
 rel=Path(shard);assert not rel.is_absolute()and'..'not in rel.parts;path=model/rel;assert path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(model.resolve());before=path.stat()
 with path.open('rb')as f:
  prefix=f.read(8);assert len(prefix)==8;length=struct.unpack('<Q',prefix)[0];assert 2<=length<64*1024**2 and length+8<=before.st_size;header=f.read(length);assert len(header)==length
 total+=length;assert total<128*1024**2;parsed=json.loads(header);wanted={k:parsed[k]for k,v in selected.items()if v==shard};after=path.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 for key,tensor in wanted.items():
  lo,hi=tensor['data_offsets'];assert 0<=lo<=hi<=before.st_size-8-length
 headers.append(dict(shard=shard,source_file_bytes=before.st_size,source_mtime_ns=before.st_mtime_ns,header_bytes=length,header_sha256=hashlib.sha256(header).hexdigest(),header_payload=base64.b64encode(header).decode(),WOA_tensors=wanted,source_stat_unchanged=True,weight_payload_not_read=True,full_file_sha256_not_computed=True))
source_specs=['/vllm-workspace/vllm-ascend/vllm_ascend/ops/linear.py','/vllm-workspace/vllm/vllm/model_executor/parameter.py'];sources=[]
for path in source_specs:
 p=subprocess.Popen(['docker','cp','-L',container+':'+path,'-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);found=[]
 with tarfile.open(fileobj=p.stdout,mode='r|')as tar:
  for m in tar:
   assert m.size<512*1024
   if m.isfile():
    b=tar.extractfile(m).read();found.append(dict(container_path=path,member=m.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),payload=base64.b64encode(b).decode()))
 rc=p.wait(timeout=5);err=p.stderr.read().decode(errors='replace');assert rc==0 and len(found)==1,(path,rc,err[-400:]);sources.extend(found)
print(json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),stopped_container_id=container,image=obj['Image'],serve_sha256=hashlib.sha256(serve.read_bytes()).hexdigest(),model_container_path=str(model_in_container),model_host_path=str(model),model_mount=dict(source=mount['Source'],destination=mount['Destination'],RW=mount['RW']),root_entries=root_entries,config_source=config_source,index_source=index_source,quant_sources=quant,index_WOA_entries=selected,safetensors_headers=headers,operator_loader_sources=sources,metadata_bytes_read=len(config)+len(index_bytes)+total,weight_payload_bytes_read=0,read_only=True,new_device_work=False,historical_loaded_weight_values_or_full_residency_attested=False)))
'''%(BASE,container)
    process=subprocess.run(ssh_args(code),capture_output=True,timeout=60);assert process.returncode==0,process.stderr.decode(errors='replace')[-1400:]
    result=json.loads(process.stdout);cache=N/'WOA-artifact-cache-v157';cache.mkdir(exist_ok=False)
    metadata=[result['config_source'],result['index_source']]+result['quant_sources']
    for row in metadata:
        b=base64.b64decode(row.pop('payload'));assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'];p=cache/'checkpoint_metadata'/row['relative'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);row['local_path']=str(p)
    for row in result['safetensors_headers']:
        b=base64.b64decode(row.pop('header_payload'));assert len(b)==row['header_bytes']and hashlib.sha256(b).hexdigest()==row['header_sha256'];p=cache/'checkpoint_headers'/(row['shard']+'.header.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);row['header_local_path']=str(p)
    for row in result['operator_loader_sources']:
        b=base64.b64decode(row.pop('payload'));assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'];p=cache/'source'/row['container_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);row['local_path']=str(p)
    result.update(observed_utc=datetime.now(timezone.utc).isoformat(),overall_complete=False);write(output,result)
    print(json.dumps(dict(status='PASS',model_container_path=result['model_container_path'],WOA_tensors=len(result['index_WOA_entries']),header_files=len(result['safetensors_headers']),loader_sources=len(result['operator_loader_sources']),metadata_bytes_read=result['metadata_bytes_read'],weight_payload_bytes_read=0,new_device_work=False)))


if __name__=='__main__':main()
