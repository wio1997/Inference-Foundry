"""Read selected checkpoint headers and exact stopped-image operator/profiler source."""
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, subprocess
from finalize_origin_coverage_v9 import N, O, read, write
from server_readonly_v159 import ssh_args


def main():
    output=O/'checkpoint-operator-metadata-v159.json';receipt=N/'checkpoint-operator-metadata-receipt-v159.json';assert not output.exists()and not receipt.exists();roots=read(O/'checkpoint-roots-v159.json');container=roots['container_id'];specs=[dict(role=x['role'],host_path=x['host_path'],entries=x['entries'])for x in roots['roots']]
    code='''import os,socket,json,hashlib,base64,struct,subprocess,tarfile,time
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
container=%r;specs=%r;models=[];metadata_total=0;header_total=0;start=time.monotonic()
state=json.loads(subprocess.run(['docker','inspect',container],capture_output=True,text=True,check=True).stdout)[0];assert state['Id']==container and not state['State']['Running']and state['Image']==%r
def readmeta(path,root):
 assert path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(root.resolve());before=path.stat();assert before.st_size<16*1024**2;b=path.read_bytes();after=path.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 return b,dict(relative=str(path.relative_to(root)),bytes=len(b),source_mtime_ns=before.st_mtime_ns,sha256=hashlib.sha256(b).hexdigest(),source_stat_unchanged=True,payload=base64.b64encode(b).decode())
for spec in specs:
 root=Path(spec['host_path']);files=[]
 for name in['config.json','configuration.json','quant_model_description.json','quantization_config.json']:
  if(root/name).is_file():
   b,entry=readmeta(root/name,root);metadata_total+=len(b);files.append(entry)
 indices=[e['name']for e in spec['entries']if e['is_file']and e['name'].endswith('.safetensors.index.json')];assert len(indices)==1
 b,index=readmeta(root/indices[0],root);metadata_total+=len(b);parsed=json.loads(b);selected={k:v for k,v in parsed['weight_map'].items()if'.wo_a.'in k or'.wo_b.'in k or k.startswith(('wo_a.','wo_b.'))};assert selected;headers=[]
 for shard in sorted(set(selected.values())):
  assert time.monotonic()-start<45
  relative=Path(shard);assert not relative.is_absolute()and'..'not in relative.parts;path=root/relative;assert path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(root.resolve());before=path.stat()
  with path.open('rb')as f:
   prefix=f.read(8);assert len(prefix)==8;length=struct.unpack('<Q',prefix)[0];assert 2<=length<32*1024**2 and length+8<=before.st_size;header=f.read(length);assert len(header)==length
  header_total+=length;assert header_total<64*1024**2;body=json.loads(header);wanted={k:body[k]for k,v in selected.items()if v==shard};after=path.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  for tensor in wanted.values():
   lo,hi=tensor['data_offsets'];assert 0<=lo<=hi<=before.st_size-8-length
  headers.append(dict(shard=shard,source_file_bytes=before.st_size,source_mtime_ns=before.st_mtime_ns,header_bytes=length,header_sha256=hashlib.sha256(header).hexdigest(),header_payload=base64.b64encode(header).decode(),selected_tensors=wanted,source_stat_unchanged=True,weight_payload_not_read=True,full_file_sha256_not_computed=True))
 models.append(dict(role=spec['role'],host_path=str(root),metadata=files,index=index,selected_index_entries=selected,headers=headers))
source_specs=['/vllm-workspace/vllm-ascend/vllm_ascend/ops/linear.py','/vllm-workspace/vllm/vllm/model_executor/parameter.py','/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/version.py','/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/experimental_config.py','/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_bean/_param_tensor_bean.py'];sources=[]
for path in source_specs:
 p=subprocess.Popen(['docker','cp','-L',container+':'+path,'-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);found=[];error=None
 try:
  with tarfile.open(fileobj=p.stdout,mode='r|')as tar:
   for m in tar:
    assert m.size<512*1024
    if m.isfile():
     b=tar.extractfile(m).read();found.append(dict(container_path=path,member=m.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),payload=base64.b64encode(b).decode()))
 except (tarfile.TarError,OSError)as e:error=type(e).__name__+':'+str(e)
 rc=p.wait(timeout=5);stderr=p.stderr.read().decode(errors='replace')[-600:];sources.append(dict(container_path=path,returncode=rc,error=error,stderr=stderr,files=found))
print(json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),container_id=container,image=state['Image'],container_running=False,models=models,source_reads=sources,metadata_bytes_read=metadata_total,header_bytes_read=header_total,weight_payload_bytes_read=0,all_sources_read_only=True,new_device_work=False,historical_loaded_values_or_all_rank_residency_attested=False,elapsed_seconds=time.monotonic()-start)))
'''%(container,specs,roots['image'])
    process=subprocess.run(ssh_args(code),capture_output=True,timeout=60);raw=process.stdout;err=process.stderr.decode(errors='replace');raw_path=N/'checkpoint-operator-metadata-raw-v159.json';raw_path.write_bytes(raw);write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=process.returncode,stdout_path=str(raw_path),stdout_bytes=len(raw),stdout_sha256=hashlib.sha256(raw).hexdigest(),stderr=err,new_device_work=False));assert process.returncode==0
    result=json.loads(raw);assert result['hostname']=='bm-jn-zs-zone1-910C-64G-10-105'and result['euid']==0 and result['new_device_work']is False
    cache=N/'checkpoint-operator-cache-v159';cache.mkdir(exist_ok=False)
    for model in result['models']:
        for row in model['metadata']+[model['index']]:
            b=base64.b64decode(row.pop('payload'));assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'];p=cache/'metadata'/model['role']/row['relative'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);row['local_path']=str(p)
        for row in model['headers']:
            b=base64.b64decode(row.pop('header_payload'));assert len(b)==row['header_bytes']and hashlib.sha256(b).hexdigest()==row['header_sha256'];p=cache/'headers'/model['role']/(row['shard']+'.header.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);row['header_local_path']=str(p)
    for attempt in result['source_reads']:
        for row in attempt['files']:
            b=base64.b64decode(row.pop('payload'));assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'];p=cache/'source'/row['container_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);row['local_path']=str(p)
    result.update(observed_utc=datetime.now(timezone.utc).isoformat(),overall_complete=False);write(output,result)
    print(json.dumps(dict(status='PASS',models=[dict(role=x['role'],operator_tensor_entries=len(x['selected_index_entries']),header_files=len(x['headers']))for x in result['models']],metadata_bytes=result['metadata_bytes_read'],header_bytes=result['header_bytes_read'],source_reads=[dict(path=x['container_path'],returncode=x['returncode'],files=len(x['files']))for x in result['source_reads']],weight_payload_bytes_read=0,new_device_work=False)))


if __name__=='__main__':main()
