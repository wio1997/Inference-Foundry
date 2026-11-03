"""Complete the configured checkpoint header census, reusing fifty saved headers."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import base64, hashlib, json, subprocess, zlib
from finalize_origin_coverage_v9 import N, O, read, write
from server_readonly_v159 import ssh_args


def main():
    output=O/'checkpoint-all-header-index-v159.json';receipt=N/'checkpoint-all-header-receipt-v159.json';assert not output.exists()and not receipt.exists()
    previous=read(O/'checkpoint-operator-metadata-v159.json')['models'][0];index_meta=previous['index'];index=read(Path(index_meta['local_path']));all_shards=sorted(set(index['weight_map'].values()));old={x['shard']:x for x in previous['headers']};missing=[x for x in all_shards if x not in old];assert missing
    packed=base64.b64encode(zlib.compress(json.dumps(missing).encode())).decode()
    code='''import os,socket,json,base64,zlib,hashlib,struct,time
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
root=Path(%r);index=root/%r;assert hashlib.sha256(index.read_bytes()).hexdigest()==%r
plan=json.loads(zlib.decompress(base64.b64decode(%r)));rows=[];total=0;begin=time.monotonic()
for name in plan:
 assert time.monotonic()-begin<45
 rel=Path(name);assert not rel.is_absolute()and'..'not in rel.parts;p=root/rel;assert p.is_file()and not p.is_symlink()and p.resolve().is_relative_to(root.resolve());before=p.stat()
 with p.open('rb')as f:
  prefix=f.read(8);assert len(prefix)==8;length=struct.unpack('<Q',prefix)[0];assert 2<=length<32*1024**2 and length+8<=before.st_size;header=f.read(length);assert len(header)==length
 total+=length;assert total<64*1024**2;after=p.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns);body=json.loads(header)
 for name0,t in body.items():
  if name0=='__metadata__':continue
  lo,hi=t['data_offsets'];assert 0<=lo<=hi<=before.st_size-8-length
 rows.append(dict(shard=name,source_file_bytes=before.st_size,source_mtime_ns=before.st_mtime_ns,header_bytes=length,header_sha256=hashlib.sha256(header).hexdigest(),payload=base64.b64encode(header).decode(),source_stat_unchanged=True,weight_payload_not_read=True,full_file_sha256_not_computed=True))
print(json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),headers=rows,header_bytes_read=total,weight_payload_bytes_read=0,index_sha_unchanged=True,new_device_work=False,elapsed_seconds=time.monotonic()-begin)))
'''%(previous['host_path'],index_meta['relative'],index_meta['sha256'],packed)
    process=subprocess.run(ssh_args(code),capture_output=True,timeout=60);raw=N/'checkpoint-all-header-raw-v159.json';raw.write_bytes(process.stdout);write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=process.returncode,stdout_path=str(raw),stdout_bytes=len(process.stdout),stdout_sha256=hashlib.sha256(process.stdout).hexdigest(),stderr=process.stderr.decode(errors='replace')[-2000:],new_device_work=False));assert process.returncode==0
    result=json.loads(process.stdout);assert result['hostname']=='bm-jn-zs-zone1-910C-64G-10-105'and result['euid']==0
    headers={name:{k:v for k,v in row.items()if k!='selected_tensors'}for name,row in old.items()}
    for row in result['headers']:
        data=base64.b64decode(row.pop('payload'));assert len(data)==row['header_bytes']and hashlib.sha256(data).hexdigest()==row['header_sha256'];p=N/'checkpoint-operator-cache-v159/headers/target'/(row['shard']+'.header.json');assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);row['header_local_path']=str(p);headers[row['shard']]=row
    assert set(headers)==set(all_shards);dtype=Counter();names_seen=set();indexed_seen=set();extra=[];payload_bytes=0;quant=read(Path(next(x['local_path']for x in previous['metadata']if x['relative']=='quant_model_description.json')));quant_categories=Counter()
    for shard,row in sorted(headers.items()):
        data=Path(row['header_local_path']).read_bytes();assert len(data)==row['header_bytes']and hashlib.sha256(data).hexdigest()==row['header_sha256'];body=json.loads(data)
        for name,t in body.items():
            if name=='__metadata__':continue
            assert name not in names_seen;names_seen.add(name);dtype[t['dtype']]+=1;payload_bytes+=t['data_offsets'][1]-t['data_offsets'][0];quant_categories[str(quant.get(name,'UNSPECIFIED'))]+=1
            if name in index['weight_map']:assert index['weight_map'][name]==shard;indexed_seen.add(name)
            else:extra.append(dict(name=name,shard=shard))
    assert indexed_seen==set(index['weight_map'])
    record=dict(version=159,origin='Currentconfiguredcheckpointmetadataonly;notloadedoperatorinventoryorhistoricalvalueattestation.',index_source=index_meta,headers=[headers[x]for x in sorted(headers)],header_files=len(headers),headers_reused_without_remote_rescan=len(old),new_headers_read=len(result['headers']),indexed_tensor_count=len(indexed_seen),all_header_tensor_count=len(names_seen),extra_header_tensors=extra,dtype_label_counts=dict(dtype),quant_description_category_counts=dict(quant_categories),total_declared_tensor_payload_bytes=payload_bytes,full_index_header_crosscheck=True,weight_payload_bytes_read=0,full_weight_file_hashes_computed=False,current_loaded_or_historical_weight_values_attested=False,new_device_work=False,overall_complete=False)
    write(output,record)
    summary=read(O/'checkpoint-profiler-summary-v159.json');summary.update(all_checkpoint_headers=record['header_files'],all_indexed_tensor_metadata=record['indexed_tensor_count'],all_header_dtype_label_counts=record['dtype_label_counts'],full_checkpoint_header_inventory_complete=True,header_inventory_is_not_actual_load_inventory=True);write(O/'checkpoint-profiler-summary-v159.json',summary)
    inputs=read(O/'checkpoint-profiler-inputs-v159.json');paths={x['path']:x for x in inputs['inputs']}
    for p in[output,receipt]+[Path(x['header_local_path'])for x in headers.values()]:paths[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    inputs['inputs']=[paths[x]for x in sorted(paths)];write(O/'checkpoint-profiler-inputs-v159.json',inputs)
    print(json.dumps(dict(status='PASS',headers=record['header_files'],reused=record['headers_reused_without_remote_rescan'],new=record['new_headers_read'],indexed_tensors=record['indexed_tensor_count'],dtype_labels=record['dtype_label_counts'],extra_header_tensors=len(extra),weight_payload_bytes_read=0,new_device_work=False)))


if __name__=='__main__':main()
