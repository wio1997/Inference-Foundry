"""One complete archived Torch field census, no device/framework imports."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import base64, hashlib, json, subprocess, zlib
from finalize_origin_coverage_v9 import N, O, read, write
from cache_saved_raw_evidence_v155 import ssh_args
from inspect_saved_raw_profile_v155 import BASE


def main():
    output = O / 'raw-torch-field-census-v155.json'
    assert not output.exists()
    inventory = read(O / 'raw-profile-inventory-v155.json')
    plan = [x for x in inventory['files'] if Path(x['relative']).name == 'torch.op_range']
    assert len(plan) == 32 and sum(x['bytes'] for x in plan) == 3214148270
    all_results = []
    for batch in range(8):
        receipt = O / ('raw-torch-field-census-batch%02d-v155.json' % batch)
        selected = plan[batch * 4:batch * 4 + 4]
        if receipt.exists():
            result = read(receipt)
            assert result['plan'] == selected
        else:
            packed = base64.b64encode(zlib.compress(json.dumps(selected).encode())).decode()
            code = '''import os,socket,json,base64,zlib,struct,hashlib,time
from pathlib import Path
from collections import Counter
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
base=Path(%r);plan=json.loads(zlib.decompress(base64.b64decode(%r)));results=[];begin=time.monotonic()
head=struct.Struct('<HI');constant=struct.Struct('<3q4QB?');assert constant.size==58
for spec in plan:
 p=base/spec['relative'];assert not p.is_symlink() and p.resolve().is_relative_to(base.resolve());before=p.stat();assert(before.st_size,before.st_mtime_ns)==(spec['bytes'],spec['mtime_ns'])
 digest=hashlib.sha256();offset=0;records=0;tags=Counter();tagbytes=Counter();outertypes=Counter();duplicates=Counter();tag_records=Counter();maxrecord=0;samples=[];errors=[]
 with p.open('rb',buffering=2**20)as f:
  while offset<before.st_size:
   assert time.monotonic()-begin<50,'bounded batch read deadline'
   raw=f.read(6);digest.update(raw)
   if len(raw)!=6:errors.append(dict(offset=offset,error='truncated_outer_header'));break
   typ,length=head.unpack(raw);outertypes[typ]+=1
   if length<58 or offset+6+length>before.st_size or length>16*1024**2:errors.append(dict(offset=offset,error='invalid_outer_length',length=length));break
   payload=f.read(length);digest.update(payload)
   if len(payload)!=length:errors.append(dict(offset=offset,error='truncated_outer_payload'));break
   meta=constant.unpack_from(payload);inner=58;seen=Counter();name=None;interesting=[]
   while inner<length:
    if inner+6>length:errors.append(dict(offset=offset,inner=inner,error='truncated_field_header'));break
    tag,size=head.unpack_from(payload,inner);inner+=6
    if inner+size>length:errors.append(dict(offset=offset,inner=inner,error='invalid_field_length',tag=tag,size=size));break
    tags[tag]+=1;tagbytes[tag]+=size;seen[tag]+=1
    if tag==2:name=payload[inner:inner+size].decode('utf-8',errors='replace')[:256]
    if tag in(5,6,7) and len(samples)<16:
     data=payload[inner:inner+size];interesting.append(dict(tag=tag,length=size,sha256=hashlib.sha256(data).hexdigest(),prefix_utf8=data[:384].decode('utf-8',errors='replace'),sample_is_full=size<=384))
    inner+=size
   for k,v in seen.items():
    tag_records[k]+=1
    if v>1:duplicates[k]+=v-1
   if interesting:samples.append(dict(record_offset=offset,name=name,raw_process_id=meta[3],raw_start_tid=meta[4],raw_start_counter=meta[0],fields=interesting))
   records+=1;maxrecord=max(maxrecord,length);offset+=6+length
   if errors:break
 after=p.stat();unchanged=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 results.append(dict(relative=spec['relative'],source_bytes=before.st_size,source_mtime_ns=before.st_mtime_ns,consumed_bytes=offset,sha256=digest.hexdigest(),records=records,outer_types=dict(outertypes),tag_occurrences=dict(tags),tag_record_counts=dict(tag_records),tag_payload_bytes=dict(tagbytes),duplicate_field_occurrences=dict(duplicates),max_record_bytes=maxrecord,samples=samples,errors=errors,source_stat_unchanged=unchanged,complete=not errors and offset==before.st_size and unchanged))
print(json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),files=results,elapsed_seconds=time.monotonic()-begin,read_only_archived_records=True,new_device_work=False)))
''' % (BASE, packed)
            process = subprocess.run(ssh_args(code), capture_output=True, timeout=60)
            assert process.returncode == 0, process.stderr.decode(errors='replace')[-1200:]
            result = json.loads(process.stdout)
            result.update(plan=selected, observed_utc=datetime.now(timezone.utc).isoformat())
            write(receipt, result)
        assert len(result['files']) == 4 and all(z['complete'] for z in result['files'])
        all_results.extend(result['files'])
        print(json.dumps(dict(batch=batch,files=4,bytes=sum(z['consumed_bytes']for z in result['files']),records=sum(z['records']for z in result['files']),elapsed_seconds=result['elapsed_seconds'])),flush=True)
    total = Counter(); records = Counter(); sizes = Counter()
    for row in all_results:
        total.update(row['tag_occurrences']);records.update(row['tag_record_counts']);sizes.update(row['tag_payload_bytes'])
    write(output,dict(version=155,scope='All32physicaltorch.op_rangefilesinthearchivedofficial-tracesinventory;schemafromexactstoppedcaptureimage.',schema_index=str(O/'raw-record-schema-source-index-v155.json'),constant_struct='<3q4QB?',constant_bytes=58,tlv_header='<HI',files=all_results,file_count=len(all_results),source_bytes=sum(z['source_bytes']for z in all_results),records=sum(z['records']for z in all_results),tag_occurrences=dict(total),tag_record_counts=dict(records),tag_payload_bytes=dict(sizes),all_complete=True,full_op_range_bytes_copied_to_local=False,no_field_absence_implies_tensor_independence=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='PASS',files=32,records=sum(z['records']for z in all_results),tags=dict(total),new_device_work=False)),flush=True)


if __name__ == '__main__': main()
