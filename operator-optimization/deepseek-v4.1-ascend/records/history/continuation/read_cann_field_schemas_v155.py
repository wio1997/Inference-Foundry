"""Read actual matching CANN record schemas from a stopped capture container."""
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, subprocess
from finalize_origin_coverage_v9 import N, O, read, write
from cache_saved_raw_evidence_v155 import ssh_args


def main():
    output = O/'raw-cann-schema-source-index-v155.json'
    assert not output.exists()
    previous = read(O/'raw-record-schema-source-index-v155.json')
    wanted = ['tensor_add_info_bean.py','tensor_add_info_parser.py','hccl_info_bean.py','hccl_info_parser.py','task_track_bean.py','task_track_parser.py','memcpy_info_bean.py','memcpy_info_parser.py','node_basic_info_bean.py','compact_info_bean.py','capture_stream_info_bean.py','capture_stream_info_parser.py','data_tag.py','hccl_info_dto.py']
    members = next(r['members'] for r in previous['results'] if r['spec']['label']=='cann_profiler_analysis')
    selected = [m['name'] for m in members if m['kind']=='file' and Path(m['name']).name in wanted]
    assert len(selected)==len(wanted), (len(selected),len(wanted))
    container = previous['container_id']
    code = '''import os,socket,json,tarfile,subprocess,base64,hashlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
container=%r;selected=set(%r)
state=subprocess.run(['docker','inspect','--format','{{.State.Running}} {{.Image}}',container],capture_output=True,text=True,check=True).stdout.strip();assert state==%r
p=subprocess.Popen(['docker','cp','-L',container+':/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);files=[];total=0;members=0
with tarfile.open(fileobj=p.stdout,mode='r|')as tar:
 for m in tar:
  total+=m.size;members+=1;assert total<64*1024**2 and members<10000
  if m.name in selected:
   assert m.isfile()and m.size<256*1024;b=tar.extractfile(m).read();files.append(dict(member=m.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),payload=base64.b64encode(b).decode()))
rc=p.wait(timeout=5);err=p.stderr.read().decode(errors='replace');assert rc==0 and{z['member']for z in files}==selected
print(json.dumps(dict(container_id=container,state=state,returncode=rc,stderr=err[-500:],files=files,scanned_member_bytes=total,scanned_members=members,read_only_stopped_filesystem=True,no_container_start_exec_or_device_work=True)))
''' % (container,selected,'false '+previous['expected_image'])
    result = subprocess.run(ssh_args(code),capture_output=True,timeout=60)
    assert result.returncode==0,result.stderr.decode(errors='replace')[-1000:]
    data=json.loads(result.stdout)
    for row in data['files']:
        payload=base64.b64decode(row.pop('payload'));assert len(payload)==row['bytes']and hashlib.sha256(payload).hexdigest()==row['sha256'];path=N/'raw-record-schema-source-v155'/'cann_fields'/row['member'];assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload);row['local_path']=str(path)
    data.update(observed_utc=datetime.now(timezone.utc).isoformat(),schema_names_come_from_actual_metadata=True,overall_complete=False)
    write(output,data);print(json.dumps(dict(status='PASS',files=len(data['files']),bytes=sum(r['bytes']for r in data['files']),new_device_work=False)))


if __name__=='__main__': main()
