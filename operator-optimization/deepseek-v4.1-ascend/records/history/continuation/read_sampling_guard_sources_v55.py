"""Read finite sampler/metadata dependencies and small generation JSON, without execution."""
from datetime import datetime,timezone
from pathlib import Path
import base64,hashlib,json
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import run_readonly_python
import mirror_saved_profile_core_v53 as transport
from read_original_source_archive_v53 import CONTAINER
transport.CONTROL=N/'ssh166-v55-xinyufeng.sock'


def main():
    assert not(O/'sampling-guard-inputs-v55.json').exists()
    base='/vllm-workspace/'
    controls={base+'vllm/vllm/v1/sample/sampler.py':'f95805b2c6d06218c724826aa5f881a606e8d0a534a566aee7c85948520dd216',
              base+'vllm-ascend/vllm_ascend/sample/sampler.py':'867f1770614bb594baf5ecfb858aa4a20dc413cbfbe4f94e06b3786900b581ee'}
    extra=[base+p for p in ['vllm/vllm/v1/sample/metadata.py','vllm/vllm/v1/worker/gpu_input_batch.py',
        'vllm/vllm/v1/engine/input_processor.py','vllm/vllm/v1/sample/logits_processor/__init__.py',
        'vllm/vllm/v1/sample/logits_processor/builtin.py','vllm-ascend/vllm_ascend/worker/npu_input_batch.py']]
    models=['/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/'+n for n in ['generation_config.json','config.json']]
    paths=list(controls)+extra+models
    code='''import os,socket,subprocess,hashlib,tarfile,io,base64,json
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
container=%r;controls=%r;paths=%r;rows=[]
for path in paths:
 p=subprocess.run(['docker','cp',container+':'+path,'-'],capture_output=True,timeout=20)
 if p.returncode:
  rows.append(dict(path=path,status='READ_ERROR',stderr=p.stderr.decode(errors='replace')[-700:]));continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as tar:
  fs=[m for m in tar.getmembers()if m.isfile()];assert len(fs)==1 and fs[0].size<2*1024*1024
  b=tar.extractfile(fs[0]).read();sha=hashlib.sha256(b).hexdigest()
 if path in controls:assert sha==controls[path]
 rows.append(dict(path=path,status='READ',bytes=len(b),sha256=sha,archive_mtime=fs[0].mtime,historical_control_match=path in controls,source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(files=rows,container_id=container,container_process_started=False)))
'''%(CONTAINER,controls,paths)
    p=run_readonly_python(code,timeout=220)
    receipt=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,stdout=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace'),no_model_device_service_health_process_work=True)
    write(N/'sampling-guard-source166-v55.json',receipt);assert p.returncode==0,receipt['stderr'][-1000:]
    data=json.loads(receipt['stdout']);rows=[]
    for row in data['files']:
        if row['status']!='READ':rows.append(row);continue
        b=base64.b64decode(row['source_b64']);assert hashlib.sha256(b).hexdigest()==row['sha256']
        dest=N/'sampling-guard-source-v55'/Path(row['path']).relative_to('/')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
        rows.append(dict({k:v for k,v in row.items()if k!='source_b64'},local_path=str(dest)))
    write(O/'sampling-guard-inputs-v55.json',dict(files=rows,two_historical_sampler_controls_match=sum(r.get('historical_control_match',False)for r in rows)==2,
        current_stage_not_historical_loaded_object_snapshot=True,no_target_execution=True,overall_complete=False))
    print(json.dumps(dict(read=sum(r['status']=='READ'for r in rows),errors=[r['path']for r in rows if r['status']!='READ'],overall_complete=False)))


if __name__=='__main__':main()
