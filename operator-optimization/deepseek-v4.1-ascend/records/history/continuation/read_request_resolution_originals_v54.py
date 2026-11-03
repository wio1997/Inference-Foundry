"""Read only saved capture records and exact direct source dependencies."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json

from finalize_origin_coverage_v9 import N, O, read, write
from mirror_saved_profile_core_v53 import fetch_pinned, run_readonly_python
from read_original_source_archive_v53 import CONTAINER


def main():
    assert not (O/'request-resolution-inputs-v54.json').exists()
    pins = {
        'runtime/scripts/serve.sh':'c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba',
        'runtime/scripts/serve-host-profile.sh':'2cd68992493ef97750ae3fd9832d3c301bec319f22b657cf4f07f6a8ac3f3766',
        'runtime/logs/service.log':'a02e582afaf94f0911b0bfb62ddac9028f2ac25936e4776373fc0ebfd2e1c61e',
        'runtime/official_profile_client.py':'42a6d9e54ea0a922843e8ab37f4096e52c54fb4dbcdd23b77bb60d69293f3131',
    }
    files=[fetch_pinned(rel,sha)for rel,sha in pins.items()]
    controls={'/vllm-workspace/vllm/vllm/entrypoints/openai/completion/serving.py':'75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527'}
    paths=list(controls)+['/vllm-workspace/vllm/vllm/'+p for p in [
        'entrypoints/openai/completion/protocol.py','sampling_params.py','config/model.py',
        'v1/engine/parallel_sampling.py','v1/engine/__init__.py']]
    code='''import os,socket,json,hashlib,subprocess,tarfile,io,base64
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
caps=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/profile-capabilities-index-v12-v2.json')
assert hashlib.sha256(caps.read_bytes()).hexdigest()=='79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095'
container=%r;paths=%r;controls=%r;rows=[]
for path in paths:
 p=subprocess.run(['docker','cp',container+':'+path,'-'],capture_output=True,timeout=20)
 if p.returncode:
  rows.append(dict(path=path,status='READ_ERROR',stderr=p.stderr.decode(errors='replace')[-1000:]));continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as tar:
  members=[m for m in tar.getmembers()if m.isfile()];assert len(members)==1 and members[0].size<2*1024*1024
  b=tar.extractfile(members[0]).read()
 sha=hashlib.sha256(b).hexdigest()
 if path in controls:assert sha==controls[path]
 rows.append(dict(path=path,status='READ',sha256=sha,bytes=len(b),source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(files=rows,container_id=container,container_process_started=False)))
'''%(CONTAINER,paths,controls)
    p=run_readonly_python(code,timeout=150)
    receipt=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,
                 stdout=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace'),
                 no_model_import_device_service_health_process_work=True)
    write(N/'request-resolution-source166-v54.json',receipt)
    assert p.returncode==0,receipt['stderr'][-1000:]
    d=json.loads(receipt['stdout']);sources=[]
    for r in d['files']:
        if r['status']!='READ':sources.append(r);continue
        b=base64.b64decode(r['source_b64']);assert hashlib.sha256(b).hexdigest()==r['sha256']
        dest=N/'request-resolution-source-v54'/Path(r['path']).relative_to('/vllm-workspace')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
        sources.append(dict({k:v for k,v in r.items()if k!='source_b64'},local_path=str(dest)))
    write(O/'request-resolution-inputs-v54.json',dict(saved_files=files,source_files=sources,
        all_source_files_read=all(r['status']=='READ'for r in sources),
        source_scope='Currentoriginal166stage files;controlservinghashmatchesv53;not historical loadedobject attestation.',
        new_capture_model_device_work=False,whole_module_closed=False,overall_complete=False))
    print(json.dumps(dict(saved_files=len(files),source_files_read=sum(r['status']=='READ'for r in sources),
        errors=[r['path']for r in sources if r['status']!='READ'],overall_complete=False)))


if __name__=='__main__':main()
