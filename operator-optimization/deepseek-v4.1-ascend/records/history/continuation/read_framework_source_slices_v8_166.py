from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,socket,subprocess,tarfile,io
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
container='wio-vllm-v41-repro166-stage-20261030'
p=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Image}} {{.State.Running}}',container],capture_output=True,text=True,check=True)
cid,image,running=p.stdout.strip().split()
assert cid=='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'
assert image=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
specs=[('vllm/distributed/device_communicators/shm_broadcast.py',[(167,207),(730,812),(878,900)]),
       ('vllm/v1/executor/multiproc_executor.py',[(70,111),(414,451)]),
       ('vllm/distributed/utils.py',[(40,52)])]
sources=[]
for suffix,ranges in specs:
    path='/vllm-workspace/vllm/'+suffix
    p=subprocess.run(['docker','cp','-L',container+':'+path,'-'],capture_output=True,check=True,timeout=30)
    with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*') as tar:
        members=[m for m in tar.getmembers() if m.isfile()]
        assert len(members)==1
        raw=tar.extractfile(members[0]).read()
    lines=raw.decode().splitlines();assert all(1<=a<=b<=len(lines) for a,b in ranges)
    sources.append(dict(container_path=path,sha256=hashlib.sha256(raw).hexdigest(),line_count=len(lines),
                        excerpts=[dict(start=a,end=b,lines=[f'{i+1}: {lines[i]}' for i in range(a-1,b)]) for a,b in ranges]))
out=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/framework-source-slices-v8.json')
assert not out.exists()
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),container_id=cid,image=image,container_running_observed=running,
            source_method='docker cp three exact files, no exec/start/restart/import',sources=sources,no_device_or_inference_work=True,
            limits=['This is a fixed migrated-image/source snapshot, not attestation of historical loaded bytes or functionCPUtime.',
                    'Only specified source slices reviewed; do not infer other code paths absent.'])
payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
result['full_result']=str(out);result['result_sha256']=hashlib.sha256(payload.encode()).hexdigest()
print(json.dumps(result,ensure_ascii=False))
