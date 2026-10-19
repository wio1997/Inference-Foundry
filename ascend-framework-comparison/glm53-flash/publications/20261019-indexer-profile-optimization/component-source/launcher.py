#!/usr/bin/env python3
"""R246 source-only launcher: node46, 16 dies, C5 image, one bounded dispatch.

No model weights are mounted; every tensor is synthesized in component.py.
Run `python3 launcher.py --manifest` to print the frozen payload SHA256 without
touching the host.  A real dispatch is `python3 launcher.py` and is once-only.
"""

import base64
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

R = pathlib.Path(__file__).resolve().parent
RUN_DIR_HOST = "/data/wio/inference-foundry/glm53-flash/runs/20261019-r246-dynamic-rowstride-tp16-score-component"
IMAGE = "sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263"
CONTAINER = "if-glm53-flash-r246-rowstride-component"

PAYLOAD = (
    "score_ops/flash_kpool_score.py",
    "score_ops/flash_kpool_score_stride.py",
    "score_ops/flash_indexer_query_parallel.py",
    "score_ops/flash_kpool_score_abi_audit.py",
    "score_ops/flash_kpool_model_audit_npu.py",
    "score_ops/flash_indexer_projection_audit.py",
    "component.py",
)

# Filled from the frozen files below; a mismatch stops before any SSH.
EXPECTED = {'score_ops/flash_kpool_score.py': '44a6f449b24955ac406007a7ddf26409b862518a96079a1dec93fca3990db15c', 'score_ops/flash_kpool_score_stride.py': 'b1821eb60ef3991fcb6a0b6e9c67ae7e8dd0b8c10fde96ece4c0253a28dbef1c', 'score_ops/flash_indexer_query_parallel.py': '636ce1b23c9ecaa50753d6a30c202bf4de9153e3ba4a1544bb39499299b2811c', 'score_ops/flash_kpool_score_abi_audit.py': '568bd87e790ee891d44371250843956451b350b429eaa93d61943730ae23a37a', 'score_ops/flash_kpool_model_audit_npu.py': 'cb2eb2e48bfacd38e7b1972dd15e91d158f0b3481242a6211472339e84521d2c', 'score_ops/flash_indexer_projection_audit.py': 'e7ccd577f6f66b820d5b4046f73e89ebedaed04f3a35c7acfa4e93488b817e8b', 'component.py': '6243e5fe842b8562d22ed924d66de099485f90255dcf246400425519b1db560f'}

HOST_PROBE = r'''import base64,hashlib,json,os,pathlib,socket,subprocess,sys,time
assert socket.gethostname()=='B8-15U-Atlas900RCKA3-46'
R=pathlib.Path('/data/wio/inference-foundry/glm53-flash/runs/20261019-r246-dynamic-rowstride-tp16-score-component')
if R.exists():
 print(json.dumps({'passed':False,'error':'owned run directory already exists: '+str(R)}));sys.exit(1)
R.mkdir(parents=True)
os.close(os.open(R/'once.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
FILES=__FILES__
MANIFEST_SHA=__MANIFEST_SHA__
name='if-glm53-flash-r246-rowstride-component'
image='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
out=dict(project_id='glm53-flash',unit='r246',node=46,container=name,image=image,manifest_sha256=MANIFEST_SHA,namespace='glm53-flash-r246',model_requests=0,weights_mounted=False,pd_used=False)
created=False
launch_attempted=False
work_deadline=time.monotonic()+900
def run(args,timeout=30):
 q=subprocess.run(args,capture_output=True,text=True,errors='replace',timeout=timeout)
 return dict(rc=q.returncode,stdout=q.stdout,stderr=q.stderr)
try:
 out['pre_npu']=run(['npu-smi','info'])
 assert out['pre_npu']['stdout'].count('No running processes found')==8,'node46 is not idle-8 before launch'
 port=socket.socket();port.bind(('127.0.0.1',29510));port.close()
 assert run(['docker','image','inspect',image,'--format','{{.Id}}'])['stdout'].strip()==image,'C5 image id mismatch'
 assert run(['docker','inspect',name])['rc']!=0,'owned container name already exists'
 for n,x in FILES.items():
  p=pathlib.PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
  b=base64.b64decode(x['data']);assert hashlib.sha256(b).hexdigest()==x['sha256'],n
  (R/p).parent.mkdir(parents=True,exist_ok=True);(R/p).write_bytes(b)
 assert not (R/'audit-enabled').exists(),'audit-enabled must be absent'
 args=['docker','run','-d','--pull=never','--name',name,
  '--label','project_id=glm53-flash','--label','run_id=r246','--label','namespace=glm53-flash-r246','--label','owner=zcode',
  '--network=host','--ipc=private','--shm-size=16g','--memory=64g','--cpus=48','--pids-limit=2048',
  '--ulimit','core=0:0','--ulimit','memlock=-1:-1',
  '-v',str(R)+':/run-evidence',
  '-e','ASCEND_RT_VISIBLE_DEVICES='+','.join(map(str,range(16))),
  '-e','HCCL_IF_IP=172.16.12.46','-e','HCCL_SOCKET_IFNAME=bond4.12',
  '-e','HCCL_OP_EXPANSION_MODE=AIV','-e','HCCL_BUFFSIZE=400',
  '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
  '-e','TRITON_CACHE_DIR=/run-evidence/triton-cache',
  '-e','FLASH_PD_SCORE_ABI_AUDIT=0','-e','FLASH_PD_COMPRESSION_ABI_AUDIT=0',
  '-e','FLASH_PD_PERF_MODE=1','-e','FLASH_SDMA_VERIFY_BYTES=0',
  '-e','R246_MANIFEST_SHA='+MANIFEST_SHA,
  '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1']
 ops='/sgl-workspace/sglang/python/sglang/srt/hardware_backend/npu/ops/'
 for src in sorted((R/'score_ops').glob('*.py')):args+=['-v',str(src)+':'+ops+src.name+':ro']
 for path in ['/usr/local/Ascend/driver','/usr/local/dcmi','/etc/ascend_install.info','/usr/local/bin/npu-smi']:
  if pathlib.Path(path).exists():args+=['-v',path+':'+path+':ro']
 for path in [f'/dev/davinci{i}' for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:
  assert pathlib.Path(path).exists(),path;args+=['--device',path]
 container_budget=max(1,int(work_deadline-time.monotonic()-90))
 args+=['--entrypoint','/bin/bash',image,'-c','source /usr/local/Ascend/ascend-toolkit/set_env.sh && exec timeout --signal=TERM --kill-after=15 '+str(container_budget)+' python3 -m torch.distributed.run --nnodes=1 --nproc-per-node=16 --master-addr=127.0.0.1 --master-port=29510 /run-evidence/component.py > /run-evidence/workers.log 2>&1']
 out['command']=args
 launch_attempted=True
 out['container_budget_seconds']=container_budget
 out['launch']=run(args,60);assert out['launch']['rc']==0,out['launch'];created=True
 try:
  out['wait']=run(['docker','wait',name],max(1,work_deadline-time.monotonic()-45))
 except subprocess.TimeoutExpired:
  out['wait']=dict(rc=None,stdout='',stderr='work-deadline wait timeout');out['bounded_wait_timeout']=True
finally:
 if launch_attempted:
  q=run(['docker','inspect',name])
  if q['rc']!=0:out['post_error']='inspect failed'
  else:
   v=json.loads(q['stdout'])[0]
   owned=(v.get('Image')==image and v.get('Config',{}).get('Labels',{}).get('project_id')=='glm53-flash' and v.get('Config',{}).get('Labels',{}).get('run_id')=='r246')
   out['owned_container_verified']=owned
   if owned and v['State']['Running']:out['bounded_stop']=run(['docker','stop','--time','15',name],30)
   out['post_inspect']=run(['docker','inspect',name,'--format','{{.State.Running}} {{.State.ExitCode}}'])
   out['logs']=run(['docker','logs','--tail','160',name])
 for _ in range(6):
  out['post_npu']=run(['npu-smi','info'])
  if out['post_npu']['stdout'].count('No running processes found')==8:break
  time.sleep(3)
 ranks=[]
 for i in range(16):
  p=R/f'rank{i}.json'
  ranks.append(json.loads(p.read_text()) if p.exists() else dict(rank=i,passed=False,error='missing_rank_artifact'))
 out['ranks']=ranks
 sa=R/'stageA-summary.json'
 out['stage_a']=json.loads(sa.read_text()) if sa.exists() else dict(passed=False,error='missing_stageA_summary')
 out['rank_passed']=sum(1 for x in ranks if x.get('passed'))
 out['passed']=(all(x.get('passed') for x in ranks) and out['stage_a'].get('passed') is True and out.get('owned_container_verified') is True and not out.get('bounded_wait_timeout',False) and out.get('post_inspect',{}).get('stdout','').strip()=='false 0' and out.get('post_npu',{}).get('stdout','').count('No running processes found')==8)
 try:
  import tarfile
  artifacts=[R/f'rank{i}.json' for i in range(16) if (R/f'rank{i}.json').exists()]
  artifacts+=[p for p in sorted((R/'stageA').glob('*.json')) if p.is_file()]
  artifacts+=[p for p in sorted((R/'stageA').glob('*.log')) if p.is_file()]
  artifacts+=[p for p in sorted((R/'stageA').glob('*.bin')) if p.is_file()]
  artifacts+=[p for p in [R/'stageA-summary.json',R/'workers.log'] if p.exists()]
  artifacts+=[p for p in sorted((R/'score_ops').glob('*.py')) if p.is_file()]
  artifacts+=[R/'component.py']
  assert len(artifacts)<=512 and sum(p.stat().st_size for p in artifacts)<=512*1024**2
  arc=R/'operator-artifacts.tar.gz'
  with tarfile.open(arc,'w:gz',compresslevel=1) as t:
   for p in artifacts:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
  assert arc.stat().st_size<=256*1024**2
  out['archive']=dict(bytes=arc.stat().st_size,sha256=hashlib.sha256(arc.read_bytes()).hexdigest(),files={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in artifacts})
 except BaseException as error:
  out['archive_error']=repr(error);out['passed']=False
 (R/'host-receipt.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out))
'''


def manifest():
    files = {}
    for name in PAYLOAD:
        path = R / name
        if not path.is_file():
            raise SystemExit("missing payload: " + name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if EXPECTED and EXPECTED.get(name) != digest:
            raise SystemExit("frozen SHA mismatch for %s: %s != %s" % (name, digest, EXPECTED.get(name)))
        files[name] = digest
    composite = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return files, composite


def dispatch(files, composite):
    lock = R / "dispatch.once.lock"
    os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    embedded = {
        name: {
            "sha256": digest,
            "data": base64.b64encode((R / name).read_bytes()).decode(),
        }
        for name, digest in files.items()
    }
    code = HOST_PROBE.replace("__FILES__", repr(embedded)).replace(
        "__MANIFEST_SHA__", repr(composite)
    )
    gateway = (
        "import json,socket,subprocess\n"
        "assert socket.gethostname()=='A3-5U-Atlas900RCKA3-1'\n"
        "q=subprocess.run(['kubectl','exec','-i','-n','default',"
        "'sglang-container-access-46-20261013','-c','hostctl','--','nsenter','-t','1','-n','-m','--','python3','-'],"
        "input=" + repr(code) + ",capture_output=True,text=True,timeout=1020)\n"
        "print(json.dumps({'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr}))\n"
    )
    started = time.time()
    completed = subprocess.run(
        [
            "ssh",
            "-S",
            "/tmp/glm53-flash-r236-control",
            "-p",
            "60011",
            "-oBatchMode=yes",
            "-oConnectTimeout=15",
            "-oServerAliveInterval=15",
            "-oServerAliveCountMax=3",
            "root@61.241.77.34",
            "python3",
            "-",
        ],
        input=gateway,
        text=True,
        capture_output=True,
        errors="replace",
        timeout=1050,
    )
    receipt = None
    remote = None
    try:
        remote = json.loads(completed.stdout)
        receipt = json.loads(remote["stdout"])
    except BaseException:
        receipt = None
    result = dict(
        unit="r246",
        manifest_sha256=composite,
        files_sha256=files,
        run_dir_host=RUN_DIR_HOST,
        image=IMAGE,
        container=CONTAINER,
        ssh_rc=completed.returncode,
        elapsed_s=time.time() - started,
        stdout_tail=completed.stdout[-4000:],
        stderr_tail=completed.stderr[-4000:],
        receipt=receipt,
        remote=remote,
    )
    (R / "launch-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest_sha256": composite,
                "ssh_rc": completed.returncode,
                "passed": None if receipt is None else receipt.get("passed"),
                "rank_passed": None if receipt is None else receipt.get("rank_passed"),
                "stage_a": None if receipt is None else receipt.get("stage_a", {}).get("observed"),
                "post_inspect": None if receipt is None else receipt.get("post_inspect", {}).get("stdout", "").strip(),
                "archive": None if receipt is None else receipt.get("archive"),
            }
        )
    )
    return 0 if completed.returncode == 0 and remote and remote.get("rc") == 0 and receipt and receipt.get("passed") else 1


def main():
    files, composite = manifest()
    if "--manifest" in sys.argv:
        print(json.dumps({"manifest_sha256": composite, "files_sha256": files}, indent=2))
        return 0
    return dispatch(files, composite)


if __name__ == "__main__":
    sys.exit(main())
