#!/usr/bin/env python3
"""Freeze a bounded probe, retain target-node evidence, and verify the downloaded raw result."""
import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

PODS = {32: "sglang-node-access-32", 33: "sglang-tuning-access-33-20261013"}
IMAGE = "quay.io/ascend/vllm-ascend:v0.30.0rc1-a3-arm64"
IMAGE_ID = "sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83"
REMOTE = "/data/wio/glm5-3-vllm-execution/20261017-inventory"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node", type=int, choices=PODS, required=True)
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--control", required=True, help="Existing SSH multiplex socket")
    p.add_argument("--image", action="store_true", help="Offline image; NPU mounts only with --npu-devices")
    p.add_argument("--npu-devices", help="Explicit devices for a model-free diagnostic under the task's NPU lease")
    p.add_argument("--external-rank", type=int, choices=[0, 1], help="2-node probe: node32 rank0, node33 rank1")
    p.add_argument("--master-port", type=int, default=29567)
    p.add_argument("--log-level", type=int, choices=[1, 3], default=3, help="1 for SDMA path diagnosis, 3 for timing control")
    p.add_argument("--timeout", type=int, default=150)
    a = p.parse_args()
    devices = [] if a.npu_devices is None else [int(s) for s in a.npu_devices.split(',')]
    if a.external_rank is not None and (len(devices) != 1 or a.external_rank != a.node - 32):
        p.error("External probe requires one explicit device and node32/rank0, node33/rank1")
    if devices and (not a.image or len(set(devices)) != len(devices)
                    or any(d < 0 or d > 15 for d in devices)):
        p.error('NPU diagnostic requires explicit unique devices 0..15 and --image')
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", a.label):
        p.error("invalid label")
    data = a.script.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    run = "node%d-%s-%s" % (a.node, a.label, sha[:12])
    remote = REMOTE + "/" + run
    # Transport uses the authenticated existing socket and hostPID access pod.
    ssh = ["ssh", "-S", a.control, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-p", "60011", "root@61.241.77.34"]
    host = "kubectl exec -i -n default %s -- nsenter -t 1 -m -- " % PODS[a.node]
    lease = None
    if devices:
        name = 'glm53-execution-reservation-%d-20261017' % a.node
        doc = json.loads(subprocess.check_output(ssh + ['kubectl get pod -n default ' + name + ' -o json']))
        assert doc['metadata']['labels']['owner'] == 'codex-execution-20261017'
        assert doc['spec']['nodeName'] == 'a3-5u-atlas900rcka3-%d' % a.node
        assert doc['status']['phase'] == 'Running'
        assert int(doc['spec']['containers'][0]['resources']['requests']['huawei.com/Ascend910']) == 16
        lease = {key: doc['metadata'][key] for key in ('name', 'uid', 'creationTimestamp')}
        lease.update(assigned_devices=doc['metadata'].get('annotations', {}).get('huawei.com/AscendReal'),
                     active_deadline_seconds=doc['spec'].get('activeDeadlineSeconds'))
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    context = {"client_started_utc": start, "node": a.node, "label": a.label, "npu_devices": devices,
        "script_sha256": sha, "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resource_lease": lease, "image_mode": a.image, "models_loaded": False,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest()}
    setup = {"root": remote, "script": base64.b64encode(data).decode(), "context": context,
             "timeout": a.timeout, "image": IMAGE if a.image else None, "devices": devices,
             "expected_image_id": IMAGE_ID, "node": a.node, "log_level": a.log_level, "external_rank": a.external_rank, "master_port": a.master_port}
    runner = '''import base64,datetime,hashlib,json,os,pathlib,re,signal,subprocess,time
c=json.loads(base64.b64decode(PAYLOAD))
r=pathlib.Path(c['root']);r.mkdir(parents=True,exist_ok=True)
s=r/'script.py';b=base64.b64decode(c['script'])
assert hashlib.sha256(b).hexdigest()==c['context']['script_sha256']
if s.exists(): assert s.read_bytes()==b
else: s.write_bytes(b)
assert not (r/'result.json').exists(), 'Existing run; use a new label instead of overwriting evidence'
(r/'context.json').write_text(json.dumps(c['context'],indent=2)+'\\n')
argv=['python3',str(s)]
image_id=None
if c['image']:
 image_id=subprocess.check_output(['docker','image','inspect',c['image'],'--format','{{.Id}}'],text=True).strip()
 assert image_id==c['expected_image_id'], 'Image tag changed; re-audit before running'
 argv=['docker','run','--rm','--pull','never','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','1','--memory','1g','--pids-limit','128','-e','PYTHONDONTWRITEBYTECODE=1','-v',str(r)+':/audit:ro','--entrypoint','python3',c['image'],'/audit/script.py']
 if c['devices']:
  argv=['docker','run','--rm','--pull','never','--name',r.name,'--network','host','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','16','--memory','16g','--pids-limit','2048','--shm-size','2g','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','ASCEND_RT_VISIBLE_DEVICES='+','.join(str(d) for d in range(len(c['devices']))),'-e','GLM53_PHYSICAL_DEVICE_IDS='+','.join(str(d) for d in c['devices']),'-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-v',str(r)+':/audit:ro']
  for d in c['devices']: argv+=['--device','/dev/davinci'+str(d)]
  for d in ['davinci_manager','devmm_svm','hisi_hdc']: argv+=['--device','/dev/'+d]
  if c['external_rank'] is not None:
   argv+=['-e','GLM53_EXTERNAL_RANK='+str(c['external_rank']),'-e','LOCAL_RANK=0','-e','GLM53_MASTER_ADDR=172.16.12.32','-e','GLM53_MASTER_PORT='+str(c['master_port'])]
  for extra in ['/usr/local/sbin/npu-smi','/usr/local/dcmi']:
   if pathlib.Path(extra).exists(): argv+=['-v',extra+':'+extra+':ro']
  (r/'artifacts').mkdir(exist_ok=True)
  argv+=['-v',str(r/'artifacts')+':/out','-e','HCCL_INTER_HCCS_DISABLE=FALSE','-e','HCCL_INTRA_ROCE_ENABLE=0','-e','HCCL_INTRA_PCIE_ENABLE=1','-e','HCCL_OP_EXPANSION_MODE=AI_CPU','-e','HCCL_EXEC_TIMEOUT=30','-e','HCCL_CONNECT_TIMEOUT=120','-e','HCCL_IF_IP=172.16.12.'+str(c['node']),'-e','ASCEND_GLOBAL_LOG_LEVEL='+str(c['log_level']),'-e','ASCEND_PROCESS_LOG_PATH=/out/ascend-log','--entrypoint','python3',image_id,'/audit/script.py']
 else: argv[argv.index(c['image'])]=image_id
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic();timed_out=False
with (r/'stdout.json').open('wb') as out,(r/'stderr.txt').open('wb') as err:
 proc=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True)
 try: rc=proc.wait(timeout=c['timeout'])
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
  try: rc=proc.wait(timeout=5)
  except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
  if c['devices']: subprocess.run(['docker','stop','-t','3',r.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
def info(name):
 b=(r/name).read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
result={'remote_root':str(r),'started_host_utc':start,'ended_host_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'duration_s':time.monotonic()-t,'argv':argv,'image_id':image_id,'returncode':rc,'timed_out':timed_out,'script_sha256':c['context']['script_sha256'],'stdout.json':info('stdout.json'),'stderr.txt':info('stderr.txt')}
raw=(r/'stdout.json').read_text(errors='replace')
try:
 if 'GLM53_RESULT_JSON_BEGIN' in raw: raw=raw.split('GLM53_RESULT_JSON_BEGIN',1)[1].split('GLM53_RESULT_JSON_END',1)[0]
 metrics=json.loads(raw);(r/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\\n');result['metrics.json']=info('metrics.json')
except ValueError: pass
artifacts=[];selected=[]
for p in sorted((r/'artifacts').rglob('*')) if (r/'artifacts').exists() else []:
 if not p.is_file() or p.is_symlink(): continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 artifacts.append({'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 if p.suffix in ['.log','.txt'] and p.stat().st_size<67108864:
  with p.open(errors='replace') as f:
   for line in f:
    if len(selected)<120 and re.search(r'sdma|transport.*(?:rdma|hccs)|intra_roce|inter_hccs|opExpansionMode',line,re.I): selected.append(str(p.relative_to(r))+': '+line[:1500])
(r/'artifacts_manifest.json').write_text(json.dumps(artifacts,indent=2)+'\\n')
(r/'sdma-log-evidence.txt').write_text(''.join(selected))
for name in ['artifacts_manifest.json','sdma-log-evidence.txt']: result[name]=info(name)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
'''.replace("PAYLOAD", repr(base64.b64encode(json.dumps(setup).encode()).decode()))
    completed = subprocess.run(ssh + [host + "python3 -"], input=runner,
                               text=True, capture_output=True, timeout=a.timeout + 40)
    a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / run
    dest.mkdir(exist_ok=False)
    (dest / "transport.json").write_text(json.dumps({"command": host + "python3 -",
        "rc": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}, indent=2) + "\n")
    if completed.returncode:
        raise RuntimeError("Remote collection failed; retained transport evidence at " + str(dest))
    result = json.loads(completed.stdout)
    for name in ("stdout.json", "stderr.txt", "context.json", "result.json", "script.py",
                 "artifacts_manifest.json", "sdma-log-evidence.txt") + (("metrics.json",) if "metrics.json" in result else ()):
        content = subprocess.check_output(ssh + [host + "cat " + shlex.quote(remote + "/" + name)], timeout=30)
        (dest / name).write_bytes(content)
        if name in result:
            assert hashlib.sha256(content).hexdigest() == result[name]["sha256"], "download mismatch"
    assert hashlib.sha256((dest / "script.py").read_bytes()).hexdigest() == sha
    print(json.dumps({"local": str(dest.resolve()), "remote": remote,
                      "rc": result["returncode"], "download_verified": True}))
    return 0 if result['returncode'] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
