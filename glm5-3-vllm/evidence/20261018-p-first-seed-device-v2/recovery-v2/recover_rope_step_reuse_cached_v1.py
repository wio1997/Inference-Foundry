"""Fetch only eight already frozen files; never rerun a completed NPU request."""
from pathlib import Path
import hashlib,io,json,shlex,subprocess,tarfile
r=Path(__file__).resolve().parent
dest=next((r/'rope-step-reuse-cached-v1').glob('*/transport.json')).parent
t=json.loads((dest/'transport.json').read_text());assert t['rc']==0
result=json.loads(t['stdout']);assert result['returncode']==0 and not result['timed_out']
remote=result['remote_root'];assert remote.endswith('/node33-rope_step_reuse_cached_v1-47f8006e6a81')
names=['stdout.json','stderr.txt','context.json','result.json','script.py','artifacts_manifest.json','sdma-log-evidence.txt','metrics.json']
ssh=['ssh','-S','/tmp/glm53-active-20261019.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
command='kubectl --request-timeout=20s exec -i -n default sglang-tuning-access-33-20261013 -- nsenter -t 1 -m -- tar -czf - -C '+shlex.quote(remote)+' '+' '.join(names)
packed=subprocess.check_output(ssh+[command],timeout=60)
with tarfile.open(fileobj=io.BytesIO(packed),mode='r:gz') as tar:
 members=tar.getmembers();assert sorted(m.name for m in members)==sorted(names) and all(m.isfile() for m in members)
 data={m.name:tar.extractfile(m).read() for m in members}
assert json.loads(data['result.json'])==result
assert hashlib.sha256(data['script.py']).hexdigest()==result['script_sha256']
for name,content in data.items():
 if name in result:
  assert len(content)==result[name]['bytes'] and hashlib.sha256(content).hexdigest()==result[name]['sha256']
 p=dest/name
 if p.exists():assert p.read_bytes()==content
 else:p.write_bytes(content)
metrics=json.loads(data['metrics.json']);assert metrics['completed'] and metrics['numeric_all_pass']
backup=Path('/Users/wio/work/glm53-execution-evidence-backup/20261019-rope-step-reuse-cached-recovery');backup.mkdir(exist_ok=True)
archive=backup/'frozen-eight-files.tar.gz'
if archive.exists():assert archive.read_bytes()==packed
else:archive.write_bytes(packed)
receipt=dict(completed=True,original_execution_duration_s=result['duration_s'],original_execution_returncode=0,original_requests_repeated=0,archive=str(archive),archive_bytes=len(packed),archive_sha256=hashlib.sha256(packed).hexdigest(),members=[dict(name=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in data.items()],scope='Read-only recovery of the original completed cached run after SSH result-download failure. All original known bytes/SHA, script and result envelope verified. No NPU work or marker mutation.')
(r/'rope-step-reuse-cached-download-recovery-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(recovered=True,numeric_all=True,original_requests_repeated=0)))
