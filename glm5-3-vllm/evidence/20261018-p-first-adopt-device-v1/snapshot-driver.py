from pathlib import Path
import concurrent.futures,json,subprocess,tarfile
root=Path('/Users/wio/work/Inference-Foundry-execution-plan'); e=root/'glm5-3-vllm/evidence/20261018-p-first-adopt-device-v1'; backup=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-adopt-device-v1')
control='/tmp/glm53-active-20261018.sock'
def snapshot(n):
 run='/data/wio/glm5-3-vllm-execution/20261018-state-tail/'+('tail-c1-live1-20261018/node32' if n==32 else 'adopt-live1-20261018/node33')
 c=['python3',str(root/'glm5-3-vllm/tools/collect_node.py'),'--node',str(n),'--script',str(root/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/finish.py'),'--label','adopt_live1_snapshot','--control',control,'--output',str(e/f'snapshot{n}'),'--timeout','150','--script-arg=--run-dir','--script-arg='+run,'--script-arg=--reason','--script-arg=stage-complete','--script-arg=--keep-running']
 r=subprocess.run(c,cwd=root,capture_output=True,text=True,timeout=205);(e/f'snapshot{n}-collector.txt').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 local=next((e/f'snapshot{n}').glob('*/result.json')).parent
 r=subprocess.run(['python3',str(root/'glm5-3-vllm/tools/backup_artifacts.py'),'--run',str(local),'--control',control,'--backup-root',str(backup)],cwd=root,capture_output=True,text=True,timeout=400);(e/f'snapshot{n}-backup.txt').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 receipt=json.loads((local/'artifacts-backup.json').read_text());dest=backup/f'node{n}'
 assert not dest.exists();dest.mkdir(parents=True)
 with tarfile.open(receipt['local_archive'],'r:gz') as tf:
  for member in tf.getmembers():
   p=Path(member.name);assert not p.is_absolute() and '..' not in p.parts and not (member.issym() or member.islnk())
  tf.extractall(dest)
 return {'node':n,'model_stopped':False,'backup':receipt,'probe_dir':str(dest/'artifacts/state-tail-probe')}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(snapshot,(32,33)))
(e/'snapshot-backups.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows),flush=True)
