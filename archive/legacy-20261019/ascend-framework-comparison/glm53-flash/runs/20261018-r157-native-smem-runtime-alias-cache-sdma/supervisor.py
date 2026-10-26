# SPDX-License-Identifier: Apache-2.0
import pathlib,json,subprocess,os,time,signal,sys,traceback,socket,hashlib
R=pathlib.Path('/run-evidence');node=int(os.environ['FLASH_COMPONENT_NODE']);out={'project_id':'glm53-flash','node':node,'passed':False,'generation_requests':0};children=[]
def stop(p):
 if p.poll() is None:
  try:os.killpg(p.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   p.wait(timeout=5)
 return p.returncode
try:
 if node==14:
  v=json.loads((R/'meta-ready.json').read_text());assert v['project_id']=='glm53-flash' and v['node']==14 and v['CPU_meta_no_NPU_devices'] is True
 else:
  deadline=time.monotonic()+180
  while not (R/'prefill-engine-permit.json').is_file():
   if (R/'abort-component.json').is_file():raise RuntimeError('P failed before SDK ready')
   if time.monotonic()>deadline:raise TimeoutError('Prefill SDK control engine not ready')
   time.sleep(.5)
  permit=json.loads((R/'prefill-engine-permit.json').read_text());assert permit['project_id']=='glm53-flash' and permit['node']==14 and permit['protocol']=='TransDataOpType.SDMA'
 # MMC1.2 runtime C-ABI explicitly dlopens this unhashed basename. The
 # original native extension loaded the same existing auditwheel DSO already.
 # A private symlink gives its alternate name to that SAME inode; no SDK edits.
 canonical=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid.libs/libmf_smem-2154e855.so');assert hashlib.sha256(canonical.read_bytes()).hexdigest()=='aaf716211fcddc1a586908a90a5015388f5f4822421d516c40fdcb6a36b2c6d6'
 aliasdir=R/'native-library-aliases';aliasdir.mkdir(exist_ok=False);alias=aliasdir/'libmf_smem.so';alias.symlink_to(canonical);assert alias.resolve()==canonical.resolve() and alias.stat().st_ino==canonical.stat().st_ino
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(aliasdir)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
 out['private_smem_runtime_alias']={'path':str(alias),'canonical':str(canonical),'same_inode':True,'canonical_SHA256':'aaf716211fcddc1a586908a90a5015388f5f4822421d516c40fdcb6a36b2c6d6','SDK_files_modified':False,'worker_env_only':True}
 with (R/'worker.log').open('wb') as log:
  worker=subprocess.Popen([sys.executable,str(R/'worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env);children.append(worker)
  try:out['worker_rc']=worker.wait(timeout=330)
  except subprocess.TimeoutExpired:out['worker_timed_out']=True;out['worker_rc']=stop(worker)
 if (R/'worker-result.json').is_file():out['worker']=json.loads((R/'worker-result.json').read_text())
 out['passed']=out.get('worker_rc')==0 and out.get('worker',{}).get('passed') is True
except BaseException as exc:out.update(error=repr(exc),traceback=traceback.format_exc())
finally:
 for child in reversed(children):stop(child)
 for name in ['worker.log']:
  if (R/name).exists():out[name+'_tail']=(R/name).read_text(errors='replace')[-16000:]
 (R/'supervisor-result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
sys.exit(0 if out['passed'] else 1)
