# SPDX-License-Identifier: Apache-2.0
import pathlib,json,hashlib,base64,subprocess,shlex,os,time,datetime,concurrent.futures,traceback
R=pathlib.Path(__file__).resolve().parent;review=json.loads((R/'review.json').read_text())
for f,h in review['files_sha256'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f
pr=json.loads((R/'prerequisites.json').read_text());P=R.parents[1]
for f,h in pr['files_sha256'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h,f
assert json.loads((P/'runs/20261018-r152-native-meta-direct-bm-binding-cpu/acceptance.json').read_text())['passed'];assert json.loads((P/'runs/20261018-r155-native-direct-bind-cache-sdma/failure-localization.json').read_text())['stopped_idle8_both']
os.close(os.open(R/'once.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
ssh=['ssh','-S','/Users/wio/work/sglang-glm53-pd/ssh-control','-p','60011','-o','BatchMode=yes','-o','ConnectTimeout=15','root@61.241.77.34'];pods={14:'zhixing-dockerctl-25xz2',15:'zhixing-dockerctl-42pbf'};root='/data/wio/inference-foundry/glm53-flash/runs/20261018-r157-native-smem-runtime-alias-cache-sdma'
out={'project_id':'glm53-flash','workstream':'A','started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'generation_requests':0,'formal_requests':0};pool=concurrent.futures.ThreadPoolExecutor(max_workers=2);futures={}
def command(node,code,timeout=40):
 cmd=['kubectl','exec','-i','-n','default',pods[node],'-c','hostctl','--','nsenter','-t1','-m','--','python3','-'];q=subprocess.run(ssh+[shlex.join(cmd)],input=code,capture_output=True,text=True,timeout=timeout);return {'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
def save():(R/'progress.json').write_text(json.dumps(out,indent=2))
def read(node,name):
 assert name in ('PD-engine-ready.json','producer-ready.json')
 return command(node,'import pathlib;print(pathlib.Path('+repr(root+'/'+name)+').read_text())',20)
def write(node,entries):
 allowed={'prefill-engine-permit.json','producer-peer.json','sdma-peer-map.json','consumer-done.json','abort-component.json'};assert set(entries)<=allowed
 code='import pathlib,os,json,socket\nassert socket.gethostname()=='+repr({14:'A6-15U-Atlas900RCKA3-14',15:'A6-23U-Atlas900RCKA3-15'}[node])+'\nr=pathlib.Path('+repr(root)+');assert (r/"once.lock").is_file()\nentries='+repr(entries)+'\nfor name,value in entries.items():\n p=r/name;fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,json.dumps(value).encode());os.close(fd)\n'
 q=command(node,code,30);assert q['rc']==0,q;return q
try:
 for node in pods:
  q=subprocess.run(ssh+[shlex.join(['kubectl','get','pod','-n','default',pods[node],'-o','jsonpath={.status.hostIP}'])],capture_output=True,text=True,timeout=25);assert q.returncode==0 and q.stdout.strip()=='172.16.12.'+str(node)
  names={'flash_native_sdk_bindings.py':'flash_native_sdk_bindings.py','meta_child.py':'meta_child.py','worker.py':'worker.py','supervisor.py':'supervisor.py','expected.json':'expected.json','vllm_flash_sdma_performance_engine.py':'vllm_flash_sdma_performance_engine.py','sdma_payload_hash.py':'sdma_payload_hash.py','mmc-meta.conf':'mmc-meta.conf','mmc-local.conf':'mmc-local-P.conf' if node==14 else 'mmc-local-D.conf'}
  files={target:{'data':base64.b64encode((R/f).read_bytes()).decode(),'sha256':review['files_sha256'][f]} for target,f in names.items()};code=(R/'host_probe.py').read_text().replace('__NODE__',str(node)).replace('__FILES__',repr(files));futures[node]=pool.submit(command,node,code,650)
 permit=peer=release=False;deadline=time.monotonic()+600
 while not all(f.done() for f in futures.values()):
  if time.monotonic()>deadline:raise TimeoutError('Bounded component orchestration deadline')
  if not permit:
   q=read(14,'PD-engine-ready.json')
   if q['rc']==0:
    v=json.loads(q['stdout']);assert v['project_id']=='glm53-flash' and v['node']==14 and v['protocol']=='TransDataOpType.SDMA';write(15,{'prefill-engine-permit.json':v});permit=True;out['prefill_control_engine_ready']=v;save()
  if permit and not peer:
   q=read(14,'producer-ready.json')
   if q['rc']==0:
    v=json.loads(q['stdout']);assert v['project_id']=='glm53-flash' and v['node']==14 and v['protocol']=='TransDataOpType.SDMA' and v['session_id']==out['prefill_control_engine_ready']['session_id'];mapping={v['session_id']:{'global_device_ordinal':0,'url':'http://172.16.12.14:18500/hash'}}
    # Publish metadata permit LAST: D may immediately submit its single read.
    write(15,{'sdma-peer-map.json':mapping,'producer-peer.json':v});peer=True;out['producer_cache_restored_ready']=v;save()
  if futures[15].done() and not release:
   q=futures[15].result();(R/'node15.json').write_text(json.dumps(q,indent=2));v=json.loads(q['stdout']) if q['rc']==0 else {};stopped=v.get('post_inspect',{}).get('stdout','').strip()=='false 0';idle=v.get('post_npu',{}).get('stdout','').count('No running processes found')==8;passed=stopped and idle and v.get('supervisor-result.json',{}).get('passed') is True and v.get('worker-result.json',{}).get('passed') is True
   write(14,{'consumer-done.json':{'project_id':'glm53-flash','consumer_stopped_idle8':stopped and idle,'consumer_passed':passed}});release=True;out['consumer_terminal']={'stopped_exit0':stopped,'idle8':idle,'passed':passed};save()
  if futures[14].done() and not futures[15].done():
   # If P failed before publication, do not leave D waiting for an absent peer.
   q=futures[14].result();(R/'node14.json').write_text(json.dumps(q,indent=2));v=json.loads(q['stdout']) if q['rc']==0 else {}
   if not v.get('supervisor-result.json',{}).get('passed'):
    write(15,{'abort-component.json':{'project_id':'glm53-flash','reason':'P component terminal failure; no further payload permitted'}});out['P_failed_before_D_terminal']=True;save();break
  time.sleep(3)
except BaseException as exc:out.update(error=repr(exc),traceback=traceback.format_exc())
finally:
 for node,f in futures.items():
  try:q=f.result(timeout=660);(R/f'node{node}.json').write_text(json.dumps(q,indent=2));out[f'node{node}_ssh_rc']=q['rc']
  except BaseException as exc:out[f'node{node}_collection_error']=repr(exc)
 pool.shutdown(wait=True);out['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat();(R/'receipt.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
