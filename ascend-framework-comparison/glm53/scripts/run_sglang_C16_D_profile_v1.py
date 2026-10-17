"""One diagnostic AIS cohort; reuse frozen controller/client and add bounded D control."""
import hashlib,importlib.util,json,shlex,threading,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261017T1920-sglang-formal-C16-D-profile-v1'
SETTINGS='config/diagnostic-runs-v1/sglang-C16-D-profile.json'
SETTINGS_SHA='5a91c2804522b430827da0338825dcd6f549db90bfdd0bdaee10a4c97aac6a24'
CONTROL_SHA='bdd2b0c4562f17cbd15511eb917f568be4400e8183ad3032268c054bb39e9bb4'
CONTROL_SETTINGS_SHA='a4145cfbd3a4c036a49c944acf379dffee4bda9ae16da69d84c34b36d6fe7ae0'
def load(name,path,sha):
 p=ROOT/path;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 admission=load('admission','scripts/glm53_sglang_paired_launch_v1.py','b227e29c24f0f85e4462457db3623a421c35cd1d3a165392e91f96bfd91f077f')
 settings,controller_path=admission.validate_static(ROOT,SETTINGS,SETTINGS_SHA);admission.validate_no_overlap(ROOT,settings)
 accepted=ROOT/'results/sglang-formal-C16-v2-measurement-v1.json';assert hashlib.sha256(accepted.read_bytes()).hexdigest()=='42a7e15e69b684fe70a41e3379f0634fbc36bab05bfc7803f55bd08fd38634fb';assert json.loads(accepted.read_text())['measurement_valid'] is True
 assert settings['diagnostic_only'] and not settings['performance_certificate_allowed'] and settings['run_id']==RID
 assert settings['client_files_sha256']['glm53_sglang_D_profile_control_v1.py']==CONTROL_SHA and settings['client_files_sha256']['D-profile-settings.json']==CONTROL_SETTINGS_SHA
 run=ROOT/'runs'/RID
 with (run/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 controller=load('controller',str(controller_path.relative_to(ROOT)),'6810044962d74782b013543486eecaf3d589a1eb1a5740df43416065b1ae9de2')
 base=load('base','scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
 original_load=controller.load;threads=[]
 def profile_target():
  # Called only after the gate callback has returned and closed its SSH.
  # One existing long-client SSH plus this one profile-control SSH: max2.
  try:
   pod,_=base.NODES[43];mapped=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.43';remote='/data/wio/inference-foundry/glm53/runs/'+RID
   command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3',remote+'/glm53_sglang_D_profile_control_v1.py','--settings',remote+'/D-profile-settings.json','--sha256',CONTROL_SETTINGS_SHA,'--source-sha256',CONTROL_SHA])
   result=base.invoke([command],timeout=500)
  except BaseException as error:result={'error':repr(error)}
  (run/'D-profile-control-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 def loader(name,path,sha):
  module=original_load(name,path,sha)
  if name!='formal_rpc':return module
  original_rpc=module.rpc
  def rpc(command,root,expected,on_gate):
   def gate(request):
    report=on_gate(request)
    if report['pass']:
     assert not threads
     thread=threading.Thread(target=profile_target,daemon=False);threads.append(thread);thread.start()
    return report
   return original_rpc(command,root,expected,gate)
  return types.SimpleNamespace(rpc=rpc)
 controller.load=loader
 try:return controller.execute(settings)
 finally:
  for thread in threads:thread.join(timeout=520)
  assert not any(t.is_alive() for t in threads),'Profile control not closed; preserve remote work, no extra SSH capture'
if __name__=='__main__':main()
