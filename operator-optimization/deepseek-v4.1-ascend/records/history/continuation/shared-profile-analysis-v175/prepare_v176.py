"""Reduce startup perturbation,retain exact diagnostic log,one new acquisition."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent;D=H.parent/'shared-profile-analysis-v176';(D/'hook').mkdir(parents=True,exist_ok=True)
for name in ['controller.py','client.py','hook/sitecustomize.py','hook/joint_runtime.py','serve-enabled.sh','stage_and_launch.py']:
 s=(H/name).read_text().replace('175','176')
 if name=='hook/joint_runtime.py':
  s=s.replace('import contextlib,','import traceback,contextlib,')
  s=s.replace('result=old_init(obj,*a,**k);resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3));selftest_npu(obj)','result=old_init(obj,*a,**k);state().emit("predecessor_NPU_interface_selftest_reused",cohort="joint175",devices_passed=16,no_new_NPU_selftest=True)')
  s=s.replace('cls.init_device=init_device;cls.profile=profile', '''cls.init_device=init_device;cls.profile=profile
  old_load=cls.load_model
  @functools.wraps(old_load)
  def load(obj,*a,**k):
   state().emit('worker_load_model_enter')
   try:
    result=old_load(obj,*a,**k);state().emit('worker_load_model_return');return result
   except BaseException:
    state().emit('worker_load_model_failure',traceback=traceback.format_exc());raise
  cls.load_model=load''')
  s=s.replace("if not self.running():yield;return", "if not self.running():yield;return\n  if self.seq%256==0:self.check_budget()")
  s=s.replace('A tiny16element NPU selftest runs once per\nworker before loading.','The16device interface selftest passed in v175; this run performs no new selftest.\nNo process-wide resource limit is changed.')
 if name=='serve-enabled.sh':
  s=s.replace('exec bash "$WIO_META_ROOT/serve-original.sh"','exec bash "$WIO_META_ROOT/serve-original.sh" >>"$WIO_META_ROOT/diagnostic-service.log" 2>&1')
 if name=='controller.py':
  s=s.replace("if any(not x.get('npu_selftest_done') or not x.get('catalogued') or not x.get('execution_trace_bytes') for x in finals)","if any(not x.get('catalogued') or not x.get('execution_trace_bytes') for x in finals)")
  s=s.replace("WorkerET/selftest/catalogue incomplete","WorkerET/catalogue incomplete")
  s=s.replace("Tiny16elementselftestperworker beforeload.","Noadditionaldevice selftest;reusedv175all16NPUinterfacePASS.No processwidefilelimit.")
  s=s.replace("failure=traceback.format_exc();state['acquisition_failure']=failure;", "failure=traceback.format_exc();state['acquisition_failure']=failure;f=ROOT/'diagnostic-service.log';state['diagnostic_log_bytes']=f.stat().st_size if f.exists() else None;")
 for f in [D/name]:
  f.parent.mkdir(exist_ok=True,parents=True);f.write_text(s)
plan=json.loads((H/'PLAN.json').read_text());plan['cohort']='joint176';plan['purpose']='One combined native ET and bank/scalar catalogue;reduced startup perturbation after v175 startup failure.'
plan['predecessor_acquisition']={'status':'FAILED_BEFORE_FULL_CASE_ORIGINAL_SERVICE_RESTORED_HTTP200','receipt':str(H/'restoration-observation-004.json'),'failure_cause':'Unresolved because startup log overwritten by original restoration;no causal assertion.','passed_NPU_interface_selftests':16,'not_repeated':True}
plan['process_file_limit']=None;plan['diagnostic_log']='independent diagnostic-service.log persists through restoration';plan.pop('sources',None)
(D/'PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
for f in D.rglob('*.py'):compile(f.read_text(),str(f),'exec')
print(json.dumps({'prepared':str(D),'compiled':len(list(D.rglob('*.py'))),'no_launch_yet':True}))
