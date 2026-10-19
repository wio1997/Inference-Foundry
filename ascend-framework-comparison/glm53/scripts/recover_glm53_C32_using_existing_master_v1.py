"""Verify existing authorized SSH master and recover exactowned interrupted run once."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
CONTROL='/Users/wio/work/sglang-glm53-pd/ssh-control'
RUN=ROOT/'runs/20261018T-L09-existing-master-recovery-v1'
CHILD='scripts/inspect_sglang_C32_L09_interrupted_control_recovery_v2.py'
SHA='5f71f518fe62a5ae8667f37e1c54ab2193e0708e43d47376905ed36ddb0119aa'
def main():
 assert Path(CONTROL).is_socket() and Path(CONTROL).stat().st_uid==os.getuid()
 assert hashlib.sha256((ROOT/CHILD).read_bytes()).hexdigest()==SHA
 assert not (ROOT/'runs/20261018T-L09-interrupted-control-recovery-v2').exists(),'No child replay'
 RUN.mkdir(exist_ok=False);(RUN/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'remote_mutations':False}
 try:
  r=subprocess.run(['ssh','-S',CONTROL,'-O','check','-p','60011','root@61.241.77.34'],capture_output=True,text=True,timeout=15);receipt['master_check']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr};assert r.returncode==0
  p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777';spec=importlib.util.spec_from_file_location('inventory',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  r=m.invoke(['hostname'],timeout=30);receipt['source_identity']=r;assert r['rc']==0 and r['stdout'].strip()=='A3-5U-Atlas900RCKA3-1'
  print('EXISTING_MASTER_SOURCE_IDENTITY_VERIFIED',flush=True)
  spec=importlib.util.spec_from_file_location('recovery',ROOT/CHILD);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.main();receipt['recovery_child_returned']=True
 except BaseException as e:receipt['error']=repr(e);raise
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
