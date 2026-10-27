"""Verify copied auditor bytes, then synthetic cases and optional saved inputs."""
from pathlib import Path,PurePosixPath
import argparse,ast,hashlib,json,os,platform,shutil,subprocess,sys,time
C=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def relative_path(raw):
 p=PurePosixPath(raw)
 if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('unsafe relative path')
 return Path(*p.parts)
def file_at(root,rel):
 p=root/relative_path(rel)
 if not p.is_file() or p.is_symlink() or any(q.is_symlink()for q in p.parents if q!=root and root in q.parents):raise ValueError('missing/symlink input: '+str(rel))
 if not p.resolve().is_relative_to(root.resolve()):raise ValueError('escaped root')
 return p
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--evidence-root',type=Path);a=ap.parse_args()
 out=a.output.absolute()
 if out.exists()or out==C or C in out.parents or out in C.parents:raise ValueError('output must be absent and outside component')
 manifest=read(C/'code-manifest.json');verified=[]
 for rel,h in manifest['files'].items():
  p=file_at(C,rel);assert sha(p)==h,('component SHA mismatch',rel)
  if p.suffix=='.py':ast.parse(p.read_text())
  verified.append({'path':rel,'sha256':h})
 expected=read(C/'expected-results.json');pinned=read(C/'pinned-inputs.json');bindings=read(C/'source-bindings.json')
 # Reject source and output overlap before creating output or executing anything.
 root=a.evidence_root.resolve()if a.evidence_root else None
 if root is not None:
  if not root.is_dir()or root==out or out in root.parents:raise ValueError('invalid evidence root/output overlap')
  for row in pinned['files']:
   p=file_at(root,row['path']);assert sha(p)==row['sha256']and p.stat().st_size==row['bytes'],('input SHA/size mismatch',row['path'])
 out.mkdir(parents=True,exist_ok=False);stage=out/'evidence';scripts=stage/'executor-run-gen35';scripts.mkdir(parents=True)
 for row in bindings['files']:
  p=file_at(C,row['path']);assert sha(p)==row['sha256'];shutil.copyfile(p,scripts/p.name)
 outputs=[]
 def run(name,code):
  env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env['PYTHONDONTWRITEBYTECODE']='1'
  cmd=[sys.executable,'-I','-B','-S','-c',code];v=subprocess.run(cmd,cwd=stage,env=env,capture_output=True,text=True,timeout=60)
  log=out/(name+'.log');log.write_text(v.stdout+v.stderr)
  outputs.append({'name':name,'returncode':v.returncode,'command':cmd,'log':log.name,'log_sha256':sha(log)})
  assert v.returncode==0,(name,v.stderr[-1500:])
 run('identity-tests','import sys,unittest;sys.path.insert(0,'+repr(str(scripts))+');s=unittest.defaultTestLoader.discover('+repr(str(scripts))+',pattern="test_sg_output_lineage.py");assert s.countTestCases()==14;r=unittest.TextTestRunner(verbosity=2).run(s);sys.exit(not r.wasSuccessful())')
 equal=False
 if root is not None:
  for row in pinned['files']:
   p=file_at(root,row['path']);q=stage/relative_path(row['path']);q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();shutil.copyfile(p,q);assert sha(q)==row['sha256']
  for name in ['audit_source_contract','audit_sg_output_lineage']:
   run(name,'import runpy;runpy.run_path('+repr(str(scripts/(name+'.py')))+',run_name="__main__")')
  assert sha(scripts/'source-contract.json')==expected['source_contract_sha256']
  actual=read(scripts/'sg-output-lineage-audit.json');actual.pop('verified_ns')
  assert hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==expected['lineage_JSON_without_verified_ns_sha256']
  assert sha(scripts/'sg-output-token-ledger.csv')==expected['ledger_csv_sha256']and sha(scripts/'sg-output-token-ledger.json')==expected['ledger_json_sha256']
  # An output tree and replay must never alter the evidence source.
  for row in pinned['files']:assert sha(file_at(root,row['path']))==row['sha256']
  equal=True
 result={'task_id':'Q27-FULL-MAP-20261025','generation':37,'status':'CPU_checks_complete','verified_ns':time.time_ns(),'CPU_Host_Architecture':platform.machine(),'tests':14,'component_files_SHA_verified':len(verified),'saved_input_files_SHA_verified':len(pinned['files'])if root is not None else 0,'saved_lineage_JSON_equal_except_verified_ns':equal,'saved_source_contract_and_ledgers_exact':equal,'commands':outputs,'NPU_experiment_executed':False,'service_started':False,'new_native_trace_audit':False,'quality_equivalence_proven':False,'optimization_adopted':False,'native204_accepted':False,'root_acceptance':'pending','overall_complete':False}
 write(out/'validation.json',result);print(json.dumps({k:v for k,v in result.items()if k!='commands'}))
if __name__=='__main__':main()
