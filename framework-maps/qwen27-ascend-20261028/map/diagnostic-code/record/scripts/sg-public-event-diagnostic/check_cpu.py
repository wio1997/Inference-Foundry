"""CPU fixtures only; isolated children omit sitecustomize and third-party paths."""
from pathlib import Path
import argparse, ast, hashlib, json, platform, re, subprocess, sys, time

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parent;bindings=json.loads((root/'source-bindings.json').read_text())
 for x in bindings['files']:
  b=(root/x['destination']).read_bytes()
  assert len(b)==x['bytes']and hashlib.sha256(b).hexdigest()==x['sha256'],x['destination']
 compiled=[]
 for path in sorted(root.rglob('*.py')):
  compile(path.read_bytes(),str(path),'exec');compiled.append(str(path.relative_to(root)))
 graph={}
 for name,expected in bindings['runtime_dependency_graph'].items():
  t=ast.parse((root/'runtime'/f'{name}.py').read_text());imports=set()
  for n in ast.walk(t):
   if isinstance(n,ast.Import):imports.update(x.name.split('.')[0]for x in n.names)
   if isinstance(n,ast.ImportFrom)and n.module:imports.add(n.module.split('.')[0])
  actual=sorted(imports&bindings['runtime_dependency_graph'].keys());assert actual==expected
  graph[name]=actual
 suites=[('runtime',['test_sg_layer_event_anchor','test_sg_replay_outer_events']),('runtime-cleanup',['test_gen67_cleanup']),('selection',['test_formal_selection']),('raw-audit',['test_audit_public_model_markers']),('native-audit',['test_sg67_auditor'])]
 result=[];a.output.mkdir(parents=True,exist_ok=False)
 for directory,names in suites:
  # -I/-S prevent implicit project hooks/sitecustomize; add only selected source dirs.
  code='import sys,unittest;sys.path.insert(0,'+repr(str(root/('runtime'if directory=='runtime-cleanup'else directory)))+');s=unittest.defaultTestLoader.loadTestsFromNames('+repr(names)+');r=unittest.TextTestRunner(verbosity=2).run(s);sys.exit(not r.wasSuccessful())'
  if directory=='runtime':
   # Old gen44 capture fixtures predate auto-cleanup; isolate that legacy seam.
   code=code.replace('import sys,unittest;', 'import sys,unittest;from unittest.mock import patch;')
   code=code.replace(';s=unittest.', ';import sg_layer_event_anchor as anchor;patcher=patch.object(anchor,"arm_owned_graph_cleanup");patcher.start();s=unittest.')
  command=[sys.executable,'-I','-B','-S','-c',code]
  s=subprocess.run(command,cwd=a.output,text=True,capture_output=True,timeout=120)
  log=s.stdout+s.stderr;(a.output/(directory+'.log')).write_text(log)
  match=re.search(r'Ran (\d+) tests?',log)
  result.append({'suite':directory,'modules':names,'returncode':s.returncode,'test_count':int(match[1])if match else None})
  assert s.returncode==0,log
 static=subprocess.run([sys.executable,'-I','-B','-S',str(root/'historical-build/test_driver_static.py')],text=True,capture_output=True,timeout=30)
 (a.output/'driver-static.log').write_text(static.stdout+static.stderr);assert static.returncode==0,static.stderr
 receipt={'task_id':'Q27-FULL-MAP-20261025','generation':31,'status':'CPU_source_bindings_and_fixtures_passed',
  'observed_ns':time.time_ns(),'local_Host_Architecture':platform.machine(),'local_python':sys.version,
  'source_files_rehashed':len(bindings['files']),'Python_files_compiled_without_import':compiled,
  'static_runtime_local_imports_closed':len(graph),'legacy_capture_fixture_seam':'arm_owned_graph_cleanup stubbed for oldgen44 capture tests only; separate gen67cleanup suite executes real function synchronously on stubs','suites':result,'tests':sum(x['test_count']for x in result),'historical_build_driver_static_check_returncode':static.returncode,
  'scope':'CPU stubs/synthetic identity fixtures, not C++ rebuild/NPU/service/output equivalence/performance or actual204 acceptance.',
  'overall_complete':False}
 (a.output/'validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
