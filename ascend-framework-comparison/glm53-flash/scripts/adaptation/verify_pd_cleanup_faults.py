# SPDX-License-Identifier: Apache-2.0
"""Fault-inject a frozen host finally block without any resource commands."""
import ast,pathlib,json,tempfile,types,contextlib,io,hashlib,tarfile,time,argparse
p=argparse.ArgumentParser();p.add_argument('--host-script',type=pathlib.Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();source=a.host_script.read_text();tree=ast.parse(source);outer=next(n for n in tree.body if isinstance(n,ast.Try));block=ast.fix_missing_locations(ast.Module(body=outer.finalbody,type_ignores=[]));compiled=compile(block,'frozen-host-finally-fixture','exec')
results=[]
for mode in ['invalid_result_json','invalid_result_and_archive_failure','identity_mismatch','normal_result']:
 with tempfile.TemporaryDirectory(prefix='flash-cleanup-fixture-') as td:
  R=pathlib.Path(td);calls=[];out={'project_id':'glm53-flash'}
  (R/'request-0-0.json').write_text(json.dumps({'attempted':True,'fixture_only':True}) if mode=='normal_result' else '{')
  def run(args,t=30):
   calls.append(list(args))
   if args[:2]==['docker','inspect']:
    text='false 0\n' if '--format' in args else json.dumps([{'Image':'fixture-image','Config':{'Labels':{'project_id':'another-project' if mode=='identity_mismatch' else 'glm53-flash','run_id':a.run_id}}}])
   elif args[:2]==['docker','stop']:text='fixture-own-container\n'
   elif args==['npu-smi','info']:text='No running processes found\n'*8
   else:raise AssertionError('Unexpected mocked command: '+str(args))
   return {'rc':0,'stdout':text,'stderr':''}
  def logs(args,**kwargs):
   assert args[:2]==['docker','logs'];calls.append(list(args));kwargs['stdout'].write(b'LOCAL_FIXTURE_NOT_RESOURCE_EVIDENCE\n');return types.SimpleNamespace(returncode=0)
  def archive_open(*args,**kwargs):
   if mode=='invalid_result_and_archive_failure':raise RuntimeError('injected archive failure')
   return tarfile.open(*args,**kwargs)
  def save(): (R/'host-status.json').write_text(json.dumps(out))
  env={'R':R,'node':46,'created':True,'name':'fixture-own-container','image':'fixture-image','out':out,'run':run,'save':save,'json':json,'time':time,'subprocess':types.SimpleNamespace(run=logs,STDOUT=-2),'hashlib':hashlib,'tarfile':types.SimpleNamespace(open=archive_open)}
  with contextlib.redirect_stdout(io.StringIO()):exec(compiled,env)
  stops=[v for v in calls if v[:2]==['docker','stop']]
  if mode=='identity_mismatch':assert not stops and out.get('cleanup_error') and 'post_inspect' not in out
  else:
   assert len(stops)==1 and out['post_inspect']['stdout'].strip()=='false 0' and out['post_npu']['stdout'].count('No running processes found')==8
   assert next(i for i,v in enumerate(calls) if v[:2]==['docker','stop'])<next(i for i,v in enumerate(calls) if v[:2]==['docker','logs'])
  if mode.startswith('invalid_result'):assert out['result_collection_errors']
  if mode=='invalid_result_and_archive_failure':assert out['artifact_collection_error']=="RuntimeError('injected archive failure')"
  if mode=='normal_result':assert out['generation_requests']==1 and not out.get('result_collection_errors')
  results.append({'case':mode,'passed':True,'mock_stop_calls':len(stops),'actual_resource_commands':0})
report={'project_id':'glm53-flash','verdict':'PASS_LOCAL_HOST_FINALLY_FAULT_INJECTION_NO_RESOURCES','host_script_sha256':hashlib.sha256(source.encode()).hexdigest(),'cases':results,'actual_Docker_NPU_model_calls':0,'limitations':'Onlyorchestrationcontrol-flowfixturewithmockedcommandfunctions. Notruntimecontainercleanup/SDK/modelqualification.'};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
