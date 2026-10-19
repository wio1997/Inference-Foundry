"""Pure-Python fresh-interpreter/spawn tests, using an AST-only synthetic package."""
import __future__,argparse,ast,hashlib,importlib.machinery,json,multiprocessing,os,subprocess,sys,types,typing
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE_SHA='48fbc94dd60c7c37e198935717357d2479150099487ee0d9a749ed41083a7f84'
REVISION='glm53-routing-host-metadata-batch-v1'

class SyntheticNormalLoader(importlib.machinery.SourceFileLoader):
 def exec_module(self,module):
  if Path(self.path).name!='normal_strategy.py':return super().exec_module(module)
  # The real package imports are deliberately never executed in this fixture.
  source=Path(self.path).read_text();cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='AlltoAllNormalCommStrategy');cls.decorator_list=[]
  module.__dict__.update({k:v for k,v in vars(typing).items() if not k.startswith('__')});module.__dict__.update(torch=types.SimpleNamespace(Tensor=type('SyntheticTensor',(),{})),EventOverlap=type('Event',(),{}),NormalEPCommStrategy=object)
  tree=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]));exec(compile(tree,self.path,'exec',flags=__future__.annotations.compiler_flag),module.__dict__)

def probe():
 package_root=ROOT/'synthetic_packages';strategy_dir=package_root/'deep_ep/strategies'
 def path_hook(path):
  if Path(path).resolve()==strategy_dir.resolve():return importlib.machinery.FileFinder(path,(SyntheticNormalLoader,importlib.machinery.SOURCE_SUFFIXES))
  raise ImportError
 sys.path_hooks.insert(0,path_hook);sys.path_importer_cache.pop(str(strategy_dir),None);sys.path.insert(0,str(package_root))
 import deep_ep.strategies.normal_strategy as module
 cls=module.AlltoAllNormalCommStrategy;enabled=os.environ.get('GLM53_DEEPEP_METADATA_BATCH')=='1'
 assert bool(getattr(cls,'_glm53_metadata_batch_installed',False))==enabled
 assert (REVISION in cls.get_dispatch_layout.__code__.co_filename)==enabled
 packed=os.environ.get('GLM53_PACKED_DISPATCH')=='1'
 assert bool(getattr(cls,'_glm53_packed_dispatch_installed',False))==packed
 assert ('glm53-C32-normal-packed-dispatch-v1' in cls.dispatch.__code__.co_filename)==packed
 if enabled and not packed:assert REVISION in cls.dispatch.__code__.co_filename
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==SOURCE_SHA
 assert not any(x.startswith(('torch_npu','memcache_hybrid','memfabric_hybrid','deep_ep_cpp')) for x in sys.modules)
 return {'pid':os.getpid(),'node':os.environ.get('GLM53_NODE'),'installed':enabled,'packed':packed,'synthetic_source_only':True,'no_real_framework_or_SDK_imports':True}

def worker(queue):
 try:queue.put({'pass':True,'result':probe()})
 except BaseException as error:queue.put({'pass':False,'error':repr(error)});raise

def child():
 parent=probe();context=multiprocessing.get_context('spawn');queue=context.Queue();processes=[context.Process(target=worker,args=(queue,)) for _ in range(2)]
 for process in processes:process.start()
 values=[queue.get(timeout=20) for _ in processes]
 for process in processes:process.join(timeout=20);assert process.exitcode==0
 assert all(x['pass'] for x in values);results=[parent,*[x['result'] for x in values]];assert len({x['pid'] for x in results})==3
 print('GLM53_HOOK_CHILD='+json.dumps({'pass':True,'processes':results}),flush=True)

def main():
 package=ROOT/'synthetic_packages/deep_ep/strategies';package.mkdir(parents=True);(package/'__init__.py').write_text('');(package.parent/'__init__.py').write_text('')
 original=(ROOT/'normal_strategy.py').read_bytes();assert hashlib.sha256(original).hexdigest()==SOURCE_SHA;target=package/'normal_strategy.py';target.write_bytes(original)
 cases=[]
 for label,node,enabled,expected_rc in [('P_noop','34',False,0),('D36_spawn','36',True,0),('D41_spawn','41',True,0),('legacy_D_L04','36',True,0),('pack_without_metadata','36',False,86),('wrong_P_role','35',True,86),('changed_source','36',True,1)]:
  target.write_bytes(original+(b'\n# fixture corruption\n' if label=='changed_source' else b''))
  environment=os.environ.copy();environment['PYTHONPATH']=str(ROOT);environment['GLM53_NODE']=node
  if enabled:environment['GLM53_DEEPEP_METADATA_BATCH']='1'
  else:environment.pop('GLM53_DEEPEP_METADATA_BATCH',None)
  if label not in ['P_noop','legacy_D_L04']:environment['GLM53_PACKED_DISPATCH']='1'
  else:environment.pop('GLM53_PACKED_DISPATCH',None)
  result=subprocess.run([sys.executable,str(Path(__file__)),'--child'],capture_output=True,text=True,env=environment,timeout=60)
  assert result.returncode==expected_rc,(label,result.returncode,result.stderr[-4000:])
  entry={'case':label,'pass':True,'rc':result.returncode,'stdout_sha256':hashlib.sha256(result.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr.encode()).hexdigest()}
  if expected_rc==0:
   output=json.loads(next(x.split('=',1)[1] for x in result.stdout.splitlines() if x.startswith('GLM53_HOOK_CHILD=')));assert output['pass'] and all(x['installed']==enabled for x in output['processes']);assert all(x['packed']==(label not in ['P_noop','legacy_D_L04']) for x in output['processes']);entry['verified_processes']=output['processes']
  elif label in ['wrong_P_role','pack_without_metadata']:assert 'GLM53_PACKED_HOOK_FATAL' in result.stderr
  else:assert 'Unsupported upstream DeepEP source' in result.stderr
  (ROOT/(label+'.stdout.txt')).write_text(result.stdout);(ROOT/(label+'.stderr.txt')).write_text(result.stderr);cases.append(entry)
 target.write_bytes(original)
 result={'project_id':'glm53','pass':True,'model_requests':0,'SDK_init_calls':0,'NPU_devices':False,'actual_framework_imports':False,'cases':cases,'D_only_parent_and_spawn_worker_injection_verified':True,'source_guard_fails_closed':True,'existing_image_sources_untouched':True}
 (ROOT/'hook-fixture-result.json').write_text(json.dumps(result,indent=2)+'\n');print('GLM53_PACKED_HOOK_CPU_FIXTURE='+json.dumps(result),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');args=parser.parse_args();child() if args.child else main()
