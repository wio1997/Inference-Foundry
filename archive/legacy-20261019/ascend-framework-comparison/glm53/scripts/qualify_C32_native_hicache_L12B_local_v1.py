"""Pure CPU admission/source/lazy-hook qualification; no SDK/framework imports."""
import ast,hashlib,importlib.util,json,os,subprocess,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=json.loads((ROOT/'config/sglang-C32-target-graph-fullmesh-stage-v3.json').read_text())['bundles'][0]
 proof={};bundles={}
 for kind in ['candidate','rollback']:
  path=ROOT/('config/sglang-C32-native-HiCache-L12B-'+kind+'-stage-v1.json');b=json.loads(path.read_text())['bundles'][0];bundles[kind]=b
  for key,item in b['files'].items():
   p=ROOT/item['local'];assert sha(p)==item['sha256']
   if p.suffix=='.py':ast.parse(p.read_text())
  for n in [34,35,36,41]:
   key='config/node-'+str(n)+'.json';before=json.loads((ROOT/old['files'][key]['local']).read_text());after=json.loads((ROOT/b['files'][key]['local']).read_text());assert before['launch_args']==after['launch_args']
   supkey='supervisor-config/node-'+str(n)+'.json';s0=json.loads((ROOT/old['files'][supkey]['local']).read_text());s1=json.loads((ROOT/b['files'][supkey]['local']).read_text());assert s1.pop('run_id')==b['runtime_run_id'];s0.pop('run_id')
   if kind=='candidate' and n in [34,35]:
    expected={'GLM53_C32_NATIVE_HICACHE_MTP':'1','GLM53_NODE':str(n),'SGLANG_HICACHE_HOST_MEM_BACKEND':'memfabric'}
    assert s1==s0|{'model_env':s0['model_env']|expected}
    assert after['environment_intent']==before['environment_intent']|expected
   else:assert s1==s0 and before==after
   for k,v in after['environment_intent'].items():assert s1['model_env'][k]==v
  for key,item in old['files'].items():
   if key.startswith('supervisor-config/'):continue
   if kind=='candidate' and key in ['config/node-34.json','config/node-35.json','python/sitecustomize.py']:continue
   assert b['files'][key]['sha256']==item['sha256'],key
  proof[kind]={'manifest_sha256':sha(path),'file_count':len(b['files']),'launch_args_all4_unchanged':True,'D_source_config_env_unchanged':True,'provider_SDK_config_unchanged':True}
 hook=ROOT/'scripts/glm53_C32_hicache_native_import_hook_v1.py';spec=importlib.util.spec_from_file_location('native_hook',hook);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert not any(k.startswith(('torch','sglang','memfabric_hybrid','memcache_hybrid')) for k in sys.modules)
 pinned=ROOT/'runs/20261018T-L11-P-trace-hotspots-readonly-v1/source/sgl-workspace__sglang__python__sglang__srt__mem_cache__pool_host__mla.py'
 class Pool:
  def _transfer_ascendc_sparse_copy(self,*args,**kwargs):raise AssertionError('Not a numerical fixture')
 module=types.SimpleNamespace(__file__=str(pinned),MLATokenToKVPoolHost=Pool)
 class Original:
  def exec_module(self,module):pass
 sys.path.insert(0,str(ROOT/'scripts'));os.environ.update(GLM53_NODE='34',GLM53_C32_NATIVE_HICACHE_MTP='1',SGLANG_HICACHE_HOST_MEM_BACKEND='memfabric')
 m.Loader(Original()).exec_module(module);assert Pool._glm53_C32_native_MTP_installed
 bad=types.SimpleNamespace(__file__=str(hook),MLATokenToKVPoolHost=Pool)
 try:m.Loader(Original()).exec_module(bad)
 except AssertionError:pass
 else:raise AssertionError('Wrong native source accepted')
 children=[]
 for node in ['34','35','36','41']:
  code='import importlib.util,sys; p='+repr(str(hook))+";s=importlib.util.spec_from_file_location('hook',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.activate();assert not any(k.startswith(('torch','sglang','memfabric_hybrid','memcache_hybrid')) for k in sys.modules);print('READY')"
  env=os.environ|{'GLM53_NODE':node};r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=10)
  assert (r.returncode==0)==(node in ['34','35']);children.append({'node':node,'accepted':r.returncode==0})
 extension=bundles['candidate']['private_optional_extension'];assert sha(ROOT/extension['local_artifact'])==extension['sha256']
 assert extension['readonly'] and extension['nodes']==[34,35]
 assert 'private_optional_extension' not in bundles['rollback']
 result={'project_id':'glm53','pass':True,'kind':'CPU_ONLY_NOT_NPU_CORRECTNESS','bundles':proof,'hook_sha256':sha(hook),'P_lazy_hook_accepts_D_rejects':children,'source_mismatch_rejected':True,'SDK_init_calls':0,'NPU_calls':0,'model_requests':0,'candidate_deployed':False,'native_numeric_fixture_required':True}
 p=ROOT/'results/C32-L12B-local-source-CPU-review-v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
