"""CPU control-contract replay of actual fixed-image factory, no torch/NPU work."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json
T=Path(__file__).resolve().parent;native_path=T/'G279_fixed_native_sources/sglang/srt/mem_cache/kv_cache_configurator.py'
unit=ast.parse(native_path.read_text());cls=next(n for n in unit.body if isinstance(n,ast.ClassDef) and n.name=='KVCacheConfigurator');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_build_token_to_kv_pool_allocator')
module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]))
class Pool:
 needs_paged_swa_allocator=True
 def register_mapping(self,mapping):self.mapping=mapping
class HiSparse:pass
class TorchFixture:
 int64='INT64_FIXTURE'
 @staticmethod
 def arange(n,**kwargs):return list(range(n))
namespace={'DeepSeekV4TokenToKVPool':Pool,'DeepSeekV4HiSparseTokenToKVPoolAllocator':HiSparse,'torch':TorchFixture,'get_disagg':lambda:SimpleNamespace(disaggregation_mode='decode')}
exec(compile(module,str(native_path),'exec'),namespace);native=namespace[fn.name]
def factory(path):
 unit=ast.parse(path.read_text());installer=next(n for n in unit.body if isinstance(n,ast.FunctionDef) and n.name=='install');wrapper=next(n for n in installer.body if isinstance(n,ast.FunctionDef) and n.name=='create')
 unit=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]));ns={'prior':native,'LowRatioNPUTokenToKVPool':Pool};exec(compile(unit,str(path),'exec'),ns);return ns['create']
old=factory(T/'sg_pd_g294/lowratio_allocator.py');fixed=factory(T/'lowratio_allocator_g301.py');rows=[]
for full_capacity in [False,True]:
 def context():
  self=SimpleNamespace(is_draft_worker=True,is_hybrid_swa=True,draft_owns_its_depth=True,draft_swa_full_capacity=full_capacity,device='CPU_INTERFACE_FIXTURE',page_size=128)
  pool=Pool();alloc=SimpleNamespace(has_window_pool=True,hands_out_virtual_ids=False,full_to_swa_index_mapping=object())
  kw=dict(sizes=SimpleNamespace(full_max_total_num_tokens=256),token_to_kv_pool=pool,is_dsv4_model=True,req_to_token_pool=object(),token_to_kv_pool_allocator=alloc)
  return self,pool,alloc,kw
 self,pool,alloc,kw=context();assert native(self,**kw) is alloc
 if full_capacity:assert pool.mapping==list(range(385))[:-1]+[-1] and pool.swa_uses_full_indices
 else:assert pool.mapping is alloc.full_to_swa_index_mapping
 self,pool,alloc,kw=context()
 try:old(self,**kw)
 except AssertionError:old_reproduced=True
 else:raise AssertionError('G294 original failure not reproduced')
 self,pool,alloc,kw=context();assert fixed(self,**kw) is alloc
 if full_capacity:assert pool.mapping==list(range(385))[:-1]+[-1] and pool.swa_uses_full_indices
 else:assert pool.mapping is alloc.full_to_swa_index_mapping
 rows.append({'draft_swa_full_capacity':full_capacity,'G294_original_assertion_reproduced':old_reproduced,'corrected_returns_exact_shared_allocator':True,'native_mapping_semantics_preserved':True})
d={'scope':'Actual fixed-image allocator-factory Python control-flow with interface fixtures; no tensors, NPU, model or PD execution','native_source_sha256':hashlib.sha256(native_path.read_bytes()).hexdigest(),'G294_wrapper_sha256':hashlib.sha256((T/'sg_pd_g294/lowratio_allocator.py').read_bytes()).hexdigest(),'candidate_wrapper_sha256':hashlib.sha256((T/'lowratio_allocator_g301.py').read_bytes()).hexdigest(),'cases':rows,'NPU_acceptance':False};(T/'G301_shared_draft_allocator_contract.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
