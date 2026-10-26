"""Original File+LRU modules and selected controller methods, no remote backend init."""
import argparse,ast,dataclasses,hashlib,importlib.util,json,logging,os,pathlib,re,sys,types,typing
from decimal import Decimal
from types import SimpleNamespace as NS
import torch
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources/installed';records=[]
def shell(name):
 m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m;return m
for n in ['sglang','sglang.srt','sglang.srt.mem_cache','sglang.srt.mem_cache.storage','sglang.srt.mem_cache.storage.file','sglang.srt.utils']:shell(n)
class Flag:
 def __init__(self,v):self.v=v
 def get(self):return self.v
flags=NS(**{'SGLANG_HICACHE_FILE_BACKEND_'+k:Flag(v) for k,v in {'STORAGE_DIR':None,'ENABLE_METADATA_CACHE':False,'METADATA_TTL':5.0,'MAX_SIZE':0,'MIN_FREE_SPACE':0,'EVICTION_RATIO':0.9}.items()})
shell('sglang.srt.environ').envs=flags
mod=types.ModuleType('L037_original_bodies');sys.modules[mod.__name__]=mod;ns=vars(mod);ns.update({k:v for k,v in vars(typing).items() if not k.startswith('_')});ns.update(dataclass=dataclasses.dataclass,field=dataclasses.field,json=json,hashlib=hashlib,torch=torch,logger=logging.getLogger('L037'),argparse=argparse,re=re,Decimal=Decimal,os=os)
def load(path,classes=(),functions=(),aliases=(),methods=None):
 nodes=[]
 for n in ast.parse((S/path).read_text()).body:
  if isinstance(n,ast.ClassDef):
   if n.name in classes:nodes.append(n)
   elif methods and n.name in methods:
    wanted=set(methods[n.name]);body=[x for x in n.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in wanted];assert {x.name for x in body}==wanted;nodes.append(ast.ClassDef(name=n.name,bases=[],keywords=[],body=body,decorator_list=[]))
  elif isinstance(n,ast.FunctionDef) and n.name in functions:nodes.append(n)
  elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in aliases for t in n.targets):nodes.append(n)
 for n in nodes:records.append({'path':path,'symbol':getattr(n,'name','constant'),'AST_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest(),'whole_original_body':True,'whole_class':isinstance(n,ast.ClassDef) and n.name in classes,'selected_methods':(methods or {}).get(getattr(n,'name',None))})
 a=ast.Module(body=[ast.parse('from __future__ import annotations').body[0]]+nodes,type_ignores=[]);ast.fix_missing_locations(a);exec(compile(a,str(S/path),'exec'),ns)
def import_source(name,path):
 p=S/path;spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);records.append({'path':path,'symbol':'WHOLE_MODULE','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'whole_original_module':True});return m
load('python/sglang/srt/utils/common.py',functions=['human_readable_int']);shell('sglang.srt.utils.common').human_readable_int=ns['human_readable_int']
evictor=import_source('sglang.srt.mem_cache.storage.file.lru_file_evictor','python/sglang/srt/mem_cache/storage/file/lru_file_evictor.py')
storage=import_source('sglang.srt.mem_cache.hicache_storage','python/sglang/srt/mem_cache/hicache_storage.py');ns['HiCacheStorageConfig']=storage.HiCacheStorageConfig
load('python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py',methods={'HybridCacheController':['clear_storage_backend','parse_storage_backend_extra_config']})
load('python/sglang/srt/mem_cache/unified_cache/storage_attachment.py',classes=['StorageAttachment'],aliases=['_PREFETCH_POLICIES','_WRITE_POLICIES'])
load('python/sglang/srt/mem_cache/storage/npu_memcache/npu_memcache_store.py',methods={'NpuMemcacheStore':['_init_runtime_fields','_tag_keys']})
load('python/sglang/srt/managers/scheduler_components/weight_updater.py',methods={'SchedulerWeightUpdaterManager':['flush_cache_after_weight_update','record_weight_version_after_update']})
def config(name,mla=False,metadata=False,**kw):
 v=dict(tp_rank=0,tp_size=2,pp_rank=0,pp_size=1,attn_cp_rank=0,attn_cp_size=1,is_mla_model=mla,enable_storage_metrics=False,is_page_first_layout=True,model_name=name,extra_config={'enable_metadata_cache':metadata,'metadata_ttl':-1.0,'max_size':0,'min_free_space':0,'eviction_ratio':0.9});v.update(kw);return storage.HiCacheStorageConfig(**v)
def controller(backend,enabled=True):
 c=ns['HybridCacheController']();c.storage_backend=backend;c.enable_storage=enabled;c.storage_backend_type='file';return c
def attachment(c):return ns['StorageAttachment'](NS(cache_controller=c))
BOUNDARIES=['Whole original HiCacheFile/HiCacheStorage/MetadataCache andLRU modules, real localfilesystem andTorchCPU rawbytes; isolatedowned Run directory only, evictionconfiguredOFF','SelectedoriginalHybrid clear/parser andwholeStorageAttachment class; no fullcontrollerinit/threads/IPC/TP/DP ornativepool','Exactincomingkey is a SHA256fixture digest of frozenCPUinput IDs, not actualnativepagehash/physicalKV. Samekey is heldfixed to inspect backendnamespace andimmutablewrite semantics','NpuMemcache runtimefields/tag methods only withrealstorageconfig andexplicitextra_backend_tag fixture; constructor/nativepackage/network/storecalls neverrun','Weightupdate flush/version callbacks are explicitspies; no weightsloaded andno claimthatfullScheduler/modelupdate ran','External OSError injection at os.remove for one owned file modelsdeletefailure; actualget/set/clear bodies execute, no realdiskfault orservicefailure asserted','AscendPD+MTP/modelidentity/featurepresence unknown; no performancebaseline orlivepatch']
