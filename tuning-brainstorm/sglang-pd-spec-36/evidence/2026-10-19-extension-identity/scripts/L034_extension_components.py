"""Original identity/registry/session CPU bodies with explicitly bounded service interfaces."""
import ast,asyncio,copy,dataclasses,enum,hashlib,importlib.util,json,logging,pathlib,sys,time,types,typing,uuid
from array import array
from collections import OrderedDict
from http import HTTPStatus
from types import SimpleNamespace as NS
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources/installed';sys.path.insert(0,str(R/'.deps/L034-pillow'))
import torch,msgspec,numpy as np
from PIL import Image
records=[];module=types.ModuleType('L034_original_bodies');sys.modules[module.__name__]=module;ns=vars(module)
ns.update({k:v for k,v in vars(typing).items() if not k.startswith('_')});ns.update({'array':array,'asyncio':asyncio,'copy':copy,'dataclasses':dataclasses,'dataclass':dataclasses.dataclass,'field':dataclasses.field,'Enum':enum.Enum,'IntEnum':enum.IntEnum,'auto':enum.auto,'torch':torch,'msgspec':msgspec,'np':np,'logger':logging.getLogger('L034'),'time':time,'uuid':uuid,'HTTPStatus':HTTPStatus,'OrderedDict':OrderedDict})
def load(path,classes=(),functions=(),aliases=(),methods=None,bases=None):
 p=S/path;t=ast.parse(p.read_text());nodes=[]
 for n in t.body:
  if isinstance(n,ast.ClassDef):
   if n.name in classes:nodes.append(n)
   elif methods and n.name in methods:
    wanted=set(methods[n.name]);body=[x for x in n.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in wanted];assert {x.name for x in body}==wanted
    nodes.append(ast.ClassDef(name=n.name,bases=[ast.Name(id=x,ctx=ast.Load()) for x in (bases or {}).get(n.name,[])],keywords=[],body=body,decorator_list=[]))
  elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in functions:nodes.append(n)
  elif isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in aliases for x in n.targets):nodes.append(n)
 for n in nodes:records.append({'path':path,'symbol':getattr(n,'name',ast.unparse(n.targets[0]) if isinstance(n,ast.Assign) else '?'),'AST_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest(),'whole_original_body':True})
 a=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);ast.fix_missing_locations(a);exec(compile(a,str(p),'exec'),ns)
def shell(name):
 m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m;return m
def import_source(name,path):
 p=S/path;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);records.append({'path':path,'symbol':'WHOLE_MODULE','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'whole_original_module':True});return m
for n in ['sglang','sglang.srt','sglang.srt.utils','sglang.srt.lora','sglang.srt.multimodal','sglang.srt.multimodal.cache']:shell(n)
load('python/sglang/srt/utils/common.py',classes=['ConcurrentCounter']);sys.modules['sglang.srt.utils'].ConcurrentCounter=ns['ConcurrentCounter'];rw=import_source('sglang.srt.utils.aio_rwlock','python/sglang/srt/utils/aio_rwlock.py');registry=import_source('sglang.srt.lora.lora_registry','python/sglang/srt/lora/lora_registry.py')
load('python/sglang/srt/model_executor/forward_batch_info.py',classes=['CaptureHiddenMode'])
load('python/sglang/srt/mem_cache/base_prefix_cache.py',classes=['CacheRequestHandle','DecLockRefParams'])
load('python/sglang/srt/disaggregation/utils.py',aliases=['FAKE_BOOTSTRAP_HOST'])
load('python/sglang/srt/dllm/mixin/req.py',classes=['DllmReqPhase','ReqDllmMixin'])
load('python/sglang/srt/managers/schedule_batch.py',classes=['BaseFinishReason','FINISH_ABORT','FINISH_LENGTH','ReqLogprob','ReqKvInfo'],functions=['get_return_hidden_states_mode'],methods={'Req':['__init__','detach_kv','finished','set_finish_with_abort']},bases={'Req':['ReqDllmMixin']})
class TimeStatsBoundary:
 def __init__(self,**kw):pass
 def set_metrics_collector(self,*a):pass
 def set_scheduler_recv_time(self):pass
ns['SchedulerReqTimeStats']=TimeStatsBoundary;ns['get_parallel']=lambda:NS(tp_rank=1)
load('python/sglang/srt/managers/io_struct.py',classes=['BaseReq','PickleWrapper','SessionParams','TokenizedGenerateReqInput','OpenSessionReqInput','OpenSessionReqOutput','CloseSessionReqInput'])
load('python/sglang/srt/session/session_controller.py',classes=['SessionReqNode','Session','SessionController']);ns['log_info_on_rank0']=lambda *a,**kw:None
load('python/sglang/srt/session/streaming_session.py',classes=['_VirtualNode','SessionSlot'],functions=['_is_streaming'],methods={'StreamingSession':['__init__','find_active_slot']})
load('python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py',classes=['UnifiedSessionRefTracker'],aliases=['_CLOSED_SESSION_TOMBSTONE_LIMIT'])
class Flag:
 def __init__(self,v):self.v=v
 def get(self):return self.v
flags=NS(SGLANG_RAGGED_VERIFY_MODE=Flag('static'),SGLANG_ENABLE_OVERLAP_PLAN_STREAM=Flag(False),SGLANG_MM_SKIP_COMPUTE_HASH=Flag(False));ns['envs']=flags;ns['resolving_view']=lambda x:x
load('python/sglang/srt/arg_groups/lora_hook.py',functions=['check_lora_speculative_compatibility'])
context=shell('sglang.srt.runtime_context');model_cfg=NS(revision='CPU_CONFIG_FIXTURE');mm_cfg=NS(disable_fast_image_processor=False,mm_process_config={});context.get_model=lambda:model_cfg;context.get_mm=lambda:mm_cfg
env_module=shell('sglang.srt.environ');env_module.envs=flags
identity=import_source('sglang.srt.multimodal.cache.identity','python/sglang/srt/multimodal/cache/identity.py')
BOUNDARIES=['Original whole registry, RWLock and media identity modules; complete ConcurrentCounter and session classes; Req constructor plus3 helpers extracted whole, not full serving-module import','Real Torch2.8 CPU empty prefix tensor and real msgspec carrier construction; no full GenerateReqInput API validation, IPC union decode, model sampling or model output','Scheduler time/metrics and parallel-rank lookups are explicit peripheral fixtures; no timing/throughput or NPU claim','Sampling attributes and resolved hook config are fixtures; server-arg normalization/publication and native LoRA/graphs/weights not executed','SessionSlot save/restore tests transfer actual ReqKvInfo metadata only; inner cache/component spies record calls, do not allocate/free native KV or prove caller reachability','Media identity uses real Pillow/numpy/Torch CPU snapshots and hash construction; effective runtime config getters are explicit fixtures; no media encoder or GLM multimodal support claim']
