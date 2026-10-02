"""Skip the unused inherited MLA workspace only for guarded GLM SFA builders.

The installed SFA build/drafting/graph paths never read the inherited MLA
workspace. Ordinary MLA and other models keep their native allocation.
Unexpected runtime reads fail before an operator can consume the empty buffer.
"""
import ast,contextvars,functools,hashlib,json,os
from pathlib import Path
SOURCES={
 "/vllm-workspace/vllm/vllm/model_executor/layers/attention/mla_attention.py":"272c01ff03e52cdfaed6fc7bc2e3e5f641d35910ea0bf5365b928c9401c4df5f",
 "/vllm-workspace/vllm-ascend/vllm_ascend/attention/sfa_v1.py":"f15426c8bd823e651eef7b5f1971b1957d7760c54ee787df43a3d4633f2366a4",
 "/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/sfa_cp.py":"17d80412c16270ac0da8a188bdd6d6b7d8b108fbbf92752a6ddf965898085e57",
 "/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/common_cp.py":"55b44b753cfe25794f0a9a42463950bd71ec59d2b4b44f0f6bf4e4c1e46bdf24",
 "/vllm-workspace/vllm-ascend/vllm_ascend/attention/utils.py":"b600a6db861ade066920d1b5666b1c56d4cd0e5015eb63822a94d133eaae9ee5"}
FIELDS=frozenset(("chunked_prefill_workspace","chunked_prefill_workspace_size"))
scope=contextvars.ContextVar("glm_sfa_unused_workspace_constructor",default=False)
def proof():
 trees={}
 for name,sha in SOURCES.items():
  b=Path(name).read_bytes()
  if hashlib.sha256(b).hexdigest()!=sha:raise RuntimeError("native SFA workspace source changed: "+name)
  trees[name]=ast.parse(b)
 sfa=trees[next(k for k in trees if k.endswith("/sfa_v1.py"))]
 c=next(n for n in sfa.body if isinstance(n,ast.ClassDef)and n.name=="AscendSFAMetadataBuilder")
 overrides={n.name for n in c.body if isinstance(n,ast.FunctionDef)}
 assert {"build","build_for_drafting","build_for_graph_capture","_build"}<=overrides
 for name,t in trees.items():
  if name.endswith("/mla_attention.py"):continue
  for cls in[n for n in t.body if isinstance(n,ast.ClassDef)and("SFAMetadataBuilder"in n.name or"SFADCPMetadataBuilder"in n.name or n.name=="DCPMetadataBuilderMixin")]:
   assert not any(isinstance(n,ast.Attribute)and n.attr in FIELDS for n in ast.walk(cls)),name
   for m in[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name!="__init__"]:
    assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=="build"and isinstance(n.func.value,ast.Call)and isinstance(n.func.value.func,ast.Name)and n.func.value.func.id=="super"for n in ast.walk(m))
 mla=trees[next(k for k in trees if k.endswith("/mla_attention.py"))]
 mc=next(n for n in mla.body if isinstance(n,ast.ClassDef)and n.name=="MLACommonMetadataBuilder")
 build=next(n for n in mc.body if isinstance(n,ast.FunctionDef)and n.name=="build")
 assert any(isinstance(n,ast.Attribute)and n.attr in FIELDS and isinstance(n.ctx,ast.Load)for n in ast.walk(build))
 capture=next(n for n in mc.body if isinstance(n,ast.FunctionDef)and n.name=="build_for_cudagraph_capture")
 assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=="self"and n.func.attr=="build"for n in ast.walk(capture))
 return dict(sources=SOURCES,SFA_overrides=sorted(overrides),ordinary_MLA_reads_workspace=True,capture_dispatches_self_build=True)
def eligible(obj,cfg):
 pc=cfg.parallel_config;mc=cfg.model_config;hf=mc.hf_config
 return (type(obj).__module__+"."+type(obj).__name__ in {
 "vllm_ascend.attention.sfa_v1.AscendSFAMetadataBuilder",
 "vllm_ascend.attention.context_parallel.sfa_cp.AscendSFADCPMetadataBuilder"}
 and getattr(hf,"model_type",None) in {"glm_moe_dsa","glm_moe_dsa_mtp"}
 and set(getattr(hf,"architectures",[]) or[])<={"GlmMoeDsaForCausalLM","GlmMoeDsaMTPModel"}
 and bool(getattr(hf,"architectures",[]) or[])
 and pc.tensor_parallel_size==16 and pc.decode_context_parallel_size==16 and pc.prefill_context_parallel_size==1
 and pc.data_parallel_size==2 and pc.enable_expert_parallel)
def install():
 evidence=proof()
 from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder as C
 if getattr(C,"_glm_unused_workspace_installed",False):return evidence
 init=C.__init__;workspace_size=C.determine_chunked_prefill_workspace_size;getattribute=C.__getattribute__
 @functools.wraps(workspace_size)
 def size(cfg):return 0 if scope.get() else workspace_size(cfg)
 @functools.wraps(init)
 def initialized(self,*args,**kwargs):
  cfg=kwargs.get("vllm_config",args[2]if len(args)>2 else None)
  enabled=cfg is not None and eligible(self,cfg)
  object.__setattr__(self,"_glm_unused_workspace_guard",False)
  token=scope.set(enabled)
  try:init(self,*args,**kwargs)
  finally:scope.reset(token)
  if enabled:
   object.__setattr__(self,"_glm_unused_workspace_guard",True)
   buf=object.__getattribute__(self,"chunked_prefill_workspace")
   assert buf.numel()==0
   nominal=workspace_size(cfg);dcp=cfg.parallel_config.decode_context_parallel_size;head=cfg.model_config.get_head_size()
   print("GLM_UNUSED_SFA_WORKSPACE_SKIPPED "+json.dumps(dict(pid=os.getpid(),builder=type(self).__name__,architectures=cfg.model_config.hf_config.architectures,nominal_elements=(nominal+nominal//dcp)*head,nominal_bytes=(nominal+nominal//dcp)*head*cfg.model_config.dtype.itemsize,actual_elements=0,native_math_changes=0)),flush=True)
 @functools.wraps(getattribute)
 def protected(self,name):
  if name in FIELDS:
   try:enabled=object.__getattribute__(self,"_glm_unused_workspace_guard")
   except AttributeError:enabled=False
   if enabled:raise RuntimeError("GLM_UNUSED_SFA_WORKSPACE_READ: "+name)
  return getattribute(self,name)
 C.determine_chunked_prefill_workspace_size=staticmethod(size);C.__init__=initialized;C.__getattribute__=protected;C._glm_unused_workspace_installed=True
 print("GLM_UNUSED_SFA_WORKSPACE_GUARD_INSTALLED "+json.dumps(dict(pid=os.getpid(),sources=SOURCES,native_math_changes=0)),flush=True)
 return evidence

