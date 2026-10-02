"""Opt-in coupled DP control metadata; native operators/models are unchanged.

Source identities and exact substitutions fail closed. Graph padding follows the
minimum mode agreed across DP ranks. SP/oproj/embedding and draft padding remain
mandatory. This module itself imports no accelerator/runtime implementation.
"""
import ast,hashlib,inspect,json,linecache,textwrap,types
METHOD_HASHES={
 "_sync_metadata_across_dp":"3ae98d968ade0f78463b04143be79792d41c484fb938f25cbb6bf3ee8b93ee8e",
 "_determine_batch_execution_and_padding":"812c989f97f1a271731f4b7d1c41938014d567f541aff92aac588dbc1373af50",
}
CALLER_OLD="""allow_dp_padding=((cudagraph_mode != CUDAGraphMode.NONE)
                                  or enable_sp(self.vllm_config)
                                  or oproj_tp_enable()
                                  or embedding_tp_enable()),"""
CALLER_NEW="""allow_dp_padding=(enable_sp(self.vllm_config)
                                  or oproj_tp_enable()
                                  or embedding_tp_enable()),"""
SYNC_OLD="if allow_dp_padding or is_draft_model:"
SYNC_NEW="if allow_dp_padding or is_draft_model or synced_cudagraph_mode != CUDAGraphMode.NONE:"

def make_sources(native_text):
 tree=ast.parse(native_text)
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="NPUModelRunner")
 rows={}
 for name,digest in METHOD_HASHES.items():
  fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
  start=min([fn.lineno]+[n.lineno for n in fn.decorator_list])
  raw="\n".join(native_text.splitlines()[start-1:fn.end_lineno])+"\n"
  if hashlib.sha256(raw.encode()).hexdigest()!=digest:raise RuntimeError("native control source changed: "+name)
  old,new=(SYNC_OLD,SYNC_NEW) if name=="_sync_metadata_across_dp" else (CALLER_OLD,CALLER_NEW)
  if raw.count(old)!=1:raise RuntimeError("native metadata substitution mismatch: "+name)
  patched=textwrap.dedent(raw.replace(old,new))
  ast.parse(patched)
  # Only the metadata condition/call argument changes; every native function call
  # and all their other arguments remain in the original source.
  rows[name]={"original":textwrap.dedent(raw),"patched":patched,"native_sha256":digest,"patched_sha256":hashlib.sha256(patched.encode()).hexdigest()}
 return rows

def compile_methods(rows,native_globals):
 methods={}
 for name,row in rows.items():
  namespace=dict(native_globals);filename=__file__+"::"+name+"::"+row["patched_sha256"]
  linecache.cache[filename]=(len(row["patched"]),None,row["patched"].splitlines(True),filename)
  exec(compile(row["patched"],filename,"exec"),namespace)
  methods[name]=namespace[name]
 return methods

def install_runner_control(runner):
 import vllm_ascend.worker.model_runner_v1 as native
 from vllm.config import CUDAGraphMode
 from vllm.logger import init_logger
 log=init_logger("vllm.glm_coupled_dp_metadata")
 if type(runner) is not native.NPUModelRunner:raise RuntimeError("opt-in requires native NPUModelRunner v1")
 if runner.dp_size<=1:raise RuntimeError("opt-in requires a coupled DP group")
 rows=make_sources(inspect.getsource(native));methods=compile_methods(rows,vars(native))
 sync=methods["_sync_metadata_across_dp"]
 def observed_sync(self,num_tokens,is_draft_model=False,cudagraph_mode=CUDAGraphMode.NONE,allow_dp_padding=False):
  result=sync(self,num_tokens,is_draft_model,cudagraph_mode,allow_dp_padding)
  max_tokens,across,mode=result
  if (not is_draft_model and not allow_dp_padding and cudagraph_mode!=CUDAGraphMode.NONE
      and mode==CUDAGraphMode.NONE and max_tokens>num_tokens):
   count=getattr(self,"_glm_dp_metadata_observations",0)
   if count<3:
    log.info("GLM_DP_METADATA %s",json.dumps({"dp_rank":self.dp_rank,"local_tokens":num_tokens,"max_tokens_across_dp":max_tokens,"synced_mode":mode.name,"local_pre_sync_mode":cudagraph_mode.name,"actual_tokens_after_padding":int(across[self.dp_rank].item()),"tokens_across_dp":across.tolist(),"preserved":"native SP/oproj/embedding/draft/Graph semantics","sample":count+1}))
    self._glm_dp_metadata_observations=count+1
  return result
 runner._sync_metadata_across_dp=types.MethodType(observed_sync,runner)
 runner._determine_batch_execution_and_padding=types.MethodType(methods["_determine_batch_execution_and_padding"],runner)
 log.info("GLM_DP_METADATA_INSTALLED %s",json.dumps({"dp_rank":runner.dp_rank,"methods":{k:{f:v[f] for f in ["native_sha256","patched_sha256"]} for k,v in rows.items()},"native_operator_implementation":"unchanged"}))
