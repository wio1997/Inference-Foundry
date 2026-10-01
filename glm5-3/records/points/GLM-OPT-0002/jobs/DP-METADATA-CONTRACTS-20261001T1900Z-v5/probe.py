import pathlib,json,types,hashlib,torch,numpy as np
from types import SimpleNamespace as NS
from vllm.config import CompilationConfig,CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher,BatchDescriptor
from coupled_dp_metadata import make_sources,compile_methods
native_text=pathlib.Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py").read_text()
rows=make_sources(native_text)
flags={"sp":False,"oproj":False,"embedding":False,"skip":False}
packed=None
class Dist:
 @staticmethod
 def all_reduce(tensor,group):tensor.copy_(packed)
ns={"torch":torch,"np":np,"CUDAGraphMode":CUDAGraphMode,"BatchDescriptor":BatchDescriptor,
 "dist":Dist,"get_dp_group":lambda:NS(cpu_group="cpu_fixture"),"should_skip_allreduce_across_dp_group":lambda v,d:flags["skip"],
 "enable_sp":lambda v:flags["sp"],"oproj_tp_enable":lambda:flags["oproj"],"embedding_tp_enable":lambda:flags["embedding"],
 "_post_process_cudagraph_mode":lambda t:int(t[1,:].min().item()),"CUDAGraphStat":object}
original={}
for name,row in rows.items():
 env=dict(ns);exec(compile(row["original"],"installed_native_control","exec"),env);original[name]=env[name]
patched=compile_methods(rows,ns)
c=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY",cudagraph_capture_sizes=[48],max_cudagraph_capture_size=48)
v=NS(compilation_config=c,num_speculative_tokens=5,scheduler_config=NS(max_num_seqs=8),lora_config=None)
dispatcher=CudagraphDispatcher(v);dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY,6)
def runner(rank,methods,prefill=False):
 config=NS(parallel_config=NS(data_parallel_size=2,data_parallel_rank=rank,tensor_parallel_size=16),observability_config=NS(cudagraph_metrics=False))
 r=NS(dp_size=2,dp_rank=rank,vllm_config=config,parallel_config=config.parallel_config,use_dcp=True,uniform_decode_query_len=6,
 input_batch=NS(num_computed_tokens_cpu=np.array([0 if prefill else 81932]*8),num_prompt_tokens=np.array([81932]*8),lora_id_to_lora_request={}),
 model_config=NS(is_encoder_decoder=False),cudagraph_dispatcher=dispatcher)
 r._pad_for_sequence_parallelism=lambda n:((n+15)//16)*16 if flags["sp"] else n
 for name,method in methods.items():setattr(r,name,types.MethodType(method,r))
 return r
def local_desc(tokens,prefill):
 return dispatcher.dispatch(((tokens+15)//16)*16 if flags["sp"] else tokens,uniform_decode=not prefill)
cases=[]
# Source method integration: asymmetric decode/prefill, both graph, both eager,
# forced SP/oproj/embedding padding; every new pair returns identical DP metadata.
for name,tokens,prefill,forced in [
 ("decode_prefill",[6,4096],[False,True],None),
 ("prefill_decode",[4096,6],[True,False],None),
 ("both_decode",[6,12],[False,False],None),
 ("both_prefill",[512,4096],[True,True],None),
 ("sp_required",[6,4096],[False,True],"sp"),
 ("oproj_required",[6,4096],[False,True],"oproj"),
 ("embedding_required",[6,4096],[False,True],"embedding"),
]:
 for k in flags:flags[k]=False
 if forced:flags[forced]=True
 if forced=="sp":
  sc=CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY",cudagraph_capture_sizes=[48],max_cudagraph_capture_size=48)
  sv=NS(compilation_config=sc,num_speculative_tokens=5,scheduler_config=NS(max_num_seqs=8),lora_config=None)
  dispatcher=CudagraphDispatcher(sv);dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY,6)
 else:
  dispatcher=CudagraphDispatcher(v);dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY,6)
 descriptors=[local_desc(n,p) for n,p in zip(tokens,prefill)]
 packed=torch.tensor([[d.num_tokens for mode,d in descriptors],[mode.value for mode,d in descriptors]],dtype=torch.int32)
 output={}
 for label,methods in [("native",original),("candidate",patched)]:
  vals=[]
  for rank in [0,1]:
   r=runner(rank,methods,prefill[rank]);n=tokens[rank]
   reqs=1 if prefill[rank] else n//6; scheduled=n if prefill[rank] else 6
   result=r._determine_batch_execution_and_padding(n,reqs,np.array([scheduled]*reqs),scheduled,False)
   vals.append({"mode":result[0].name,"tokens_after_padding":result[1].num_tokens,"across":result[3].tolist()})
  output[label]=vals
 assert output["candidate"][0]["across"]==output["candidate"][1]["across"],(name,output)
 if name=="decode_prefill":assert output["native"][0]["tokens_after_padding"]==4096 and output["candidate"][0]["tokens_after_padding"]==48
 if name=="prefill_decode":assert output["native"][1]["tokens_after_padding"]==4096 and output["candidate"][1]["tokens_after_padding"]==48
 if forced:assert output["candidate"]==output["native"],(name,output)
 if name in ["both_decode","both_prefill"]:assert output["candidate"]==output["native"],(name,output)
 cases.append({"name":name,"output":output})
# Direct sync preserves draft padding, DP1 and skipped-collective path.
for name,draft,size,skip,modes in [
 ("draft_eager_required",True,2,False,[0,0]),("single_DP_unchanged",False,1,False,[0,0]),
 ("skip_collective_unchanged",False,2,True,[2,0]),("synced_piecewise_padding",False,2,False,[2,1]),
]:
 for k in flags:flags[k]=False
 flags["skip"]=skip;packed=torch.tensor([[6,12],modes],dtype=torch.int32);out={}
 for label,methods in [("native",original),("candidate",patched)]:
  r=runner(0,methods);r.dp_size=size
  value=r._sync_metadata_across_dp(6,is_draft_model=draft,cudagraph_mode=CUDAGraphMode(modes[0]),allow_dp_padding=modes[0]!=0)
  out[label]={"max_tokens":value[0],"across":value[1].tolist() if value[1] is not None else None,"mode":value[2].name}
 assert out["native"]==out["candidate"],(name,out);cases.append({"name":name,"output":out})
# Fail closed on changed source/version and repeated/different substitutions.
mut=native_text.replace("if allow_dp_padding or is_draft_model:","if allow_dp_padding:")
try:make_sources(mut)
except RuntimeError:pass
else:raise AssertionError("source guard accepted changed metadata")
print(json.dumps({"valid":True,"cases":cases,"case_count":len(cases)+1,"source_guard":True,
 "methods":{k:{f:v[f] for f in ["native_sha256","patched_sha256"]} for k,v in rows.items()},
 "installed_native_source_sha256":hashlib.sha256(native_text.encode()).hexdigest(),"inference_calls":0,
 "limits":["CPU source method+actual native dispatcher(maxK5,maxseq8,onecapture48), simulated CPU collective inputs; not exact livecapturekeys","No NPU allocation/EngineCore/worker/model weights or inference created","SP/oproj/embedding/draft/Graph invariants covered; live multi-rank semantics/cost remain to test"]}))
