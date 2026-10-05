"""Task-private native draft decode capture geometry; math/operators unchanged."""
import copy,functools,hashlib,inspect,json,os
from pathlib import Path
def install():
 from vllm.config import replace
 from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import AutoRegressiveAclGraphManager as C
 original=C.__init__;replay=C.run_fullgraph
 assert hashlib.sha256(Path(inspect.getfile(C)).read_bytes()).hexdigest()=="ae42c6970145aaa0e7edc232bb7e322b00070a3917b3d6c01e5651853d486de5"
 assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest()=="baa3740a0c306ec51d202216b40c056c7a4c5a41652d1d026caf0d4aa99d45d4"
 @functools.wraps(original)
 def dense_init(self,vllm_config,device,cudagraph_mode,decode_query_len,lora_capture_cases=None):
  target_sizes=list(vllm_config.compilation_config.cudagraph_capture_sizes)
  target_max=vllm_config.compilation_config.max_cudagraph_capture_size
  chosen=vllm_config
  if decode_query_len==1:
   assert vllm_config.scheduler_config.max_num_seqs==8 and vllm_config.num_speculative_tokens==2
   assert not vllm_config.speculative_config.enforce_eager
   comp=copy.copy(vllm_config.compilation_config);comp.cudagraph_capture_sizes=list(range(1,9));comp.max_cudagraph_capture_size=8
   chosen=replace(vllm_config,compilation_config=comp)
  result=original(self,chosen,device,cudagraph_mode,decode_query_len,lora_capture_cases)
  assert vllm_config.compilation_config.cudagraph_capture_sizes==target_sizes and vllm_config.compilation_config.max_cudagraph_capture_size==target_max
  if decode_query_len==1:assert self.capture_sizes==list(range(1,9))
  print("GLM_DRAFT_DENSE_CAPTURE_INIT "+json.dumps(dict(pid=os.getpid(),decode_query_len=decode_query_len,draft_decode=decode_query_len==1,capture_sizes=self.capture_sizes,target_sizes=target_sizes,target_unchanged=True,native_math_changes=0)),flush=True)
  return result
 @functools.wraps(replay)
 def checked_replay(self,desc):
  if not self.is_draft_model_prefill:
   assert desc.num_tokens==desc.num_reqs==self.speculator.input_batch.num_reqs
   if Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0225/diagnostic_enable.json").exists():
    q=self.speculator.input_buffers.query_start_loc[:desc.num_reqs+1].cpu().tolist()
    positions=self.speculator.input_buffers.positions[:desc.num_tokens].cpu().tolist()
    print("GLM_DRAFT_DENSE_REPLAY_DIAGNOSTIC "+json.dumps(dict(pid=os.getpid(),active_reqs=self.speculator.input_batch.num_reqs,graph_tokens=desc.num_tokens,graph_reqs=desc.num_reqs,query_GPU=q,positions=positions,padding=0,mode=str(desc.cg_mode))),flush=True)
    assert q==list(range(desc.num_reqs+1))
  return replay(self,desc)
 C.__init__=dense_init;C.run_fullgraph=checked_replay
 print("GLM_DRAFT_DENSE_CAPTURE_INSTALLED "+json.dumps(dict(pid=os.getpid(),native_math_changes=0)),flush=True)
