import os,json,functools
from pathlib import Path
from vllm_ascend.attention.context_parallel import sfa_cp
def install():
 classes=[c for c in vars(sfa_cp).values()if isinstance(c,type)and "_build_slot_mapping_replicated_view"in vars(c)]
 assert len(classes)==1
 cls=classes[0];original=cls._build_slot_mapping_replicated_view
 @functools.wraps(original)
 def observe(self,common,block_table):
  if Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0222/diagnostic_enable.json").exists():
   qgpu=common.query_start_loc[:common.num_reqs+1].cpu().tolist()
   qcpu=common.query_start_loc_cpu[:common.num_reqs+1].tolist()
   positions=common.positions[:common.num_input_tokens].cpu().tolist()
   payload=dict(pid=os.getpid(),num_reqs=common.num_reqs,num_input_tokens=common.num_input_tokens,num_actual_tokens=common.num_actual_tokens,query_GPU=qgpu,query_CPU=qcpu,positions=positions[:32],block_table_shape=list(block_table.shape),dcp_size=self.dcp_size,replicated_view_block_size=self.replicated_view_block_size,query_sum=qgpu[-1]-qgpu[0],repeat_output_size=common.num_input_tokens)
   print("GLM_SFA_DCP_METADATA_DIAGNOSTIC "+json.dumps(payload),flush=True)
  return original(self,common,block_table)
 cls._build_slot_mapping_replicated_view=observe
 print("GLM_SFA_DCP_METADATA_DIAGNOSTIC_INSTALLED "+json.dumps(dict(pid=os.getpid(),class_name=cls.__name__,math_changes=0)),flush=True)
