# SPDX-License-Identifier: Apache-2.0
"""Bounded Ascend SFA NoPE call vs independent FP32 softmax oracle.

One request, three synthetic tokens, four local heads and one 128-token page;
not a GLM DSA layer, production cache, distributed run or performance test.
"""
import argparse,datetime,hashlib,importlib.metadata,json,platform
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','output-dir','assignment','owner-identity','library'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--authorized-npu',action='store_true');p.add_argument('--zero-rope',action='store_true');a=p.parse_args()
from deepinfer.hugging_face.glm5_next_cann_probe import require_owner_authorization
assignment=json.loads(a.assignment.read_text());require_owner_authorization(a.authorized_npu,assignment,json.loads(a.owner_identity.read_text()))
if assignment.get('NPU_resources_claimed')!=['Ascend910C:physical-die0']:raise ValueError('explicit assigned die0 claim required')
versions={n:importlib.metadata.version(n) for n in ('torch','torch-npu','flash-linear-attention-npu-a3')}
if versions!={'torch':'2.10.0+cpu','torch-npu':'2.10.0.post7','flash-linear-attention-npu-a3':'26.9.1'}:raise RuntimeError('native tuple differs')
import torch
import torch_npu
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
config=Glm5NextConfig.from_checkpoint(a.checkpoint);d=config.mla_dimensions
if (d['num_attention_heads'],d['qk_nope_head_dim'],d['kv_lora_rank'],d['qk_rope_head_dim'])!=(64,256,512,0):raise ValueError('GLM NoPE config differs')
torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)
lib_sha=hashlib.sha256(a.library.read_bytes()).hexdigest();torch.ops.load_library(str(a.library));op=torch.ops._C_ascend.npu_sparse_flash_attention
schema=str(op.default._schema)
a.output_dir.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[2]
report={'status':'in_progress','architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'versions':versions,'library':str(a.library),'library_sha256':lib_sha,'schema':schema,'source_sha256':hashlib.sha256((root/'tests/ascend/probe_glm5_next_sfa_npu.py').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest(),'NPU_execution':True,'workload':{'request_count':1,'tokens':3,'heads':4,'latent':512,'query_score_dimension':256,'scale':0.0625,'block_size':128,'index_width':2051,'input':'synthetic BF16 tensors, independently calculated CPU FP32 reference','zero_rope_dim':64 if a.zero_rope else 0},'checks':{},'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
 q=a.output_dir/'sfa-result.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(q)
save()
try:
 torch.manual_seed(5344);query=(torch.randn(3,4,512)*.2).bfloat16();values=(torch.randn(3,512)*.2).bfloat16()
 cache=torch.zeros((1,128,1,512),dtype=torch.bfloat16);cache[0,:3,0]=values
 indices=torch.full((3,1,2051),-1,dtype=torch.int32)
 for i in range(3):indices[i,0,:i+1]=torch.arange(i+1,dtype=torch.int32)
 reference=torch.empty(3,4,512)
 for i in range(3):
  scores=query[i].float()@values[:i+1].float().T*0.0625
  reference[i]=torch.softmax(scores,dim=-1)@values[:i+1].float()
 with torch.inference_mode():
  result=op(query=query.npu(),key=cache.npu(),value=cache.npu(),sparse_indices=indices.npu(),scale_value=0.0625,sparse_block_size=1,block_table=torch.tensor([[0]],dtype=torch.int32).npu(),actual_seq_lengths_query=torch.tensor([3],dtype=torch.int32).npu(),actual_seq_lengths_kv=torch.tensor([3],dtype=torch.int32).npu(),query_rope=torch.zeros((3,4,64),dtype=torch.bfloat16).npu() if a.zero_rope else None,key_rope=torch.zeros((1,128,1,64),dtype=torch.bfloat16).npu() if a.zero_rope else None,layout_query='TND',layout_kv='PA_BSND',sparse_mode=3,attention_mode=2,return_softmax_lse=False)
  torch_npu.npu.synchronize();actual=result[0].detach().cpu()
 if actual.shape!=(3,4,512) or actual.dtype!=torch.bfloat16 or not bool(torch.isfinite(actual).all()):raise ValueError('SFA output shape/dtype/finite differs')
 diff=actual.float()-reference;rel=float(diff.square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-12));ma=float(diff.abs().max());bound=.03*float(reference.abs().max())+1e-5
 passed=rel<=.01 and ma<=bound;report['checks']['npu_sfa_vs_fp32_softmax']={'relative_rmse':rel,'relative_rmse_limit':.01,'max_abs':ma,'max_abs_limit':bound,'passed':passed};torch.save({'query':query,'values':values,'indices':indices,'actual':actual,'reference_fp32':reference},a.output_dir/'sfa-outputs.pt');save()
 if not passed:raise AssertionError('SFA oracle mismatch')
 report['status']='passed'
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
