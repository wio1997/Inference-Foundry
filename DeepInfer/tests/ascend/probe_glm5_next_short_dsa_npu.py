# SPDX-License-Identifier: Apache-2.0
"""Actual GLM DSA projections plus short native SFA, bounded single-die probe.

Compare selected real-weight heads to existing independent FP32 CPU absorbed
MLA reference and check complete versus segmented state. No sparse ranking
beyond topk, distributed execution, full decoder/MoE or generation claim.
"""
import argparse,datetime,hashlib,importlib.metadata,json,platform
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','output-dir','assignment','owner-identity','library'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--authorized-npu',action='store_true');a=p.parse_args()
from deepinfer.hugging_face.glm5_next_cann_probe import require_owner_authorization
assignment=json.loads(a.assignment.read_text());require_owner_authorization(a.authorized_npu,assignment,json.loads(a.owner_identity.read_text()))
if assignment.get('NPU_resources_claimed')!=['Ascend910C:physical-die0']:raise ValueError('explicit assigned die0 claim required')
versions={n:importlib.metadata.version(n) for n in ('torch','torch-npu','flash-linear-attention-npu-a3')}
if versions!={'torch':'2.10.0+cpu','torch-npu':'2.10.0.post7','flash-linear-attention-npu-a3':'26.9.1'}:raise RuntimeError('native tuple differs')
import torch
import torch_npu
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout,state_from_aligned_prefix
from deepinfer.model_executor.models.glm5_next.projections import GlmMLAProjections,GlmIndexerProjections
from deepinfer.model_executor.models.glm5_next.hybrid_decoder import DSAStep,PreparedDSAAttention
from deepinfer.model_executor.models.glm5_next.text_boundary import GlmTextBoundary
from deepinfer.model_executor.models.glm5_next.native_dsa import make_dsa_operations
from deepinfer.model_executor.models.glm5_next.absorbed_mla import AbsorbedMLAContract,split_kv_b_cpu,absorbed_sparse_mla_cpu

torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);a.output_dir.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['tests/ascend/probe_glm5_next_short_dsa_npu.py','deepinfer/model_executor/models/glm5_next/native_dsa.py','deepinfer/model_executor/models/glm5_next/projections.py','deepinfer/model_executor/models/glm5_next/hybrid_decoder.py','deepinfer/model_executor/models/glm5_next/cache.py','deepinfer/model_executor/models/glm5_next/absorbed_mla.py','deepinfer/hugging_face/modelslim910c.py','deepinfer/npu_ops/w8a8.py']
report={'status':'in_progress','architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'versions':versions,'NPU_execution':True,'library_sha256':sha(a.library),'source_sha256':{n:sha(root/n) for n in names},'config_sha256':sha(a.checkpoint/'config.json'),'index_sha256':sha(a.checkpoint/'quant_model_weights.safetensors.index.json'),'tokens':[1,2,3,4,5],'layer_id':3,'scope':'real checkpoint embeddings/MLA/indexer, all64 heads, 5-token unsharded DSA; four heads independently checked; no MoE/whole-model claim','checks':{},'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
 q=a.output_dir/'short-dsa-result.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(q)
def compare(name,x,y,limit=.01):
 x=x.detach().float().cpu();y=y.detach().float().cpu()
 if x.shape!=y.shape or not bool(torch.isfinite(x).all()):raise ValueError('comparison shape/finite differs: '+name)
 d=x-y;rel=float(d.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12));ma=float(d.abs().max()) if d.numel() else 0.;bound=.03*float(y.abs().max())+1e-5 if y.numel() else 1e-5
 passed=rel<=limit and ma<=bound;report['checks'][name]={'shape':list(x.shape),'relative_rmse':rel,'relative_rmse_limit':limit,'max_abs':ma,'max_abs_limit':bound,'passed':passed};save()
 if not passed:raise AssertionError('short DSA comparison failed: '+name)
save()
try:
 config=Glm5NextConfig.from_checkpoint(a.checkpoint);reader=ModelSlimCheckpoint(a.checkpoint);layout=DSACacheLayout.from_config(config,logical_block_size=128)
 with torch.inference_mode():
  boundary=GlmTextBoundary.from_checkpoint(reader,'npu',max_unique_tokens=5,max_tokens=5);hidden=boundary.embed(torch.tensor(report['tokens'],dtype=torch.int64))
  mla=GlmMLAProjections.from_checkpoint(reader,3,'npu',logical_block_size=128);indexer=GlmIndexerProjections.from_checkpoint(reader,config,3,'npu',logical_block_size=128)
  operation=make_dsa_operations(config,sfa_library=a.library)[3].bind(mla,indexer);attention=PreparedDSAAttention(mla,indexer,operation)
  initial=state_from_aligned_prefix(torch.empty((0,512),dtype=torch.bfloat16,device='npu'),torch.empty((0,128),dtype=torch.bfloat16,device='npu'),layout)
  proj=mla.project(hidden);idx=indexer(hidden,proj.query_rank);step=DSAStep(proj,idx,torch.arange(5,device='npu',dtype=torch.int64),1/16)
  head_values,projected_state=operation(step,initial)
  packed=torch.full((5,2051),-1,dtype=torch.int32)
  for i in range(5):packed[i,:i+1]=torch.arange(i+1,dtype=torch.int32)
  contract=AbsorbedMLAContract.from_config(config,(0,1,2,3));kv=mla.kv_b.weight[:4*512].detach().cpu().contiguous();weights=split_kv_b_cpu(kv,contract,max_weight_bytes=4*1024**2)
  oracle=absorbed_sparse_mla_cpu(proj.query[:,:4].detach().cpu(),proj.latent.detach().cpu(),packed,torch.arange(5),weights,sink_policy='absent',max_work_elements=8*1024**2)
  compare('actual_weights_four_heads_vs_cpu_absorbed',head_values[:,:4],oracle.values)
  whole,whole_state=attention(hidden,initial);a1,s1=attention(hidden[:3],initial);a2,s2=attention(hidden[3:],s1)
  compare('segmented_projected_output',torch.cat((a1,a2)),whole)
  for name in ('latent','compressed','tail','tail_positions'):
   compare('segmented_state_'+name,getattr(s2,name),getattr(whole_state,name),limit=.001)
  if (projected_state.tokens_seen,whole_state.tokens_seen,s1.tokens_seen,s2.tokens_seen,initial.tokens_seen)!=(5,5,3,5,0):raise ValueError('DSA state positions differ')
  torch.save({'native_selected_heads':head_values[:,:4].cpu(),'cpu_reference_selected_heads':oracle.values,'projected_output':whole.cpu(),'segmented_output':torch.cat((a1,a2)).cpu(),'query_selected':proj.query[:,:4].cpu(),'latent':proj.latent.cpu()},a.output_dir/'short-dsa-outputs.pt')
  report['status']='passed'
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
