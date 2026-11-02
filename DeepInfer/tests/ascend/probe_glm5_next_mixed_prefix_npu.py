# SPDX-License-Identifier: Apache-2.0
"""Real checkpoint four-layer KDA/dense + DSA/MoE prefix on one die.

Compare complete versus segmented forward and states. This is not an independent
whole-model oracle, 45-layer model, full-vocabulary generation or performance test.
"""
import argparse,datetime,hashlib,importlib.metadata,json,platform
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','output-dir','assignment','owner-identity','library'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--authorized-npu',action='store_true');p.add_argument('--selected-experts',action='store_true');p.add_argument('--reference-prefix',type=Path);a=p.parse_args()
from deepinfer.hugging_face.glm5_next_cann_probe import require_owner_authorization
assignment=json.loads(a.assignment.read_text());require_owner_authorization(a.authorized_npu,assignment,json.loads(a.owner_identity.read_text()))
if assignment.get('NPU_resources_claimed')!=['Ascend910C:physical-die0']:raise ValueError('explicit assigned die0 claim required')
versions={n:importlib.metadata.version(n) for n in ('torch','torch-npu','flash-linear-attention-npu-a3')}
if versions!={'torch':'2.10.0+cpu','torch-npu':'2.10.0.post7','flash-linear-attention-npu-a3':'26.9.1'}:raise RuntimeError('native tuple differs')
import torch
import torch_npu
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.cache import DSAState
from deepinfer.model_executor.models.glm5_next.native_model import load_native_decoder
from deepinfer.model_executor.models.glm5_next.native_dsa import make_dsa_operations
from deepinfer.model_executor.models.glm5_next.text_boundary import GlmTextBoundary

torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);a.output_dir.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['tests/ascend/probe_glm5_next_mixed_prefix_npu.py','deepinfer/model_executor/models/glm5_next/native_model.py','deepinfer/model_executor/models/glm5_next/native_dsa.py','deepinfer/model_executor/models/glm5_next/hybrid_decoder.py','deepinfer/model_executor/models/glm5_next/decoder.py','deepinfer/model_executor/models/glm5_next/kda.py','deepinfer/model_executor/models/glm5_next/moe.py','deepinfer/model_executor/models/glm5_next/expert_bank.py','deepinfer/model_executor/models/glm5_next/cache.py','deepinfer/npu_ops/w8a8.py']
report={'status':'in_progress','architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'versions':versions,'NPU_execution':True,'library_sha256':sha(a.library),'source_sha256':{n:sha(root/n) for n in names},'config_sha256':sha(a.checkpoint/'config.json'),'index_sha256':sha(a.checkpoint/'quant_model_weights.safetensors.index.json'),'tokens':[1,2,3,4,5],'layer_ids':[0,1,2,3],'expert_policy':'selected_per_token' if a.selected_experts else 'all','scope':'real checkpoint first four layers:3 KDA/dense plus 1 DSA/all288expert MoE; selected32 output rows; no 45-layer/generation claim','checks':{},'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
 q=a.output_dir/'mixed-prefix-result.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(q)
def compare(name,x,y,limit=.01):
 x=x.detach().float().cpu();y=y.detach().float().cpu()
 if x.shape!=y.shape or not bool(torch.isfinite(x).all()):raise ValueError('comparison shape/finite differs: '+name)
 d=x-y;rel=float(d.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12)) if d.numel() else 0.;ma=float(d.abs().max()) if d.numel() else 0.;bound=.03*float(y.abs().max())+1e-5 if y.numel() else 1e-5
 passed=rel<=limit and ma<=bound;report['checks'][name]={'shape':list(x.shape),'relative_rmse':rel,'relative_rmse_limit':limit,'max_abs':ma,'max_abs_limit':bound,'passed':passed};save()
 # Collect all diagnostics before reporting a failed comparison.
def state_tensors(state):
 if isinstance(state,DSAState):return {name:getattr(state,name).detach().cpu().clone() for name in ('latent','compressed','tail','tail_positions')}
 return {'recurrent':state.recurrent.detach().cpu().clone(),**{f'conv{i}':v.detach().cpu().clone() for i,v in enumerate(state.conv)}}
def forward(layers,hidden,states):
 updates=[];intermediates=[]
 for layer,state in zip(layers,states,strict=True):
  hidden,state=layer(hidden,state);updates.append(state);intermediates.append(hidden.clone())
 return hidden,updates,intermediates
save()
try:
 config=Glm5NextConfig.from_checkpoint(a.checkpoint)
 if [(s.attention,s.mlp) for s in config.layers[:4]]!=[('linear_attention','dense')]*3+[('deepseek_sparse_attention','sparse')]:raise ValueError('unexpected first-four actual topology')
 reader=ModelSlimCheckpoint(a.checkpoint);operations=make_dsa_operations(config,sfa_library=a.library);layers=[]
 with torch.inference_mode():
  boundary=GlmTextBoundary.from_checkpoint(reader,'npu',max_unique_tokens=5,max_tokens=5);hidden=boundary.embed(torch.tensor(report['tokens'],dtype=torch.int64))
  for i in range(4):
   print('loading actual layer',i,flush=True)
   layers.append(load_native_decoder(reader,i,'npu',dsa_operation=operations.get(i),max_expert_checkpoint_bytes=512*1024**2 if a.selected_experts else 12*1024**3,max_expert_resident_bytes=512*1024**2 if a.selected_experts else 12*1024**3,expert_policy='selected_per_token' if a.selected_experts else 'all'))
  print('all four layers loaded; running full and segmented forwards',flush=True)
  full,full_states,full_layers=forward(layers,hidden,[None]*4)
  first,first_states,first_layers=forward(layers,hidden[:3],[None]*4);last,last_states,last_layers=forward(layers,hidden[3:],first_states)
  torch_npu.npu.synchronize()
  raw={'full_layers':[v.cpu() for v in full_layers],'segmented_layers':[torch.cat((a,b)).cpu() for a,b in zip(first_layers,last_layers,strict=True)],'full_states':[state_tensors(s) for s in full_states],'segmented_states':[state_tensors(s) for s in last_states]}
  torch.save(raw,a.output_dir/'mixed-prefix-diagnostics.pt')
  for i,(whole,left,right) in enumerate(zip(full_layers,first_layers,last_layers,strict=True)):
   compare(f'layer{i}_residual_streams',torch.cat((left,right)),whole,limit=.001)
  compare('segmented_residual_streams',torch.cat((first,last)),full)
  for i,(whole,part) in enumerate(zip(full_states,last_states,strict=True)):
   if whole.tokens_seen!=5 or part.tokens_seen!=5:raise ValueError('state token count differs')
   for name,value in state_tensors(whole).items():compare(f'layer{i}_{name}',state_tensors(part)[name],value,limit=.001)
  # Diagnostic fixed vocabulary shard only; final boundary normally follows all45 layers.
  head=boundary.head_shard(0,32,max_rows=32);full_logits=head(boundary.final_hidden(full));parts=head(boundary.final_hidden(torch.cat((first,last))))
  compare('segmented_diagnostic_head32',parts,full_logits)
  if a.reference_prefix is not None:
   baseline=torch.load(a.reference_prefix,map_location='cpu',weights_only=True)
   compare('selected_vs_all_experts_full_streams',full,baseline['streams'],limit=.001)
   compare('selected_vs_all_experts_head32',full_logits,baseline['head32'],limit=.001)
  torch_npu.npu.synchronize();torch.save({'streams':full.cpu(),'segmented_streams':torch.cat((first,last)).cpu(),'head32':full_logits.cpu(),'states':[state_tensors(s) for s in full_states]},a.output_dir/'mixed-prefix-outputs.pt')
  failed=[name for name,value in report['checks'].items() if not value['passed']]
  if failed:raise AssertionError('mixed prefix comparisons failed: '+', '.join(failed))
  report['status']='passed'
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
