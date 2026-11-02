# SPDX-License-Identifier: Apache-2.0
"""Isolate native layer31 with identical real inputs captured from 45 layers.

Observe repeat and segmentation drift in DSA, MoE and residual separately.
No model-accuracy or performance claim.
"""
import argparse,datetime,hashlib,importlib.metadata,json,platform
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','output-dir','assignment','owner-identity','library','prior'):p.add_argument('--'+name,type=Path,required=True)
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
from deepinfer.model_executor.models.glm5_next.native_model import load_native_decoder
from deepinfer.model_executor.models.glm5_next.native_dsa import make_dsa_operations

torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);a.output_dir.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();names=['tests/ascend/probe_glm5_next_layer31_isolation_npu.py','deepinfer/model_executor/models/glm5_next/native_model.py','deepinfer/model_executor/models/glm5_next/native_dsa.py','deepinfer/model_executor/models/glm5_next/hybrid_decoder.py','deepinfer/model_executor/models/glm5_next/expert_bank.py','deepinfer/model_executor/models/glm5_next/moe.py','deepinfer/model_executor/models/glm5_next/mhc.py','deepinfer/npu_ops/w8a8.py']
report={'status':'in_progress','architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'versions':versions,'NPU_execution':True,'library_sha256':sha(a.library),'source_sha256':{n:sha(root/n) for n in names},'config_sha256':sha(a.checkpoint/'config.json'),'prior_sha256':sha(a.prior),'scope':'real layer31 DSA/MoE with exactly equal saved layer30 inputs, two fresh repeated/segmented executions, captured DSA/FFN outputs; no external model reference','comparisons':{},'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
 q=a.output_dir/'layer31-isolation-result.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(q)
def compare(name,x,y):
 x=x.detach().float().cpu();y=y.detach().float().cpu();d=x-y
 report['comparisons'][name]={'shape':list(x.shape),'relative_rmse':float(d.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12)) if d.numel() else 0.,'max_abs':float(d.abs().max()) if d.numel() else 0.,'nonzero_count':int((d!=0).sum())}
save()
try:
 raw=torch.load(a.prior,map_location='cpu',weights_only=True);x=raw['prefill'][30];y=raw['segmented'][30]
 if not torch.equal(x,y):raise ValueError('captured preceding-layer inputs are not bit identical')
 reader=ModelSlimCheckpoint(a.checkpoint);config=Glm5NextConfig.from_checkpoint(a.checkpoint);op=make_dsa_operations(config,sfa_library=a.library)[31]
 with torch.inference_mode():
  layer=load_native_decoder(reader,31,'npu',dsa_operation=op,max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2,expert_policy='selected_per_token')
  phase=['A'];captures={k:{n:[] for n in ('attention','ffn','ffn_input','route_ids','route_weights','expert_output')} for k in ('A','B','C')};hooks=[]
  for name,module in (('attention',layer.attention),('ffn',layer.mlp)):
   def capture(module,args,out,label=name):captures[phase[0]][label].append((out[0] if label=='attention' else out).detach().clone())
   hooks.append(module.register_forward_hook(capture))
  def capture_ffn_input(module,args,out):captures[phase[0]]['ffn_input'].append(args[0].detach().clone())
  def capture_routes(module,args,out):
   captures[phase[0]]['route_ids'].append(out.expert_ids.detach().clone())
   captures[phase[0]]['route_weights'].append(out.weights.detach().clone())
  def capture_experts(module,args,out):captures[phase[0]]['expert_output'].append(out.detach().clone())
  hooks.append(layer.mlp.register_forward_hook(capture_ffn_input))
  hooks.append(layer.mlp.router.register_forward_hook(capture_routes))
  hooks.append(layer.mlp.experts.register_forward_hook(capture_experts))
  x=x.npu();out_a,state_a=layer(x);phase[0]='B';out_b,state_b=layer(x);phase[0]='C';left,state_c=layer(x[:1]);right,state_c=layer(x[1:],state_c);out_c=torch.cat((left,right))
  torch_npu.npu.synchronize()
  compare('same_input_repeat_residual',out_b,out_a);compare('same_input_segmented_residual',out_c,out_a)
  for name in ('attention','ffn','ffn_input','route_ids','route_weights','expert_output'):
   aa=torch.cat(captures['A'][name]);bb=torch.cat(captures['B'][name]);cc=torch.cat(captures['C'][name]);compare('same_input_repeat_'+name,bb,aa);compare('same_input_segmented_'+name,cc,aa)
  for name in ('latent','compressed','tail','tail_positions'):
   compare('same_input_repeat_state_'+name,getattr(state_b,name),getattr(state_a,name));compare('same_input_segmented_state_'+name,getattr(state_c,name),getattr(state_a,name))
  compare('replay_vs_prior_prefill',out_a,raw['prefill'][31]);compare('prior_segmented_vs_prefill',raw['segmented'][31],raw['prefill'][31])
  for h in hooks:h.remove()
  torch.save({'A':out_a.cpu(),'B':out_b.cpu(),'C':out_c.cpu(),'captures':{k:{n:[v.cpu() for v in z] for n,z in val.items()} for k,val in captures.items()}},a.output_dir/'layer31-isolation-outputs.pt')
 report['status']='observed'
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
