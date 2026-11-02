# SPDX-License-Identifier: Apache-2.0
"""Isolate layer1 sensitivity to chunking versus actual preceding-layer drift.

Uses real checkpoint weights and saved target-side four-layer diagnostics.
Observation only: reports unchanged thresholds; no performance/full-model claim.
"""
import argparse,datetime,hashlib,importlib.metadata,json,platform
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','output-dir','assignment','owner-identity','prior'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--authorized-npu',action='store_true');a=p.parse_args()
from deepinfer.hugging_face.glm5_next_cann_probe import require_owner_authorization
assignment=json.loads(a.assignment.read_text());require_owner_authorization(a.authorized_npu,assignment,json.loads(a.owner_identity.read_text()))
if assignment.get('NPU_resources_claimed')!=['Ascend910C:physical-die0']:raise ValueError('explicit assigned die0 claim required')
versions={n:importlib.metadata.version(n) for n in ('torch','torch-npu','flash-linear-attention-npu-a3')}
if versions!={'torch':'2.10.0+cpu','torch-npu':'2.10.0.post7','flash-linear-attention-npu-a3':'26.9.1'}:raise RuntimeError('native tuple differs')
import torch
import torch_npu
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.native_model import load_native_decoder
from deepinfer.model_executor.models.glm5_next.mhc import rms_norm

torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);a.output_dir.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();names=['tests/ascend/probe_glm5_next_chunk_sensitivity_npu.py','deepinfer/model_executor/models/glm5_next/native_model.py','deepinfer/model_executor/models/glm5_next/hybrid_decoder.py','deepinfer/model_executor/models/glm5_next/mhc.py','deepinfer/model_executor/models/glm5_next/kda.py','deepinfer/model_executor/models/glm5_next/decoder.py']
report={'status':'in_progress','architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'versions':versions,'NPU_execution':True,'source_sha256':{n:sha(root/n) for n in names},'config_sha256':sha(a.checkpoint/'config.json'),'prior_artifact_sha256':sha(a.prior),'scope':'real layer1, same five saved inputs: full vs 3+2; full on actual tiny layer0-drift input; not independent full model','comparisons':{},'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
 q=a.output_dir/'chunk-sensitivity-result.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(q)
def compare(name,x,y):
 x=x.detach().float().cpu();y=y.detach().float().cpu();d=x-y
 report['comparisons'][name]={'shape':list(x.shape),'relative_rmse':float(d.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12)),'max_abs':float(d.abs().max()),'nonzero_count':int((d!=0).sum())};save()
save()
try:
 raw=torch.load(a.prior,map_location='cpu',weights_only=True);full_input=raw['full_layers'][0].npu();seg_input=raw['segmented_layers'][0].npu()
 reader=ModelSlimCheckpoint(a.checkpoint)
 with torch.inference_mode():
  layer=load_native_decoder(reader,1,'npu',max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2)
  full,full_state=layer(full_input);left,left_state=layer(full_input[:3]);right,seg_state=layer(full_input[3:],left_state);perturbed,perturbed_state=layer(seg_input)
  full_pre,_=layer.attn_hc.pre(full_input);seg_pre,_=layer.attn_hc.pre(seg_input)
  full_norm=rms_norm(full_pre,layer.input_norm,layer.norm_eps);seg_norm=rms_norm(seg_pre,layer.input_norm,layer.norm_eps)
  compare('input_drift_from_layer0',seg_input,full_input);compare('mHC_pre_input_effect',seg_pre,full_pre);compare('attention_norm_input_effect',seg_norm,full_norm)
  compare('same_input_chunking_output',torch.cat((left,right)),full);compare('same_input_chunking_recurrent',seg_state.recurrent,full_state.recurrent)
  compare('same_shape_input_perturbation_output',perturbed,full);compare('same_shape_input_perturbation_recurrent',perturbed_state.recurrent,full_state.recurrent)
  compare('full_replay_vs_prior',full,raw['full_layers'][1]);compare('prior_segmented_vs_full',raw['segmented_layers'][1],raw['full_layers'][1])
  torch_npu.npu.synchronize();torch.save({'full':full.cpu(),'same_input_segmented':torch.cat((left,right)).cpu(),'perturbed':perturbed.cpu(),'full_norm':full_norm.cpu(),'perturbed_norm':seg_norm.cpu()},a.output_dir/'chunk-sensitivity-outputs.pt')
 report['status']='observed'
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
