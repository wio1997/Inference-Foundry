# SPDX-License-Identifier: Apache-2.0
"""Actual checkpoint layer0 factory comparison on one authorized Ascend die.

Compare the newly wired native factory to the previously exercised dense KDA
layer on identical real checkpoint embedding rows and weights. This is not an
independent implementation oracle or full-backbone/generation validation.
"""
import argparse,datetime,hashlib,importlib.metadata,json,platform
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--assignment',type=Path,required=True)
p.add_argument('--owner-identity',type=Path,required=True)
p.add_argument('--authorized-npu',action='store_true')
a=p.parse_args()
from deepinfer.hugging_face.glm5_next_cann_probe import require_owner_authorization
require_owner_authorization(a.authorized_npu,json.loads(a.assignment.read_text()),json.loads(a.owner_identity.read_text()))
assignment=json.loads(a.assignment.read_text())
if assignment.get('NPU_resources_claimed')!=['Ascend910C:physical-die0']:raise ValueError('explicit assigned die0 claim required')
versions={n:importlib.metadata.version(n) for n in ('torch','torch-npu','flash-linear-attention-npu-a3')}
if versions!={'torch':'2.10.0+cpu','torch-npu':'2.10.0.post7','flash-linear-attention-npu-a3':'26.9.1'}:raise RuntimeError('KDA native tuple differs')
import torch
import torch_npu
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.decoder import Glm5NextDenseDecoder
from deepinfer.model_executor.models.glm5_next.native_model import load_native_decoder
from deepinfer.model_executor.models.glm5_next.text_boundary import GlmTextBoundary

torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)
a.output_dir.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2]
files=['tests/ascend/probe_glm5_next_native_factory_npu.py','deepinfer/model_executor/models/glm5_next/native_model.py','deepinfer/model_executor/models/glm5_next/hybrid_decoder.py','deepinfer/model_executor/models/glm5_next/decoder.py','deepinfer/model_executor/models/glm5_next/kda.py','deepinfer/model_executor/models/glm5_next/mhc.py','deepinfer/model_executor/models/glm5_next/text_boundary.py','deepinfer/hugging_face/modelslim910c.py','deepinfer/npu_ops/w8a8.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'status':'in_progress','architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'versions':versions,'NPU_execution':True,'tokens':[1,2,3],'layer_id':0,'scope':'real checkpoint embeddings/weights; new unsharded native factory vs previous native dense layer, no independent full-model oracle','source_sha256':{n:sha(root/n) for n in files},'config_sha256':sha(a.checkpoint/'config.json'),'index_sha256':sha(a.checkpoint/'quant_model_weights.safetensors.index.json'),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':{}}
def save():
 q=a.output_dir/'native-factory-result.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(q)
def snapshot(output,state):
 return {'output':output.detach().cpu().clone(),'recurrent':state.recurrent.detach().cpu().clone(),**{f'conv{i}':v.detach().cpu().clone() for i,v in enumerate(state.conv)}}
def compare(name,value,reference):
 if value.shape!=reference.shape or value.dtype!=reference.dtype:raise ValueError('tensor shape/dtype changed: '+name)
 x=value.float();y=reference.float();diff=x-y;rel=float(diff.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12));ma=float(diff.abs().max());bound=.03*float(y.abs().max())+1e-5
 passed=bool(torch.isfinite(x).all()) and rel<=.001 and ma<=bound
 report['checks'][name]={'shape':list(value.shape),'dtype':str(value.dtype),'relative_rmse':rel,'relative_rmse_limit':.001,'max_abs':ma,'max_abs_limit':bound,'passed':passed};save()
 if not passed:raise AssertionError('factory comparison failed: '+name)
save()
try:
 reader=ModelSlimCheckpoint(a.checkpoint)
 with torch.inference_mode():
  boundary=GlmTextBoundary.from_checkpoint(reader,'npu',max_unique_tokens=3,max_tokens=3)
  hidden=boundary.embed(torch.tensor(report['tokens'],dtype=torch.int64))
  previous=Glm5NextDenseDecoder.from_checkpoint(reader,0,'npu',kda_backend='ascendc')
  output,state=previous(hidden);torch_npu.npu.synchronize();assert state.tokens_seen==3
  expected=snapshot(output,state);del previous,output,state
  torch_npu.npu.empty_cache()
  new=load_native_decoder(reader,0,'npu',max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2)
  output,state=new(hidden);torch_npu.npu.synchronize();assert state.tokens_seen==3
  actual=snapshot(output,state)
  torch.save({'expected':expected,'actual':actual},a.output_dir/'native-factory-outputs.pt')
  for name in expected:compare(name,actual[name],expected[name])
  report['status']='passed'
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
