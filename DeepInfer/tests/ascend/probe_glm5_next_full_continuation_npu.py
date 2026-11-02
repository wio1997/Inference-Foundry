# SPDX-License-Identifier: Apache-2.0
"""Actual45-layer full-vocabulary prefill/continuation consistency on one die.

Compare two-token prefill against two single-token calls and prior independent
request execution of the same implementation. Not an external accuracy oracle.
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
from deepinfer.model_executor.models.glm5_next.cache import DSAState,state_from_aligned_prefix
from deepinfer.model_executor.models.glm5_next.native_model import load_native_text_runner
from deepinfer.model_executor.models.glm5_next.native_dsa import make_dsa_operations
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState

torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);a.output_dir.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['tests/ascend/probe_glm5_next_full_continuation_npu.py','deepinfer/model_executor/models/glm5_next/native_model.py','deepinfer/model_executor/models/glm5_next/native_dsa.py','deepinfer/model_executor/models/glm5_next/hybrid_decoder.py','deepinfer/model_executor/models/glm5_next/expert_bank.py','deepinfer/model_executor/models/glm5_next/moe.py','deepinfer/model_executor/models/glm5_next/text_request.py','deepinfer/npu_ops/w8a8.py']
report={'status':'in_progress','architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'versions':versions,'NPU_execution':True,'library_sha256':sha(a.library),'source_sha256':{n:sha(root/n) for n in names},'config_sha256':sha(a.checkpoint/'config.json'),'index_sha256':sha(a.checkpoint/'quant_model_weights.safetensors.index.json'),'prior_sha256':sha(a.prior),'scope':'actual45-layer text, two-token prefill vs single-token continuation/fresh request and previous same-implementation logits; no external accuracy reference','checks':{},'events':[],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
 q=a.output_dir/'full-continuation-result.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(q)
def event(name,**extra):report['events'].append({'name':name,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra});save();print(name,extra,flush=True)
def compare(name,x,y,limit=.001):
 x=x.detach().float().cpu();y=y.detach().float().cpu()
 if x.shape!=y.shape or not bool(torch.isfinite(x).all()):raise ValueError('comparison shape/finite differs: '+name)
 d=x-y;rel=float(d.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12)) if d.numel() else 0.;ma=float(d.abs().max()) if d.numel() else 0.;bound=.03*float(y.abs().max())+1e-5 if y.numel() else 1e-5
 passed=rel<=limit and ma<=bound;report['checks'][name]={'shape':list(x.shape),'relative_rmse':rel,'relative_rmse_limit':limit,'max_abs':ma,'max_abs_limit':bound,'passed':passed}
def state_tensors(state):
 if isinstance(state,DSAState):return {name:getattr(state,name) for name in ('latent','compressed','tail','tail_positions')}
 return {'recurrent':state.recurrent,**{f'conv{i}':v for i,v in enumerate(state.conv)}}
save()
try:
 prior=torch.load(a.prior,map_location='cpu',weights_only=True);ids=prior['input_and_generated_ids'][:2]
 if len(ids)!=2:raise ValueError('two prior input IDs required')
 config=Glm5NextConfig.from_checkpoint(a.checkpoint)
 if len(config.layers)!=45:raise ValueError('actual45-layer config required')
 reader=ModelSlimCheckpoint(a.checkpoint,max_read_bytes=2*1024**3);operations=make_dsa_operations(config,sfa_library=a.library);event('start_actual_model_load',tokens=ids)
 with torch.inference_mode():
  runner=load_native_text_runner(reader,'npu',dsa_operations=operations,expert_policy='selected_per_token',max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2,vocab_start=0,vocab_stop=config.vocab_size,max_head_rows=config.vocab_size,max_tokens=2,max_context_tokens=4,logical_block_size=128)
  captured={'prefill':{},'segmented':{}};phase=['prefill'];hooks=[]
  for i,layer in enumerate(runner.stack.layers):
   def capture(module,args,output,layer_id=i):captured[phase[0]].setdefault(layer_id,[]).append(output[0].detach().clone())
   hooks.append(layer.register_forward_hook(capture))
  states={}
  for i,layer in enumerate(runner.stack.layers):
   states[i]=layer.attention.initial_state() if config.layers[i].attention=='linear_attention' else state_from_aligned_prefix(torch.empty((0,512),dtype=torch.bfloat16,device='npu'),torch.empty((0,128),dtype=torch.bfloat16,device='npu'),runner.stack.layout)
  event('start_two_token_prefill');q=runner.open_request(MixedRequestState.capture('prefill',config,runner.stack.layout,states))
  try:whole=q.step(torch.tensor(ids,dtype=torch.int64),expected_tokens_seen=0);whole_state=q.snapshot()
  finally:q.close()
  phase[0]='segmented';event('start_two_single_token_steps');q=runner.open_request(MixedRequestState.capture('segmented',config,runner.stack.layout,states))
  try:first=q.step(torch.tensor(ids[:1],dtype=torch.int64),expected_tokens_seen=0);last=q.step(torch.tensor(ids[1:],dtype=torch.int64),expected_tokens_seen=1);seg_state=q.snapshot()
  finally:q.close()
  torch_npu.npu.synchronize()
  raw_layers={'prefill':{},'segmented':{}}
  for i in range(45):
   full_layer=torch.cat(captured['prefill'][i]);seg_layer=torch.cat(captured['segmented'][i])
   raw_layers['prefill'][i]=full_layer.cpu();raw_layers['segmented'][i]=seg_layer.cpu()
   compare(f'layer{i}_residual_streams',full_layer,seg_layer)
  torch.save(raw_layers,a.output_dir/'full-continuation-layer-outputs.pt')
  for hook in hooks:hook.remove()
  compare('full_vocab_logits_prefill_vs_segmented',whole.logits,torch.cat((first.logits,last.logits)));compare('final_hidden_prefill_vs_segmented',whole.hidden,torch.cat((first.hidden,last.hidden)));compare('logits_vs_prior_same_implementation',whole.logits,torch.cat(prior['full_vocab_logits']))
  if whole_state.tokens_seen!=2 or seg_state.tokens_seen!=2:raise ValueError('request token count differs')
  for i in range(45):
   for name,value in state_tensors(whole_state.layers[i]).items():compare(f'layer{i}_{name}',value,state_tensors(seg_state.layers[i])[name])
  whole_ids=whole.logits.argmax(dim=-1).cpu().tolist();seg_ids=torch.cat((first.logits,last.logits)).argmax(dim=-1).cpu().tolist()
  report.update(input_ids=ids,prefill_argmax_ids=whole_ids,segmented_argmax_ids=seg_ids,all_state_layers_compared=45)
  if whole_ids!=seg_ids:raise AssertionError('greedy token IDs differ')
  torch.save({'input_ids':ids,'prefill_logits':whole.logits.cpu(),'segmented_logits':torch.cat((first.logits,last.logits)).cpu(),'prefill_hidden':whole.hidden.cpu(),'segmented_hidden':torch.cat((first.hidden,last.hidden)).cpu()},a.output_dir/'full-continuation-outputs.pt')
  failed=[k for k,v in report['checks'].items() if not v['passed']];save()
  if failed:raise AssertionError('continuation comparisons failed: '+', '.join(failed))
 report['status']='passed';report['accuracy_accepted']=False
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
