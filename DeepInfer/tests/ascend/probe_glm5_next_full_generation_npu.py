# SPDX-License-Identifier: Apache-2.0
"""Bounded actual 45-layer text generation on one authorized Ascend die.

Uses tokenwise short-context DSA and per-token selected expert residency. Two
full-vocabulary greedy choices are an execution smoke test, not an independent
accuracy, multimodal/MTP, serving, long-context, TP/EP or performance claim.
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
from deepinfer.model_executor.models.glm5_next.cache import state_from_aligned_prefix
from deepinfer.model_executor.models.glm5_next.native_model import load_native_text_runner
from deepinfer.model_executor.models.glm5_next.native_dsa import make_dsa_operations
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState

torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);a.output_dir.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['tests/ascend/probe_glm5_next_full_generation_npu.py','deepinfer/model_executor/models/glm5_next/native_model.py','deepinfer/model_executor/models/glm5_next/native_dsa.py','deepinfer/model_executor/models/glm5_next/hybrid_decoder.py','deepinfer/model_executor/models/glm5_next/decoder.py','deepinfer/model_executor/models/glm5_next/kda.py','deepinfer/model_executor/models/glm5_next/moe.py','deepinfer/model_executor/models/glm5_next/expert_bank.py','deepinfer/model_executor/models/glm5_next/cache.py','deepinfer/model_executor/models/glm5_next/text_boundary.py','deepinfer/model_executor/models/glm5_next/text_request.py','deepinfer/model_executor/models/glm5_next/request_state.py','deepinfer/npu_ops/w8a8.py']
report={'status':'in_progress','architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'versions':versions,'NPU_execution':True,'library_sha256':sha(a.library),'source_sha256':{n:sha(root/n) for n in names},'config_sha256':sha(a.checkpoint/'config.json'),'index_sha256':sha(a.checkpoint/'quant_model_weights.safetensors.index.json'),'quant_sidecar_sha256':sha(a.checkpoint/'quant_model_description.json'),'scope':'actual 45-layer text, one prompt token and two full-vocabulary greedy choices, selected top8 plus shared weights per token; no independent full-model reference','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'events':[],'token_ids':[]}
def save():
 q=a.output_dir/'full-generation-result.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(q)
def event(name,**extra):report['events'].append({'name':name,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra});save();print(name,extra,flush=True)
save()
try:
 config=Glm5NextConfig.from_checkpoint(a.checkpoint)
 if len(config.layers)!=45 or sum(s.attention=='deepseek_sparse_attention' for s in config.layers)!=11:raise ValueError('expected actual45-layer mixed text configuration')
 reader=ModelSlimCheckpoint(a.checkpoint,max_read_bytes=2*1024**3);operations=make_dsa_operations(config,sfa_library=a.library)
 event('start_actual_model_load',layers=len(config.layers),vocab_size=config.vocab_size)
 with torch.inference_mode():
  runner=load_native_text_runner(reader,'npu',dsa_operations=operations,expert_policy='selected_per_token',max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2,vocab_start=0,vocab_stop=config.vocab_size,max_head_rows=config.vocab_size,max_tokens=1,max_context_tokens=4,logical_block_size=128)
  torch_npu.npu.synchronize();free,total=torch_npu.npu.mem_get_info();event('actual_model_loaded',free_device_bytes=int(free),total_device_bytes=int(total))
  if free<4*1024**3:raise RuntimeError('insufficient free device memory for bounded selected expert step')
  states={}
  for i,layer in enumerate(runner.stack.layers):
   if config.layers[i].attention=='linear_attention':states[i]=layer.attention.initial_state()
   else:states[i]=state_from_aligned_prefix(torch.empty((0,512),dtype=torch.bfloat16,device='npu'),torch.empty((0,128),dtype=torch.bfloat16,device='npu'),runner.stack.layout)
  request=runner.open_request(MixedRequestState.capture('glm53-native-smoke',config,runner.stack.layout,states));logits=[];inputs=[1]
  try:
   for step in range(2):
    event('start_greedy_step',step=step,input_token=inputs[-1])
    result=request.step(torch.tensor([inputs[-1]],dtype=torch.int64),expected_tokens_seen=step)
    torch_npu.npu.synchronize();scores=result.logits.detach().float().cpu()
    if scores.shape!=(1,config.vocab_size) or not bool(torch.isfinite(scores).all()):raise ValueError('finite full-vocabulary logits required')
    chosen=int(scores.argmax(dim=-1).item());inputs.append(chosen);logits.append(scores)
    snapshot=request.snapshot()
    if snapshot.tokens_seen!=step+1 or any(s.tokens_seen!=step+1 for s in snapshot.layers.values()):raise ValueError('45-layer state token counts differ')
    top_values,top_ids=torch.topk(scores,5,dim=-1);event('finish_greedy_step',step=step,output_token=chosen,top5_ids=top_ids[0].tolist(),top5_logits=top_values[0].tolist(),state_tokens=snapshot.tokens_seen)
   report['token_ids']=inputs;torch.save({'input_and_generated_ids':inputs,'full_vocab_logits':logits},a.output_dir/'full-generation-outputs.pt')
  finally:request.close()
 report['status']='observed';report['accuracy_accepted']=False
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
