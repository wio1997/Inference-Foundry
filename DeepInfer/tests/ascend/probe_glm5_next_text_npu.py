# SPDX-License-Identifier: Apache-2.0
"""Future checkpoint-native single layer / text acceptance entry, UNVERIFIED.

Requires later user authorization, pinned runtime and exclusive resources.
No service/container/card control. Full text/DSA stages additionally require the
missing native_dsa binding; this entry never substitutes CPU callbacks. Golden
reference must come from an independently accepted matching model/config run.
"""
import argparse,datetime,hashlib,importlib.metadata,json,platform
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--stage',choices=('layer','text'),required=True)
p.add_argument('--layer-id',type=int,default=0)
p.add_argument('--token-ids',default='1,2,3')
p.add_argument('--vocab-start',type=int,default=0)
p.add_argument('--vocab-stop',type=int,default=32)
p.add_argument('--max-expert-checkpoint-bytes',type=int,required=True)
p.add_argument('--max-expert-resident-bytes',type=int,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--assignment',type=Path,required=True)
p.add_argument('--owner-identity',type=Path,required=True)
p.add_argument('--sfa-library',type=Path)
p.add_argument('--authorized-npu',action='store_true')
a=p.parse_args()
if not a.authorized_npu:p.error('later explicit user NPU authorization and resource ownership required')
from deepinfer.hugging_face.glm5_next_cann_probe import require_owner_authorization
require_owner_authorization(a.authorized_npu,json.loads(a.assignment.read_text()),json.loads(a.owner_identity.read_text()))
if platform.machine()!='aarch64':raise RuntimeError('aarch64 target required')
versions={n:importlib.metadata.version(n) for n in ('torch','torch-npu','flash-linear-attention-npu-a3')}
if versions!={'torch':'2.10.0+cpu','torch-npu':'2.10.0.post7','flash-linear-attention-npu-a3':'26.9.1'}:raise RuntimeError('pinned historical KDA tuple required; DSA/core ABI still unverified')
import torch
import torch_npu
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.native_model import load_native_decoder,load_native_text_runner
from deepinfer.model_executor.models.glm5_next.text_boundary import GlmTextBoundary
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState
from deepinfer.model_executor.models.glm5_next.cache import DSAState,state_from_aligned_prefix
c=Glm5NextConfig.from_checkpoint(a.checkpoint)
ids=torch.tensor([int(v) for v in a.token_ids.split(',')],dtype=torch.int64)
if not 1<=len(ids)<=8:raise ValueError('1..8 actual embedding token IDs required')
if not 0<=a.layer_id<len(c.layers):raise ValueError('backbone layer ID required')
# Resolve the missing native DSA integration BEFORE device selection or payload.
operations={}
if a.stage=='text' or c.layers[a.layer_id].attention=='deepseek_sparse_attention':
    try:from deepinfer.model_executor.models.glm5_next.native_dsa import make_dsa_operations
    except ImportError as e:raise RuntimeError('native_dsa.make_dsa_operations is not implemented; this stage is UNVERIFIED/unavailable') from e
    operations=make_dsa_operations(c,sfa_library=a.sfa_library)
reference=torch.load(a.reference,map_location='cpu',weights_only=True)
config_sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
# Index hash identifies shard mapping, not an attestation of all weight bytes.
index_sha=hashlib.sha256((a.checkpoint/'quant_model_weights.safetensors.index.json').read_bytes()).hexdigest()
expected={'config_sha256':config_sha,'checkpoint_index_sha256':index_sha,'stage':a.stage,'layer_id':a.layer_id,'token_ids':ids.tolist(),'vocab_start':a.vocab_start,'vocab_stop':a.vocab_stop}
if reference.get('metadata')!=expected:raise ValueError('reference metadata must match config/stage/layer/tokens/vocab exactly')
torch.set_num_threads(2);torch.manual_seed(5343)
torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)
a.output_dir.mkdir(parents=True,exist_ok=True)
source=Path(__file__).resolve().parents[2]
report={'status':'in_progress','NPU_execution':True,'architecture':platform.machine(),'versions':versions,'metadata':expected,'checks':{},'source_sha256':{str(n.relative_to(source)):hashlib.sha256(n.read_bytes()).hexdigest() for n in sorted((source/'deepinfer/model_executor/models/glm5_next').glob('*.py'))},'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'bounded actual native component/text correctness; no performance/vision/MTP/TP/EP claim'}
def save():
    target=a.output_dir/'text-native-result.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(target)
def tensors(state):
    result={}
    for i,s in state.items():
        if isinstance(s,DSAState):
            for n in ('latent','compressed','tail','tail_positions'):result[f'{i}/{n}']=getattr(s,n).detach().cpu().clone()
        else:
            result[f'{i}/recurrent']=s.recurrent.detach().cpu().clone()
            for j,v in enumerate(s.conv):result[f'{i}/conv{j}']=v.detach().cpu().clone()
    return result
save()
try:
    reader=ModelSlimCheckpoint(a.checkpoint)
    bounds=dict(max_expert_checkpoint_bytes=a.max_expert_checkpoint_bytes,max_expert_resident_bytes=a.max_expert_resident_bytes)
    with torch.inference_mode():
        if a.stage=='layer':
            layer=load_native_decoder(reader,a.layer_id,'npu',dsa_operation=operations.get(a.layer_id),**bounds)
            boundary=GlmTextBoundary.from_checkpoint(reader,'npu',max_tokens=8,max_unique_tokens=8)
            hidden=boundary.embed(ids)
            if a.layer_id: hidden=hidden[:,None,:].expand(-1,c.hf_config['text_config']['hc_mult'],-1).clone()
            output,state=layer(hidden)
            if state.tokens_seen!=len(ids):raise ValueError('layer did not advance exactly once per token')
            actual={'output':output.detach().cpu(),**tensors({a.layer_id:state})}
        else:
            runner=load_native_text_runner(reader,'npu',dsa_operations=operations,vocab_start=a.vocab_start,vocab_stop=a.vocab_stop,**bounds)
            states={}
            for i,layer in enumerate(runner.stack.layers):
                if c.layers[i].attention=='linear_attention':states[i]=layer.attention.initial_state()
                else:states[i]=state_from_aligned_prefix(torch.empty(0,512,dtype=torch.bfloat16,device='npu'),torch.empty(0,128,dtype=torch.bfloat16,device='npu'),runner.stack.layout)
            request=runner.open_request(MixedRequestState.capture('native-acceptance',c,runner.stack.layout,states))
            try:
                result=request.step(ids,expected_tokens_seen=0)
                snapshot=request.snapshot()
                if snapshot.tokens_seen!=len(ids) or any(s.tokens_seen!=len(ids) for s in snapshot.layers.values()):raise ValueError('text state token counts differ')
                actual={'output':result.hidden.detach().cpu(),'logits':result.logits.detach().cpu(),**tensors(snapshot.layers)}
            finally:request.close()
        torch_npu.npu.synchronize()
        torch.save({'metadata':expected,'tensors':actual},a.output_dir/'actual.pt')
        if set(actual)!=set(reference['tensors']):raise ValueError('reference tensor coverage differs')
        for name,value in actual.items():
            wanted=reference['tensors'][name]
            if value.shape!=wanted.shape or value.dtype!=wanted.dtype:raise ValueError('reference shape/dtype differs: '+name)
            if not value.is_floating_point():passed=torch.equal(value,wanted);metric={'exact':passed}
            else:
                diff=value.float()-wanted.float();rel=float(diff.square().mean().sqrt()/wanted.float().square().mean().sqrt().clamp_min(1e-12)) if value.numel() else 0.
                maximum=float(diff.abs().max()) if value.numel() else 0.;bound=.03*float(wanted.abs().max())+1e-5 if value.numel() else 1e-5
                passed=bool(torch.isfinite(value).all()) and rel<=.001 and maximum<=bound
                metric={'relative_rmse':rel,'limit':.001,'max_abs':maximum,'max_abs_limit':bound}
            report['checks'][name]={'passed':passed,**metric};save()
            if not passed:raise AssertionError('reference comparison failed: '+name)
    report['status']='passed'
except BaseException as e:report.update(status='failed',failure=repr(e));raise
finally:report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
