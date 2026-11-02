# SPDX-License-Identifier: Apache-2.0
"""Actual FLOAT decoder assembly with explicit attention/expert test doubles.

Not actual DSA/KDA arithmetic, expert FFN, model forward/generation or hardware
validation. Actual mHC/norm/router parameters exercise the composition seam.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import torch
from torch import nn
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout,state_from_aligned_prefix,append_dsa
from deepinfer.model_executor.models.glm5_next.kda import KDAState
from deepinfer.model_executor.models.glm5_next.moe import GlmMoERouter
from deepinfer.model_executor.models.glm5_next.expert_bank import SelectedExpertBank
from deepinfer.model_executor.models.glm5_next.hybrid_decoder import GlmMoEMLP,GlmHybridDecoder

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--mhc-evidence',type=Path,required=True)
p.add_argument('--moe-evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);torch.manual_seed(5321)
config=Glm5NextConfig.from_checkpoint(a.checkpoint);layout=DSACacheLayout.from_config(config,logical_block_size=128)
src=Path(__file__).resolve().parents[2];reference_digests={}
for evidence in (a.mhc_evidence,a.moe_evidence):
    prior=json.loads(evidence.read_text());assert prior['status']=='passed' and prior['NPU_execution'] is False
    assert prior['config_sha256']==hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
    for name,digest in prior['source_sha256'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest
    reference_digests[evidence.name]=hashlib.sha256(evidence.read_bytes()).hexdigest()


class Reader(ModelSlimCheckpoint):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.records={}
    def tensor(self,name):
        value,dtype=super().tensor(name)
        assert dtype in ('BF16','F32') and '.experts.' not in name and '.shared_experts.' not in name
        path,start,desc=self.descriptor(name);begin,end=desc['data_offsets']
        with path.open('rb') as f:f.seek(start+begin);raw=f.read(end-begin)
        assert value.contiguous().view(torch.uint8).numpy().tobytes()==raw
        self.records[name]={'shape':list(value.shape),'dtype':dtype,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        return value,dtype


class AffineBank(nn.Module):
    count=288
    def __init__(self):super().__init__();self.calls=0
    def forward_grouped(self,x,groups,*,group_list_type):
        assert group_list_type==0;self.calls+=1;out=torch.empty_like(x);start=0
        for expert,end in enumerate(groups.tolist()):
            if end>start:out[start:end]=(x[start:end].float()*(1+expert/288)*.125).bfloat16()
            start=end
        assert start==len(x);return out


class Shared(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x):self.calls+=1;return (x.float()*.125).bfloat16()


class AttentionStub(nn.Module):
    def __init__(self,kind):super().__init__();self.kind=kind;self.calls=0
    def forward(self,x,state):
        self.calls+=1
        if self.kind=='deepseek_sparse_attention':
            if state is None:state=state_from_aligned_prefix(x.new_empty((0,512)),x.new_empty((0,128)),layout)
            next_state=append_dsa(x[:,:512],x[:,:128].float(),torch.zeros(len(x),128),torch.zeros(4,128),state,layout,start_position=state.tokens_seen)
        else:
            linear=config.linear_attn_config;heads,dim,w=(linear[n] for n in ('num_heads','head_dim','short_conv_kernel_size'))
            if state is None:state=KDAState(tuple(torch.zeros(w-1,heads*dim,dtype=torch.bfloat16) for _ in range(3)),torch.zeros(heads,dim,dim),0)
            recurrent=state.recurrent.clone()
            for row in range(len(x)):recurrent[0,0,0]+=x[row,0].float()
            next_state=KDAState(tuple(t.clone() for t in state.conv),recurrent,state.tokens_seen+len(x))
        return (x.float()*.125).bfloat16(),next_state


reader=Reader(a.checkpoint,max_read_bytes=8*1024*1024)
report={'status':'in_progress','native_round':21,'NPU_execution':False,'actual_attention_executed':False,'actual_expert_FFN_executed':False,'model_generation':False,
        'architecture':platform.machine(),'torch':torch.__version__,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256':hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest(),'reused_evidence_sha256':reference_digests,
        'checks':{},'layers':{},'payloads':{},'per_read_bound_bytes':reader.max_read_bytes,
        'source_sha256':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in [
            'deepinfer/model_executor/models/glm5_next/hybrid_decoder.py','tests/ascend/test_glm5_next_hybrid_decoder.py','tests/ascend/probe_glm5_next_hybrid_decoder_cpu.py']}}


def check(name,x,y,*,atol=0.,rtol=0.):
    torch.testing.assert_close(x,y,atol=atol,rtol=rtol);delta=(x.float()-y.float()).abs()
    report['checks'][name]={'passed':True,'shape':list(x.shape),'max_abs':float(delta.max()) if delta.numel() else 0.,'atol':atol,'rtol':rtol}


try:
    for layer_id in (3,4):
        attention=AttentionStub(config.layers[layer_id].attention);router=GlmMoERouter.from_checkpoint(reader,layer_id)
        bank,shared=AffineBank(),Shared();experts=SelectedExpertBank(range(288),bank,num_experts=288,shared=shared)
        mlp=GlmMoEMLP(router,experts,hidden_size=4096,num_experts=288)
        layer=GlmHybridDecoder.from_checkpoint_components(reader,layer_id,attention=attention,mlp=mlp)
        captures=[]
        handle=router.register_forward_hook(lambda module,args,result:captures.append((args[0].clone(),result)))
        x=(torch.randn(5,4,4096)*.125).bfloat16();saved=x.clone()
        output,state=layer(x);check(f'layer{layer_id}_output_shape',torch.tensor(output.shape),torch.tensor([5,4,4096]))
        check(f'layer{layer_id}_input_unchanged',x,saved)
        check(f'layer{layer_id}_state_advance',torch.tensor(state.tokens_seen),torch.tensor(5))
        assert len(captures)==bank.calls==shared.calls==1
        hidden,routes=captures[0];assert hidden.dtype==torch.bfloat16 and hidden.shape==(5,4096)
        check(f'layer{layer_id}_route_weight_mass',routes.weights.sum(-1),torch.full((5,),2.5),atol=5e-7,rtol=0.)
        assert routes.expert_ids.shape==(5,8) and all(len(set(row))==8 for row in routes.expert_ids.tolist())
        # Repeat from a fresh request: real FLOAT mHC/router assembly, synthetic
        # attention/expert callbacks, no model or native arithmetic claim.
        again,other=layer(x);check(f'layer{layer_id}_fresh_request_equal',again,output)
        if config.layers[layer_id].attention=='deepseek_sparse_attention':
            check(f'layer{layer_id}_private_tail_repeat',other.tail,state.tail)
            assert other.tail.data_ptr()!=state.tail.data_ptr()
        else:
            check(f'layer{layer_id}_private_recurrent_repeat',other.recurrent,state.recurrent)
            assert other.recurrent.data_ptr()!=state.recurrent.data_ptr()
        before=(len(captures),bank.calls,shared.calls)
        empty,_=layer(x[:0],state);assert empty.shape==(0,4,4096)
        assert before==(len(captures),bank.calls,shared.calls)
        report['checks'][f'layer{layer_id}_empty_skips_MoE']={'passed':True}
        handle.remove()
        root=f'model.language_model.layers.{layer_id}.'
        names=[n for n in reader.records if n.startswith(root)]
        assert all(not n.startswith('model.language_model.layers.0.') for n in names)
        report['layers'][str(layer_id)]={'attention_kind':layer.attention_kind,'mlp_kind':layer.mlp_kind,'tokens':5,'streams':4,'hidden':4096,
            'route_ids':routes.expert_ids.tolist(),'router_calls_per_nonempty_step':1,'routed_stub_calls_per_step':1,'shared_stub_calls_per_step':1,
            'FLOAT_payload_names':names,'attention':'explicit non-model test double','experts':'synthetic affine grouped callback; no expert weights'}
    report['payloads']=reader.records;report['unique_FLOAT_payload_bytes']=sum(v['bytes'] for v in reader.records.values())
    assert 'torch_npu' not in sys.modules
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'cpu-hybrid-decoder-result.json').write_text(json.dumps(report,indent=2)+'\n')
