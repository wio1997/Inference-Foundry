# SPDX-License-Identifier: Apache-2.0
"""Actual config/prior receipts, synthetic FLOAT chain and projected decoder.

No checkpoint weight/channel/header reads, INT8 math, native import or TP sum.
"""
import argparse,datetime,hashlib,json,os,platform,sys
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
from test_glm5_next_rank_dsa import data,attend,run,digest,MLA,Indexer
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.models.glm5_next.sfa_inputs import NoPESFAContract
from deepinfer.model_executor.models.glm5_next.rank_dsa import execute_rank_dsa_cpu,CPUProjectedDSAOperation
from deepinfer.model_executor.models.glm5_next.absorbed_mla import absorbed_sparse_mla_cpu
from deepinfer.model_executor.models.glm5_next.private_pages import gather_private_latent_cpu
from deepinfer.model_executor.models.glm5_next.hybrid_decoder import PreparedDSAAttention,GlmHybridDecoder
from deepinfer.model_executor.models.glm5_next.cache import state_from_aligned_prefix

p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','evidence-root','output-dir'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'));torch.set_num_threads(2)
sha=lambda b:hashlib.sha256(b).hexdigest();config_sha=sha((a.checkpoint/'config.json').read_bytes());c=Glm5NextConfig.from_checkpoint(a.checkpoint)
receipts={str(i):json.loads((a.evidence_root/f'round{i}'/n).read_text()) for i,n in [(37,'sfa-weight-result.json'),(38,'oproj-payload-result.json'),(39,'oproj-execution-result.json')]}
assert all(v['status']=='passed' and v['config_sha256']==config_sha and not v['NPU_execution'] for v in receipts.values())
source_contract=NoPESFAContract.from_config(c,reference_source_sha256=receipts['37']['reference_source_sha256']);root=Path(__file__).resolve().parents[2]
files=['deepinfer/model_executor/models/glm5_next/rank_dsa.py','deepinfer/model_executor/models/glm5_next/hybrid_decoder.py','tests/ascend/test_glm5_next_rank_dsa.py','tests/ascend/probe_glm5_next_rank_dsa_cpu.py']
r={'status':'in_progress','native_round':40,'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,'reference_source_sha256':source_contract.reference_source_sha256,'prior_receipts_sha256':{str(i):sha((a.evidence_root/f'round{i}'/n).read_bytes()) for i,n in [(37,'sfa-weight-result.json'),(38,'oproj-payload-result.json'),(39,'oproj-execution-result.json')]},'source_sha256':{n:sha((root/n).read_bytes()) for n in files},'weight_channel_or_shard_payload_bytes':0,'synthetic_weights_states_projections_and_mHC':True,'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'INT8_GEMM_or_dequantization':False,'activation_quantization':False,'TP_collective':False,'model_generation':False,'checks':{},'cases':[],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def check(name,ok):
    assert name not in r['checks']
    if not ok:raise AssertionError(name)
    r['checks'][name]={'passed':True}
def actual_contract(d):
    p,g,raw,stage,ids,kvb,oproj,q=d;p=replace(p,contract=source_contract)
    kvb=replace(kvb,manifest=replace(kvb.manifest,contract=source_contract));oproj=replace(oproj,manifest=replace(oproj.manifest,contract=source_contract))
    return p,g,raw,stage,ids,kvb,oproj,q
try:
    check('actual_45_layers_11_DSA',len(c.layers)==45 and tuple(v.layer_id for v in c.layers if v.attention=='deepseek_sparse_attention')==tuple(range(3,44,4)))
    check('actual_NoPE_dimensions_and_source_scale',(c.hidden_size,c.mla_dimensions['num_attention_heads'],c.mla_dimensions['qk_nope_head_dim'],c.mla_dimensions['kv_lora_rank'],c.mla_dimensions['v_head_dim'],source_contract.scale)==(4096,64,256,512,256,1/16))
    for rank,layer in ((0,3),(15,43)):
        d=actual_contract(data(rank,layer));p,g,raw,stage,ids,kvb,oproj,q=d;tag=str(layer);old=next(v for v in receipts['38']['cases'] if v['layer']==layer);oldkv=next(v for v in receipts['37']['cases'] if v['layer']==layer)
        check(tag+'/prior_actual_KVB_Oproj_head_metadata',p.heads.head_ids==tuple(old['heads'])==tuple(oldkv['global_heads']) and (oproj.manifest.first_column,oproj.manifest.stop_column)==tuple(old['input_columns']))
        logical=gather_private_latent_cpu(raw,stage,g,torch.arange(p.candidate.end));oracle=absorbed_sparse_mla_cpu(q,logical,ids,torch.arange(p.candidate.start,p.candidate.end),kvb.weights,sink_policy='absent');events=[];before=raw.clone();before_stage=stage.main.clone()
        def attention(packet):
            events.append('attention');check(tag+'/packet_query_matches_separate_absorbed_reference',torch.equal(packet.query,oracle.query_latent) and packet.identity==p.identity)
            kw=packet.source_kwargs_cpu();check(tag+'/packet_NoPE_roles',kw['key'] is kw['value'] and kw['query_rope'] is None and kw['key_rope'] is None and kw['scale_value']==1/16 and kw['actual_seq_lengths_query'].tolist()==[3] and kw['actual_seq_lengths_kv'].tolist()==[6] and not packet.native_ABI_validated)
            result=attend(packet);check(tag+'/latent_matches_separate_reference',torch.equal(result.latent,oracle.attention_latent));return result
        def apply(x,w,s,*,spec):
            events.append('rank_apply');check(tag+'/UV_values_and_Oproj_input_order_match',torch.equal(x,oracle.values.reshape(3,1024)) and w.dtype==torch.int8 and w.shape==(4096,1024) and s.shape==(4096,) and spec.identity==p.identity and spec.input_columns==tuple(old['input_columns']));return torch.full((3,4096),rank+1,dtype=torch.bfloat16)
        def reduce(y,**kw):events.append('sum_stub');return y+5
        out=run(d,attention=attention,rank_apply=apply,reduce_sum=reduce)
        check(tag+'/one_attention_apply_sum_and_projected_output',events==['attention','rank_apply','sum_stub'] and out.identity==p.identity and out.hidden.shape==(3,4096) and bool((out.hidden==rank+6).all()))
        check(tag+'/candidate_source_backing_unchanged',torch.equal(raw,before) and torch.equal(stage.main,before_stage))
        r['cases'].append({'layer':layer,'rank':rank,'heads':p.heads.head_ids,'input_columns':old['input_columns'],'prior_actual_weight_sha256':old['weight_sha256'],'executed_data':'synthetic FLOAT KV-B/states/query and I8 -7/BF16ones payload;NO actual checkpoint math','attention':'bounded CPU FP32 softmax/BF16 latent callback','INT8_backend_and_sum':'markers only','reference_compare':'separate existing absorbed CPU FLOAT reference;bitwise'})
    d=actual_contract(data(start=0,count=4));p,g,raw,stage,ids,kvb,oproj,_=d;mla=MLA();indexer=Indexer(g.layout);events=[]
    state=state_from_aligned_prefix(torch.empty(0,512,dtype=torch.bfloat16),torch.empty(0,128,dtype=torch.bfloat16),g.layout);following=state_from_aligned_prefix(torch.zeros(4,512,dtype=torch.bfloat16),torch.zeros(1,128,dtype=torch.bfloat16),g.layout)
    def operation(step,working,*,plan,expected_identity):
        events.append('operation');working.tail.fill_(99)
        def attention(packet):events.append('attention');return attend(packet)
        def apply(x,w,s,**kw):events.append('rank_apply');return torch.full((4,4096),2,dtype=torch.bfloat16)
        def reduce(y,**kw):events.append('sum_stub');return y+1
        result=execute_rank_dsa_cpu(step.mla.query[:,plan.heads.head_ids].contiguous(),raw,stage,ids,plan=plan,expected_identity=expected_identity,kv_b=kvb,o_proj=oproj,expected_runtime_scale_sha256=digest(oproj.weight_scale),attention=attention,rank_apply=apply,reduce_sum=reduce,quantization_policy='rank_local_dynamic_int8',bias_policy='absent',index_namespace='logical_token')
        return result,following
    wrapped=CPUProjectedDSAOperation(p,p.identity,operation);attention=PreparedDSAAttention(mla,indexer,lambda *a:None)
    class HC(nn.Module):
        streams=4;hidden_size=4096
        def pre(self,residual):return residual.float().mean(1).bfloat16(),None
        def post(self,value,residual,mix):return residual+value[:,None,:]
    class FFN(nn.Module):
        def forward(self,x):events.append('FFN_stub');return torch.zeros_like(x)
    layer=GlmHybridDecoder(layer_id=3,attention_kind='deepseek_sparse_attention',mlp_kind='dense',attn_hc=HC(),ffn_hc=HC(),input_norm=torch.ones(4096,dtype=torch.bfloat16),ffn_norm=torch.ones(4096,dtype=torch.bfloat16),attention=attention,mlp=FFN())
    # layer3 is actually MoE; this FFN injection checks output/control flow only.
    # No claim of configured FFN coverage or real mHC/decoder arithmetic.
    out,next_state=layer(torch.zeros(4,4,4096,dtype=torch.bfloat16),state,dsa_operation=wrapped)
    check('projected_hybrid_output_actual_hidden_stream_geometry',out.shape==(4,4,4096) and bool((out==3).all()))
    check('projected_hybrid_no_second_Oproj',mla.finishes==0 and mla.projects==1 and indexer.calls==1 and events==['operation','attention','rank_apply','sum_stub','FFN_stub'])
    check('projected_hybrid_state_isolation_and_advance',state.tokens_seen==0 and not bool(state.tail.any()) and next_state.tokens_seen==4 and next_state.latent.data_ptr()!=following.latent.data_ptr())
    check('no_native_modules_imported',not any(n in sys.modules for n in ('torch_npu','fla_npu','_C_ascend')))
    r['status']='passed'
except BaseException as exc:r.update(status='failed',failure=repr(exc));raise
finally:
    r['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'rank-dsa-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
