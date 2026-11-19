"""CPU independent BOS0 attention reference from pinned actual checkpoint.
Explicit BF16 boundaries and dynamicINT8 projection; no RoPE at position0.
This diagnoses one operator boundary, never fullmodelquality/performance.
"""
from pathlib import Path
import datetime,hashlib,json,math,os,time,torch
from safetensors import safe_open
from safetensors.torch import load_file
assert not torch.cuda.is_initialized();torch.set_num_threads(1)
R=Path('/task/client_ds41_g1122');OUT=R/'layer0_BOS_components_g1172';OUT.mkdir(exist_ok=False);M=Path('/models/ds41');start=time.monotonic();raw=(M/'config.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7';config=json.loads(raw).get('text_config');assert config['head_dim']==512 and config['num_attention_heads']==64 and config['o_groups']==8
index=json.loads((M/'quant_model_weights.safetensors.index.json').read_text())['weight_map'];desc=json.loads((M/'quant_model_description.json').read_text());weights={};pins={}
for short in ['wq_a.weight','q_norm.weight','wq_b.weight','wq_b.weight_scale','wq_b.weight_offset','wkv.weight','kv_norm.weight','attn_sink','wo_a.weight','wo_b.weight']:
 name='layers.0.attn.'+short
 with safe_open(M/index[name],framework='pt',device='cpu') as f:weights[short]=f.get_tensor(name).clone()
 w=weights[short];pins[short]={'checkpoint_name':name,'shard':index[name],'shape':list(w.shape),'dtype':str(w.dtype),'quant_description':desc.get(name),'raw_tensor_SHA256':hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()}
assert torch.count_nonzero(weights['wq_b.weight_offset'])==0,'Nonzero weight-offset semantics require source audit before oracle'
N=load_file(str(R/'native_layer_vectors_g1151/rank0_layer00.safetensors'));S=load_file('/task/sg_pd_g1163/layer_probe/G1171_QKV_same30/rank0_layer00.safetensors');assert torch.equal(N['input'],S['input']);x=N['input'][0:1].float();eps=config['rms_norm_eps']
def bf16(t):return t.to(torch.bfloat16).float()
def norm(t,w):return bf16(t*torch.rsqrt(t.square().mean(-1,keepdim=True)+eps)*w.float())
qa=bf16(torch.nn.functional.linear(x,weights['wq_a.weight'].float()));qr=norm(qa,weights['q_norm.weight']);kv=norm(bf16(torch.nn.functional.linear(x,weights['wkv.weight'].float())),weights['kv_norm.weight']);scale=qr.abs().amax(-1,keepdim=True)/127;qi=torch.round(qr/scale).clamp(-127,127)
# Integer GEMM accumulated in FP64 provides an exact integer dot for this bound.
acc=qi.double()@weights['wq_b.weight'].double().T;q=bf16((acc*scale.double()*weights['wq_b.weight_scale'].reshape(1,-1).double()).float()).reshape(64,512)
score=(q.double()*kv.double()).sum(-1)/math.sqrt(512);sink=weights['attn_sink'].double();woa=weights['wo_a.weight'].float().reshape(8,1024,4096);wob=weights['wo_b.weight'].float().reshape(5120,8,1024)
def metrics(a,b):
 a=a.float().flatten();b=b.float().flatten();d=b-a;return {'max_abs':float(d.abs().max()),'relative_L2':float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(a)),'RMSE':float(d.square().mean().sqrt()),'cosine':float(torch.nn.functional.cosine_similarity(a,b,dim=0))}
from safetensors.torch import save_file
import re
folder=Path('/task/sg_pd_g1163/forward_probe/G1171_QKV_same30');layerfolder=Path('/task/sg_pd_g1163/layer_probe/G1171_QKV_same30')
worker_to_rank={}
for f in layerfolder.glob('rank*_layer00.json'):
 m=json.loads(f.read_text());worker_to_rank[m['pid']]=m['rank']
assert len(worker_to_rank)==8
rows=[];actual_outputs={};actual_qs={};actual_kvs={}
for f in sorted(folder.glob('*_ratio0_call00.pt')):
 snap=torch.load(f,map_location='cpu',weights_only=True);pid=snap['pid'];rank=worker_to_rank[pid];assert snap['layer_id']==0 and snap['ratio']==0 and snap['input_ids'][:30].tolist()==json.loads((layerfolder/f'rank{rank}_layer00.json').read_text())['input_ids']
 assert snap['q_rows'][0]==0 and int(snap['positions'][0])==0 and int(snap['input_ids'][0])==0
 valid=snap['original_indices'][0];valid=valid[valid>=0];assert valid.numel()==1
 loc=torch.where(snap['original_slots']==valid[0])[0];assert loc.numel()==1;ak=snap['original_values'][loc[0]].float();aq=snap['q'][0,:8].float();ao=snap['result'][0,:8].float();sink_actual=snap['sink'][:8].double();actual_outputs[rank]=ao;actual_qs[rank]=aq;actual_kvs[rank]=ak
 expectedq=q[rank*8:(rank+1)*8].float();expectedk=kv[0].float();actualscore=(aq.double()*ak.double()).sum(-1)*snap['scaling'];o_from_actual=bf16(torch.sigmoid(actualscore-sink_actual)[:,None]*ak.double())
 rows.append({'rank':rank,'workerpid':pid,'file':f.name,'q_vs_checkpoint_reference':metrics(expectedq,aq),'kv_vs_checkpoint_reference':metrics(expectedk,ak),'sink_raw_checkpoint_equal':torch.equal(sink_actual,sink[rank*8:(rank+1)*8]),'kernel_vs_independent_using_actual_QKV':metrics(o_from_actual,ao),'q_shape':list(snap['q'].shape),'q_real_head_count':8,'original_key_count_BOS0':1,'scaling':snap['scaling']})
assert len(rows)==8 and set(actual_outputs)==set(range(8))
def project(o):
 o=o.reshape(8,4096).float();a=bf16(torch.bmm(woa,o[:,:,None]).squeeze(-1));p=bf16(torch.einsum('hgr,gr->gh',wob,a));return bf16(p.sum(0))
ao=torch.cat([actual_outputs[i] for i in range(8)],dim=0);from_actual=project(ao)
ref_o=bf16(torch.sigmoid(score-sink)[:,None]*kv.double());from_reference=project(ref_o)
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'CPU_only':True,'zero_weight_offset_verified':True,'sameactualLayer0_input_native_and_SG':True,'stage_reference':'CPU checkpointBF16 q_a/qnorm, dynamicINT8 q_b exactintegeraccum, BF16 kv/norm, BOS0noRoPE','rank_rows':sorted(rows,key=lambda r:r['rank']),'final_projection_from_actual_SG_attentionO_vs_actual_SG_layer0':metrics(S['output'][0],from_actual),'final_projection_from_reference_attentionO_vs_native_layer0':metrics(N['output'][0],from_reference),'final_projection_from_reference_attentionO_vs_SG_layer0':metrics(S['output'][0],from_reference),'scope':'IndependentBOS0 samex/checkpoint stage diagnosis all8rankQ/K/kernelO thenCPUWOA/B, nativevsSGBF16TPsumrounding disclosed. Notfullmodelquality/solecause/performanceacceptance.'};(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');save_file({'reference_q':q,'reference_kv':kv,'actual_q':torch.cat([actual_qs[i] for i in range(8)]),'actual_kv':torch.stack([actual_kvs[i] for i in range(8)]),'actual_attentionO':ao,'reference_attentionO':ref_o,'projected_from_actualO':from_actual,'actual_SG_projected':S['output'][0]},str(OUT/'BOS_components.safetensors'));print(json.dumps(result))
