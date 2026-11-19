"""CPU independent BOS0 attention reference from pinned actual checkpoint.
Explicit BF16 boundaries and dynamicINT8 projection; no RoPE at position0.
This diagnoses one operator boundary, never fullmodelquality/performance.
"""
from pathlib import Path
import datetime,hashlib,json,math,os,time,torch
from safetensors import safe_open
from safetensors.torch import load_file
assert not torch.cuda.is_initialized();torch.set_num_threads(1)
R=Path('/task/client_ds41_g1122');OUT=R/'layer0_BOS_oracle_g1155';OUT.mkdir(exist_ok=False);M=Path('/models/ds41');start=time.monotonic();raw=(M/'config.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7';config=json.loads(raw).get('text_config');assert config['head_dim']==512 and config['num_attention_heads']==64 and config['o_groups']==8
index=json.loads((M/'quant_model_weights.safetensors.index.json').read_text())['weight_map'];desc=json.loads((M/'quant_model_description.json').read_text());weights={};pins={}
for short in ['wq_a.weight','q_norm.weight','wq_b.weight','wq_b.weight_scale','wq_b.weight_offset','wkv.weight','kv_norm.weight','attn_sink','wo_a.weight','wo_b.weight']:
 name='layers.0.attn.'+short
 with safe_open(M/index[name],framework='pt',device='cpu') as f:weights[short]=f.get_tensor(name).clone()
 w=weights[short];pins[short]={'checkpoint_name':name,'shard':index[name],'shape':list(w.shape),'dtype':str(w.dtype),'quant_description':desc.get(name),'raw_tensor_SHA256':hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()}
assert torch.count_nonzero(weights['wq_b.weight_offset'])==0,'Nonzero weight-offset semantics require source audit before oracle'
N=load_file(str(R/'native_layer_vectors_g1151/rank0_layer00.safetensors'));S=load_file('/task/sg_pd_g1147/layer_probe/G1149_same30/rank0_layer00.safetensors');assert torch.equal(N['input'],S['input']);x=N['input'][0:1].float();eps=config['rms_norm_eps']
def bf16(t):return t.to(torch.bfloat16).float()
def norm(t,w):return bf16(t*torch.rsqrt(t.square().mean(-1,keepdim=True)+eps)*w.float())
qa=bf16(torch.nn.functional.linear(x,weights['wq_a.weight'].float()));qr=norm(qa,weights['q_norm.weight']);kv=norm(bf16(torch.nn.functional.linear(x,weights['wkv.weight'].float())),weights['kv_norm.weight']);scale=qr.abs().amax(-1,keepdim=True)/127;qi=torch.round(qr/scale).clamp(-127,127)
# Integer GEMM accumulated in FP64 provides an exact integer dot for this bound.
acc=qi.double()@weights['wq_b.weight'].double().T;q=bf16((acc*scale.double()*weights['wq_b.weight_scale'].reshape(1,-1).double()).float()).reshape(64,512)
score=(q.double()*kv.double()).sum(-1)/math.sqrt(512);sink=weights['attn_sink'].double();woa=weights['wo_a.weight'].float().reshape(8,1024,4096);wob=weights['wo_b.weight'].float().reshape(5120,8,1024)
def metrics(a,b):
 a=a.float().flatten();b=b.float().flatten();d=b-a;return {'max_abs':float(d.abs().max()),'relative_L2':float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(a)),'RMSE':float(d.square().mean().sqrt()),'cosine':float(torch.nn.functional.cosine_similarity(a,b,dim=0))}
rows=[];candidates={}
for policy,sinkvalue in [('sink_raw',sink),('sink_times_softmax_scale',sink/math.sqrt(512)),('no_sink',torch.full_like(sink,-torch.inf))]:
 probability=torch.sigmoid(score-sinkvalue);o=bf16(probability[:,None]*kv.double()).reshape(8,4096);absorbed=bf16(torch.bmm(woa,o[:,:,None]).squeeze(-1));partials=bf16(torch.einsum('hgr,gr->gh',wob,absorbed));y=bf16(partials.sum(0));y_full=bf16(torch.nn.functional.linear(absorbed.reshape(1,-1),weights['wo_b.weight'].float())).flatten();candidates[policy]=y
 rows.append({'policy':policy,'stage_fp32sum_of_BF16_TPpartials_vs_native':metrics(N['output'][0],y),'stage_fp32sum_of_BF16_TPpartials_vs_SG':metrics(S['output'][0],y),'fullFP32finalGEMM_vs_native':metrics(N['output'][0],y_full),'fullFP32finalGEMM_vs_SG':metrics(S['output'][0],y_full),'sink_range':[float(sinkvalue.min()),float(sinkvalue.max())],'attention_probability_range':[float(probability.min()),float(probability.max())]})
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'torch':torch.__version__,'CPU_only':True,'position':0,'native_vs_SG':metrics(N['output'][0],S['output'][0]),'weight_tensor_pins':pins,'zero_weight_offset_verified':True,'activation_scale':float(scale[0,0]),'variants':rows,'scope':'Independent oneBOS-row formula withBF16stageboundaries, exactCPUintegerQ projection, causalSWAonlyselfKVplusoptionaldenominatorsink. Position0RoPEidentity, ratio0noindex/compressor/FFN. FinalTP reductionrounding andF32GEMMimplementation differ fromNPU; comparecloseness separately, nofullquality/solecause/performance acceptance.'};(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
