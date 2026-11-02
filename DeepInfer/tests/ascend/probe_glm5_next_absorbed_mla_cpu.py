# SPDX-License-Identifier: Apache-2.0
"""Actual two-head kv_b slices + synthetic queries/latents; CPU reference only."""
import argparse,ast,datetime,hashlib,json,os,platform
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.absorbed_mla import AbsorbedMLAContract,split_kv_b_cpu,absorbed_sparse_mla_cpu,inspect_sfa_source_contract

p=argparse.ArgumentParser(description=__doc__)
for n in ('checkpoint','sfa-reference','prior-evidence','output-dir'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);torch.manual_seed(5331);config=Glm5NextConfig.from_checkpoint(a.checkpoint);reader=ModelSlimCheckpoint(a.checkpoint,max_read_bytes=16*1024**2)
prior=json.loads(a.prior_evidence.read_text());config_sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest();assert prior['status']=='passed' and not prior['NPU_execution'] and prior['config_sha256']==config_sha
src=Path(__file__).resolve().parents[2];files=['deepinfer/model_executor/models/glm5_next/absorbed_mla.py','tests/ascend/test_glm5_next_absorbed_mla.py','tests/ascend/probe_glm5_next_absorbed_mla_cpu.py']
report={'status':'in_progress','native_round':31,'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'model_generation':False,'INT8_projection_executed':False,
    'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'reference_sha256':hashlib.sha256(a.sfa_reference.read_bytes()).hexdigest(),
    'prior_evidence_sha256':hashlib.sha256(a.prior_evidence.read_bytes()).hexdigest(),'payload_bytes':0,'per_read_bound_bytes':reader.max_read_bytes,'checks':{},'layers':{}}
def check(name,condition):
    if not condition:raise AssertionError(name)
    report['checks'][name]={'passed':True}
def np64(t):return t.float().numpy().astype(np.float64)
def rounded(v):return torch.tensor(v,dtype=torch.float64).bfloat16()
def close_bf16(name,actual,expected):
    expected=rounded(expected);f=expected.float();spacing=torch.maximum((torch.nextafter(expected,torch.full_like(expected,float('inf'))).float()-f).abs(),(torch.nextafter(expected,torch.full_like(expected,float('-inf'))).float()-f).abs())
    error=(actual.float()-f).abs();relative=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(f).clamp_min(1e-20))
    check(name,bool((error<=spacing+1e-5).all()) and relative<=.001)
    report['checks'][name].update(max_abs=float(error.max()) if error.numel() else 0.,relative_RMSE=relative,guard='CPU only:1BF16ULP+1e-5,relativeRMSE<=0.001')
def method(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
def installed_weight_views(tree,weight,c):
    body=[]
    for n in method(tree,'process_weights_after_loading').body:
        body.append(n)
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='replace_parameter' and isinstance(n.value.args[1],ast.Constant) and n.value.args[1].value=='W_UK_T':break
    def replace(obj,name,value,**kw):setattr(obj,name,value)
    ns={'torch':torch,'torch_npu':NS(npu_format_cast=lambda x,format:x),'ACL_FORMAT_FRACTAL_ND':2,'replace_parameter':replace,
        'self':NS(kv_b_proj=NS(weight=weight),kv_lora_rank=c.kv_rank,local_num_heads=c.heads,qk_nope_head_dim=c.qk_dim,v_head_dim=c.value_dim),'act_dtype':torch.bfloat16}
    exec(compile(ast.Module(body=body,type_ignores=[]),'extracted_weight_views','exec'),ns)
    return ns['self'].W_UK_T,ns['self'].W_UV
def installed_f32_method(tree,name,obj,x):
    n=method(tree,name);ns={'torch':torch,'torch_npu':NS(),'TRANSPOSE_BMM_MAX_SUPPORTED_DIM':65536}
    exec(compile(ast.Module(body=[n],type_ignores=[]),'extracted_f32_bmm','exec'),ns)
    return ns[name](obj,x)
try:
    source=a.sfa_reference.read_text();tree=ast.parse(source);report['source_contract']=inspect_sfa_source_contract(source)
    check('source_call_layout_and_write_order',report['source_contract']['source_only'] and not report['source_contract']['native_ABI_validated'])
    for label,old,new in [('scale','scale_value=scale','scale_value=scale * 2'),('layout','layout_kv="PA_BSND"','layout_kv="PA_BBND"'),('orientation','torch.bmm(q_nope, self.W_UK_T)','torch.bmm(self.W_UK_T, q_nope)')]:
        altered=source.replace(old,new);assert altered!=source
        try:inspect_sfa_source_contract(altered)
        except ValueError:check('source_drift_'+label,True)
        else:raise AssertionError('source drift accepted:'+label)
    heads=(0,config.mla_dimensions['num_attention_heads']-1);c=AbsorbedMLAContract.from_config(config,heads)
    check('actual_contract',(c.total_heads,c.head_ids,c.qk_dim,c.kv_rank,c.value_dim,c.scale,c.output_width)==(64,(0,63),256,512,256,1/16,2051))
    for layer in (3,43):
        name=f'model.language_model.layers.{layer}.self_attn.kv_b_proj.weight';path,offset,desc=reader.descriptor(name)
        check(f'{layer}/FLOAT_dense_manifest',reader.quant[name]=='FLOAT' and desc['dtype'] in ('BF16','F32') and desc['shape']==[64*512,512])
        payload=[];ledger=[]
        for head in heads:
            first=head*(c.qk_dim+c.value_dim);last=first+c.qk_dim+c.value_dim;t,dtype=reader.row_slice(name,first,last)
            raw=t.contiguous().view(torch.uint8).numpy().tobytes();size=len(raw);report['payload_bytes']+=size
            with path.open('rb') as stream:stream.seek(offset+desc['data_offsets'][0]+first*512*t.element_size());independent=stream.read(size)
            check(f'{layer}/head{head}_raw_bytes',raw==independent)
            ledger.append({'global_head':head,'rows':[first,last],'dtype':dtype,'bytes':size,'sha256':hashlib.sha256(raw).hexdigest()});payload.append(t.bfloat16())
        weight=torch.cat(payload);saved=weight.clone();weights=split_kv_b_cpu(weight,c);uk,uv=installed_weight_views(tree,weight,c)
        check(f'{layer}/source_weight_views',torch.equal(uk,weights.key_up) and torch.equal(uv,weights.value_up))
        q=(torch.randn(4,2,256)*.2).bfloat16();latent=(torch.randn(9,512)*.3).bfloat16();ids=torch.tensor([[0,-1,-1,-1],[3,0,2,-1],[8,2,6,0],[-1]*4],dtype=torch.int32);pos=torch.tensor([0,3,8,-999])
        q[-1]=float('nan');latent[7]=float('nan');latent_saved=latent.clone();q_saved=q.clone()
        out=absorbed_sparse_mla_cpu(q,latent,ids[:,None],pos,weights,sink_policy='absent')
        uk_np=np64(weight).reshape(2,512,512)[:,:256];uv_np=np64(weight).reshape(2,512,512)[:,256:].transpose(0,2,1)
        ql=rounded(np.einsum('thp,hpl->thl',np64(q[:3]),uk_np));close_bf16(f'{layer}/absorbed_query_np64',out.query_latent[:3],np64(ql))
        latent_out=np.zeros((4,2,512));lse=np.full((4,2),-np.inf);algebra_max=0.
        for row in range(3):
            selected=[i for i in ids[row].tolist() if i>=0];x=np64(latent)[selected];logits=np64(ql)[row]@x.T*c.scale;m=logits.max(1,keepdims=True);exp=np.exp(logits-m);probs=exp/exp.sum(1,keepdims=True)
            latent_out[row]=np64(rounded(probs@x));lse[row]=(m+np.log(exp.sum(1,keepdims=True)))[:,0]
            # Independent UNROUNDED absorption identity versus expanded K/V.
            full_query=np.einsum('hp,hpl->hl',np64(q[row]),uk_np);expanded_keys=np.einsum('sl,hpl->shp',x,uk_np)
            logits_expanded=np.einsum('hp,shp->hs',np64(q[row]),expanded_keys)*c.scale;logits_absorbed=full_query@x.T*c.scale
            algebra_max=max(algebra_max,float(np.max(np.abs(logits_expanded-logits_absorbed))))
        close_bf16(f'{layer}/latent_reduce_np64',out.attention_latent,latent_out)
        close_bf16(f'{layer}/value_projection_np64',out.values,np.einsum('thl,hlv->thv',latent_out,uv_np))
        np.testing.assert_allclose(np64(out.lse[:3]),lse[:3],atol=3e-5,rtol=1e-5);check(f'{layer}/lse_np64',True)
        check(f'{layer}/unrounded_logit_algebra',algebra_max<1e-10);report['checks'][f'{layer}/unrounded_logit_algebra']['max_abs']=algebra_max
        obj=NS(q_proj=lambda unused:(q[:3].float().flatten(1),None),local_num_heads=2,kv_lora_rank=512,qk_head_dim=256,qk_nope_head_dim=256,qk_rope_head_dim=0,W_UK_T=uk.float(),W_UV=uv.float(),v_head_dim=256)
        source_ql,_=installed_f32_method(tree,'_q_proj_and_k_up_proj',obj,None)
        close_bf16(f'{layer}/source_f32_query_bmm',out.query_latent[:3],source_ql.double().numpy())
        source_values=installed_f32_method(tree,'_v_up_proj',obj,out.attention_latent.float()).reshape(4,2,256)
        close_bf16(f'{layer}/source_f32_value_bmm',out.values,source_values.double().numpy())
        check(f'{layer}/padding_zero_and_lse',bool((out.values[-1]==0).all()) and bool(torch.isneginf(out.lse[-1]).all()))
        check(f'{layer}/private_inputs_unchanged',torch.equal(weight,saved) and torch.allclose(q,q_saved,equal_nan=True) and torch.allclose(latent,latent_saved,equal_nan=True))
        report['layers'][str(layer)]={'head_ids':heads,'payloads':ledger,'weight_runtime_bytes':weight.numel()*weight.element_size(),'query_shape':list(q.shape),'latent_shape':list(latent.shape),'values_shape':list(out.values.shape)}
    report['status']='passed'
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:
    a.output_dir.mkdir(parents=True,exist_ok=True);report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'absorbed-mla-result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
