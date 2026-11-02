# SPDX-License-Identifier: Apache-2.0
"""Actual contiguous FLOAT rank-head rows;bounded CPU projection/source checks."""
import argparse,ast,datetime,hashlib,json,os,platform,sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from test_glm5_next_sfa_inputs import fixture,prepare
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.sfa_inputs import NoPESFAContract
from deepinfer.model_executor.models.glm5_next.sfa_weights import load_rank_kv_b_cpu
from deepinfer.model_executor.models.glm5_next.absorbed_mla import absorbed_sparse_mla_cpu,inspect_sfa_source_contract
from deepinfer.model_executor.models.glm5_next.private_pages import gather_private_latent_cpu

p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','prior-weight-evidence','prior-packet-evidence','reference-source','output-dir'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);torch.manual_seed(5337);c=Glm5NextConfig.from_checkpoint(a.checkpoint);reader=ModelSlimCheckpoint(a.checkpoint,max_read_bytes=8*1024**2)
config_sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest();old=json.loads(a.prior_weight_evidence.read_text());packet_prior=json.loads(a.prior_packet_evidence.read_text())
assert old['status']==packet_prior['status']=='passed' and old['config_sha256']==packet_prior['config_sha256']==config_sha and not old['NPU_execution'] and not packet_prior['NPU_execution']
source=a.reference_source.read_text();source_sha=hashlib.sha256(a.reference_source.read_bytes()).hexdigest();assert source_sha==old['reference_sha256']==packet_prior['reference_source_sha256']
contract=NoPESFAContract.from_config(c,reference_source_sha256=source_sha);tree=ast.parse(source)
src=Path(__file__).resolve().parents[2];old_harness=src/'tests/ascend/probe_glm5_next_absorbed_mla_cpu.py';harness_tree=ast.parse(old_harness.read_text())
# Reuse only the three already-validated extraction helpers;never execute old CLI.
ns={'ast':ast,'torch':torch,'NS':NS};names={'method','installed_weight_views','installed_f32_method'}
helpers=[n for n in harness_tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in helpers}==names
exec(compile(ast.Module(body=helpers,type_ignores=[]),'reused_source_helpers','exec'),ns)
files=['deepinfer/model_executor/models/glm5_next/sfa_weights.py','tests/ascend/test_glm5_next_sfa_weights.py','tests/ascend/probe_glm5_next_sfa_weights_cpu.py']
r={'status':'in_progress','native_round':37,'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,'reference_source_sha256':source_sha,
 'prior_weight_evidence_sha256':hashlib.sha256(a.prior_weight_evidence.read_bytes()).hexdigest(),'prior_packet_evidence_sha256':hashlib.sha256(a.prior_packet_evidence.read_bytes()).hexdigest(),'reused_harness_sha256':hashlib.sha256(old_harness.read_bytes()).hexdigest(),
 'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'INT8_execution':False,'model_generation':False,'full_model_forward':False,
 'row_payload_bytes':0,'independent_raw_verification_bytes':0,'scope':'actual contiguous FLOAT KV-B only;synthetic query/cache/latent,CPU references;no INT8o_proj/TP collective','checks':{},'cases':[],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def check(name,value):
    if name in r['checks']:raise AssertionError('duplicatecheck '+name)
    if not value:raise AssertionError(name)
    r['checks'][name]={'passed':True}
def fp64(x):return x.float().numpy().astype(np.float64)
def close(name,value,expected):
    expected=torch.tensor(expected,dtype=torch.float64).bfloat16();f=expected.float();spacing=torch.maximum((torch.nextafter(expected,torch.full_like(expected,float('inf'))).float()-f).abs(),(torch.nextafter(expected,torch.full_like(expected,float('-inf'))).float()-f).abs());err=(value.float()-f).abs();relative=float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(f).clamp_min(1e-20))
    check(name,bool((err<=spacing+1e-5).all()) and relative<=.001);r['checks'][name].update(max_abs=float(err.max()) if err.numel() else 0.,relative_RMSE=relative,guard='CPU reference only:1BF16ULP+1e-5 andrelativeRMSE<=0.001;unchanged round31guard')
try:
    ledger=inspect_sfa_source_contract(source);check('pinned_source_roles_reused',ledger['source_only'] and not ledger['native_ABI_validated'])
    for layer,rank in ((3,0),(43,15)):
        tag=str(layer);plan,g,main,stage,_,indices,_=fixture(start=3,count=4,world=16,rank=rank);plan=replace(plan,layer_id=layer,contract=contract,revision=9)
        prepared=load_rank_kv_b_cpu(reader,c,layer,plan.heads,contract);m=prepared.manifest;r['row_payload_bytes']+=m.checkpoint_bytes
        path,header,desc=reader.descriptor(m.tensor_name);size=2 if desc['dtype']=='BF16' else 4;offset=header+desc['data_offsets'][0]+m.first_row*512*size
        with path.open('rb') as f:f.seek(offset);raw=f.read(m.checkpoint_bytes)
        r['independent_raw_verification_bytes']+=len(raw);check(tag+'/bounded_contiguous_raw_span_hash',len(raw)==m.checkpoint_bytes and hashlib.sha256(raw).hexdigest()==prepared.checkpoint_sha256)
        weight=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16 if desc['dtype']=='BF16' else torch.float32).reshape(m.stop_row-m.first_row,512).bfloat16()
        check(tag+'/runtime_BF16_rounding_hash',hashlib.sha256(weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==prepared.runtime_sha256)
        # Reuse historical head0/head63 payload digests without rereading those windows.
        overlap=[];head_bytes=512*512*size
        for record in old['layers'][tag]['payloads']:
            head=record['global_head']
            if head not in plan.heads.head_ids:continue
            first=(head-plan.heads.head_ids[0])*head_bytes;digest=hashlib.sha256(raw[first:first+head_bytes]).hexdigest()
            check(tag+'/historical_head'+str(head)+'_raw_hash_reused',record['dtype']==desc['dtype'] and digest==record['sha256']);overlap.append(head)
        check(tag+'/has_prior_overlap_anchor',len(overlap)==1)
        uk,uv=ns['installed_weight_views'](tree,weight,prepared.weights.contract)
        check(tag+'/installed_source_UK_UV_views_bitwise',torch.equal(uk,prepared.weights.key_up) and torch.equal(uv,prepared.weights.value_up))
        query=(torch.randn(4,4,256)*.2).bfloat16();latent=(torch.randn(4,4,512)*.125).bfloat16();query_saved=query.clone();latent_saved=latent.clone()
        ql=prepared.absorb_query_cpu(query,plan=plan);values=prepared.value_up_cpu(latent,plan=plan)
        npw=fp64(weight).reshape(4,512,512)
        close(tag+'/query_absorption_FP64',ql,np.einsum('thp,hpl->thl',fp64(query),npw[:,:256]))
        close(tag+'/value_up_FP64',values,np.einsum('thl,hlv->thv',fp64(latent),npw[:,256:].transpose(0,2,1)))
        obj=NS(q_proj=lambda unused:(query.float().flatten(1),None),local_num_heads=4,kv_lora_rank=512,qk_head_dim=256,qk_nope_head_dim=256,qk_rope_head_dim=0,W_UK_T=uk.float(),W_UV=uv.float(),v_head_dim=256)
        source_ql,_=ns['installed_f32_method'](tree,'_q_proj_and_k_up_proj',obj,None)
        source_values=ns['installed_f32_method'](tree,'_v_up_proj',obj,latent.float()).reshape(4,4,256)
        close(tag+'/installed_F32_query_BMM',ql,source_ql.double().numpy());close(tag+'/installed_F32_value_BMM',values,source_values.double().numpy())
        dense=gather_private_latent_cpu(main,stage,g,torch.arange(plan.candidate.end));positions=torch.arange(plan.candidate.start,plan.candidate.end)
        oracle=absorbed_sparse_mla_cpu(query,dense,indices,positions,prepared.weights,sink_policy='absent')
        check(tag+'/query_matches_absorbed_reference_bitwise',torch.equal(ql,oracle.query_latent))
        packet=prepare(plan,main,stage,ql,indices)
        check(tag+'/actual_weight_query_binds_CPU_SFA_packet',packet.identity==plan.identity and packet.scale==1/16 and packet.query.shape==(4,4,512) and not packet.native_ABI_validated)
        check(tag+'/value_up_consumes_attention_reference_bitwise',torch.equal(prepared.value_up_cpu(oracle.attention_latent,plan=plan),oracle.values))
        check(tag+'/query_latent_inputs_unchanged',torch.equal(query,query_saved) and torch.equal(latent,latent_saved))
        r['cases'].append({'layer':layer,'declared_TP':16,'rank':rank,'global_heads':plan.heads.head_ids,'rows':[m.first_row,m.stop_row],'source_dtype':m.source_dtype,'checkpoint_bytes':m.checkpoint_bytes,'runtime_bytes':m.runtime_bytes,'checkpoint_sha256':prepared.checkpoint_sha256,'runtime_sha256':prepared.runtime_sha256,'descriptor_sha256':m.descriptor_sha256,'prior_anchor_heads':overlap,'query_shape':list(query.shape),'absorbed_shape':list(ql.shape),'head_values_shape':list(values.shape)})
    check('exactly_two_bounded_rank_windows',len(r['cases'])==2 and r['row_payload_bytes']==4194304 and r['independent_raw_verification_bytes']==4194304)
    check('no_native_backend_module_imports',not any(n in sys.modules for n in ('torch_npu','fla_npu','_C_ascend')))
    r['status']='passed'
except BaseException as e:r.update(status='failed',failure=repr(e));raise
finally:
    r['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'sfa-weight-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
