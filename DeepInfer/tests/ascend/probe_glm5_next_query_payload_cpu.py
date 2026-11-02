# SPDX-License-Identifier: Apache-2.0
"""Two actual signed q_b row windows/full-channel checks; CPU callback stubs.

AST source-role checks only,no imported vendor code/INT8 math/native execution.
"""
import argparse,datetime,hashlib,json,os,platform,sys
from dataclasses import replace
from pathlib import Path
import torch
from test_glm5_next_sfa_inputs import fixture
from test_glm5_next_sfa_weights import components
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.sfa_inputs import NoPESFAContract
from deepinfer.model_executor.models.glm5_next.mhc import rms_norm
from deepinfer.model_executor.models.glm5_next.query_payload import load_rank_qb_cpu,inspect_qb_sources

p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','evidence-root','reference-dir','output-dir'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'));torch.set_num_threads(2)
sha=lambda b:hashlib.sha256(b).hexdigest();root=Path(__file__).resolve().parents[2];config_sha=sha((a.checkpoint/'config.json').read_bytes());c=Glm5NextConfig.from_checkpoint(a.checkpoint)
kvb_prior_path=a.evidence_root/'round37/sfa-weight-result.json';kvb_prior=json.loads(kvb_prior_path.read_text());oproj_prior=json.loads((a.evidence_root/'round38/oproj-payload-result.json').read_text());assert kvb_prior['status']==oproj_prior['status']=='passed' and kvb_prior['config_sha256']==oproj_prior['config_sha256']==config_sha
contract=NoPESFAContract.from_config(c,reference_source_sha256=kvb_prior['reference_source_sha256']);ref_hashes=json.loads((a.reference_dir/'sha256.json').read_text());assert all(sha((a.reference_dir/n).read_bytes())==h for n,h in ref_hashes.items());sources={n:(a.reference_dir/n).read_text() for n in ref_hashes}
class TrackedReader(ModelSlimCheckpoint):
    def __init__(self,*args,**kw):super().__init__(*args,**kw);self.reads=[];self.channels={}
    def row_slice(self,name,first,last):
        assert name.endswith('.q_b_proj.weight');value,dtype=super().row_slice(name,first,last);self.reads.append({'kind':'weight_rows','name':name,'rows':[first,last],'bytes':value.numel()});return value,dtype
    def tensor(self,name):
        assert name.endswith(('.weight_scale','.weight_offset'));value,dtype=super().tensor(name);self.reads.append({'kind':'full_channel','name':name,'bytes':value.numel()*4});self.channels[name]=value;return value,dtype
    def forbidden(*args,**kw):raise AssertionError('native/full matrix path forbidden')
    matrix=materialize=load_linear=forbidden
reader=TrackedReader(a.checkpoint,max_read_bytes=4*1024**2)
files=['deepinfer/model_executor/models/glm5_next/query_payload.py','tests/ascend/test_glm5_next_query_payload.py','tests/ascend/probe_glm5_next_query_payload_cpu.py']
r={'status':'in_progress','native_round':42,'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,'reference_source_sha256':contract.reference_source_sha256,'reference_sha256':ref_hashes,'source_sha256':{n:sha((root/n).read_bytes()) for n in files},'prior_KVB_evidence_sha256':sha(kvb_prior_path.read_bytes()),'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'INT8_GEMM_or_dequantization':False,'activation_quantization':False,'TP_collective':False,'model_generation':False,'full_weight_matrix_read_or_allocated':False,'primary_weight_bytes':0,'primary_channel_bytes':0,'independent_verification_bytes':0,'checks':{},'cases':[],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def check(name,ok):
    assert name not in r['checks']
    if not ok:raise AssertionError(name)
    r['checks'][name]={'passed':True}
def raw_part(name,first=None,last=None):
    path,start,d=reader.descriptor(name);begin,stop=d['data_offsets'];size=stop-begin
    if first is not None:begin+=first*d['shape'][1];size=(last-first)*d['shape'][1]
    with path.open('rb') as f:f.seek(start+begin);raw=f.read(size)
    assert len(raw)==size;r['independent_verification_bytes']+=size;return raw
try:
    ledger=inspect_qb_sources(sources);r['source_contract']=ledger;check('source_GLMB_columns_fullqr_replicated_A_norm_parameter_axes',ledger['source_only'] and not ledger['native_ABI_validated'])
    for name,file,old,new in [('q_b_type','glm_attention.py','self.q_b_proj = ColumnParallelLinear(','self.q_b_proj = ReplicatedLinear('),('fused_tp','deepseek_v2.py','disable_tp=True','disable_tp=False'),('output_axis','method_adapters.py','{"input_dim": 1, "output_dim": 0}','{"input_dim": 1, "output_dim": 1}'),('loader_axis','linear.py','loaded_weight.narrow(output_dim, start_idx, shard_size)','loaded_weight.narrow(input_dim, start_idx, shard_size)')]:
        changed=dict(sources);changed[file]=changed[file].replace(old,new);assert changed[file]!=sources[file]
        try:inspect_qb_sources(changed)
        except ValueError:check('source_drift_'+name+'_rejects',True)
        else:raise AssertionError('source drift accepted '+name)
    for layer,rank in ((3,0),(43,15)):
        tag=str(layer);plan,*_=fixture(start=3,count=3,rank=rank);plan=replace(plan,layer_id=layer,contract=contract,revision=42);payload=load_rank_qb_cpu(reader,c,layer,plan.heads,contract);m=payload.manifest
        anchor=next(v for v in kvb_prior['cases'] if v['layer']==layer);output_anchor=next(v for v in oproj_prior['cases'] if v['layer']==layer)
        check(tag+'/rank_head_metadata_agrees_prior_KVB_Oproj',plan.heads.head_ids==tuple(anchor['global_heads'])==tuple(output_anchor['heads']) and (m.first_row,m.stop_row)==tuple(output_anchor['input_columns']))
        r['primary_weight_bytes']+=m.weight_bytes;r['primary_channel_bytes']+=16384*8
        raw=raw_part(m.prefix+'.weight',m.first_row,m.stop_row)
        check(tag+'/signed_INT8_rows_match_independent_raw_bytes',payload.weight_nk.dtype==torch.int8 and payload.weight_nk.shape==(1024,1536) and bool((payload.weight_nk<0).any()) and payload.weight_nk.view(torch.uint8).numpy().tobytes()==raw and payload.weight_sha256==sha(raw))
        channels={}
        for suffix in ('weight_scale','weight_offset'):
            name=m.prefix+'.'+suffix;raw_channel=raw_part(name);value=reader.channels[name];channels[suffix]=value
            check(tag+'/'+suffix+'_full_F32_raw_bytes',value.dtype==torch.float32 and value.shape==(16384,1) and value.view(torch.uint8).numpy().tobytes()==raw_channel)
        expected=channels['weight_scale'][m.first_row:m.stop_row].flatten().bfloat16()
        check(tag+'/only_local_output_scales_BF16_positive_symmetric',torch.equal(payload.weight_scale,expected) and payload.weight_scale.shape==(1024,) and bool((payload.weight_scale>0).all()) and not bool(channels['weight_offset'].any()))
        qr_raw=(torch.arange(3*1536).reshape(3,1536).float().remainder(97)*.01).bfloat16();qr=rms_norm(qr_raw,torch.ones(1536,dtype=torch.bfloat16),c.hf_config['text_config']['rms_norm_eps']);marker=torch.arange(3*1024).reshape(3,1024).float().remainder(37).bfloat16();events=[]
        def backend(x,w,s,*,spec):
            events.append('rank_backend');check(tag+'/full_normalized_qr_input_no_second_norm',x.shape==(3,1536) and torch.equal(x,qr) and spec.input_shape==(3,1536) and spec.input_policy=='rms_normalized_qr1536')
            check(tag+'/backend_payload_rows_scale_and_plan_bound',torch.equal(w,payload.weight_nk) and torch.equal(s,payload.weight_scale) and spec.identity==plan.identity and spec.output_rows==(m.first_row,m.stop_row) and spec.quantization_policy=='full_qr_dynamic_int8');return marker
        result=payload.project_cpu(qr,plan=plan,expected_identity=plan.identity,backend=backend,input_policy='rms_normalized_qr1536',quantization_policy='full_qr_dynamic_int8',bias_policy='absent')
        check(tag+'/owned_rank_query_shape_head_order',events==['rank_backend'] and result.identity==plan.identity and result.query.shape==(3,4,256) and torch.equal(result.query,marker.reshape(3,4,256)) and result.query.data_ptr()!=marker.data_ptr())
        # Existing KV-B actual bytes are NOT reread. Synthetic UK shape integration only.
        _,_,_,uk=components(rank=rank,layer=layer);uk=replace(uk,manifest=replace(uk.manifest,contract=contract))
        absorbed=uk.absorb_query_cpu(result.query,plan=plan)
        check(tag+'/local_query_fits_existing_rank_UK_contract',absorbed.shape==(3,4,512) and absorbed.dtype==torch.bfloat16 and bool(torch.isfinite(absorbed).all()))
        r['cases'].append({'layer':layer,'rank':rank,'heads':plan.heads.head_ids,'output_rows':[m.first_row,m.stop_row],'weight_shape':list(payload.weight_nk.shape),'weight_sha256':payload.weight_sha256,'runtime_scale_sha256':payload.runtime_scale_sha256,'descriptor_sha256':m.descriptor_sha256,'negative_codes':int((payload.weight_nk<0).sum()),'weight_bytes':m.weight_bytes,'full_F32_channel_bytes':131072,'logical_runtime_bytes':m.runtime_bytes,'backend_output':'synthetic markers;NO actual q_b compute','UK_integration':'synthetic FLOAT weights;prior actual KV-B bytes reused without reread'})
    check('existing_reader_exact_two_row_windows_four_full_channels',len(reader.reads)==6 and sum(v['bytes'] for v in reader.reads if v['kind']=='weight_rows')==3145728 and sum(v['bytes'] for v in reader.reads if v['kind']=='full_channel')==262144)
    check('primary_and_independent_logical_byte_totals',r['primary_weight_bytes']==3145728 and r['primary_channel_bytes']==262144 and r['independent_verification_bytes']==3407872)
    check('no_native_modules_imported',not any(n in sys.modules for n in ('torch_npu','fla_npu','_C_ascend')))
    r['reader_calls']=reader.reads;r['status']='passed'
except BaseException as exc:r.update(status='failed',failure=repr(exc));raise
finally:
    r['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'query-payload-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
