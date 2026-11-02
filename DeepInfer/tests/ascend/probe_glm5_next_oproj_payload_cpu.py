# SPDX-License-Identifier: Apache-2.0
"""Actual signedINT8 O-proj column bytes/channels;no matrix/quantization math."""
import argparse,datetime,hashlib,json,mmap,os,platform,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from test_glm5_next_sfa_inputs import fixture
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.sfa_inputs import NoPESFAContract
from deepinfer.model_executor.models.glm5_next.oproj_payload import load_o_proj_columns_cpu,inspect_row_parallel_sources

p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','prior-evidence','reference-dir','output-dir'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);c=Glm5NextConfig.from_checkpoint(a.checkpoint);reader=ModelSlimCheckpoint(a.checkpoint,max_read_bytes=8*1024**2);sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest();prior=json.loads(a.prior_evidence.read_text());assert prior['status']=='passed' and prior['config_sha256']==sha and not prior['NPU_execution']
linear=(a.reference_dir/'linear.py').read_text();quant=(a.reference_dir/'w8a8_dynamic.py').read_text();refs={n:hashlib.sha256((a.reference_dir/n).read_bytes()).hexdigest() for n in ('linear.py','w8a8_dynamic.py')};assert refs==json.loads((a.reference_dir/'sha256.json').read_text())
contract=NoPESFAContract.from_config(c,reference_source_sha256=prior['reference_source_sha256']);src=Path(__file__).resolve().parents[2];files=['deepinfer/model_executor/models/glm5_next/oproj_payload.py','tests/ascend/test_glm5_next_oproj_payload.py','tests/ascend/probe_glm5_next_oproj_payload_cpu.py']
r={'status':'in_progress','native_round':38,'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':sha,'prior_evidence_sha256':hashlib.sha256(a.prior_evidence.read_bytes()).hexdigest(),'reference_sha256':refs,
 'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'INT8_GEMM_or_dequantization':False,'activation_quantization':False,'TP_collective':False,'model_generation':False,
 'full_weight_matrix_read_or_allocated':False,'primary_weight_bytes':0,'primary_channel_bytes':0,'independent_verification_bytes':0,'scope':'actual2 signedINT8 input-column windows andfulloutputchannel scales;no compute/runtime readiness','checks':{},'cases':[],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def check(name,value):
    if name in r['checks']:raise AssertionError('duplicate '+name)
    if not value:raise AssertionError(name)
    r['checks'][name]={'passed':True}
try:
    ledger=inspect_row_parallel_sources(linear,quant);r['source_contract']=ledger
    check('source_roles_ast_only',ledger['source_only'] and not ledger['native_ABI_validated'])
    for name,old,new in (('axis','loaded_weight.narrow(input_dim, start_idx, shard_size)','loaded_weight.narrow(0, start_idx, shard_size)'),('apply','self.quant_method.apply(self, input_parallel, bias_)','self.quant_method.apply(self, input_, bias_)')):
        changed=linear.replace(old,new);assert changed!=linear
        try:inspect_row_parallel_sources(changed,quant)
        except ValueError:check('source_drift_'+name+'_rejected',True)
        else:raise AssertionError('source drift accepted')
    changed=quant.replace('torch.empty(output_size, input_size, dtype=torch.int8)','torch.empty(input_size, output_size, dtype=torch.int8)');assert changed!=quant
    try:inspect_row_parallel_sources(linear,changed)
    except ValueError:check('source_drift_NK_rejected',True)
    else:raise AssertionError('NK drift accepted')
    for layer,rank in ((3,0),(43,15)):
        tag=str(layer);plan,*_=fixture(start=3,count=3,rank=rank);plan=replace(plan,layer_id=layer,contract=contract,revision=12)
        anchor=next(v for v in prior['cases'] if v['layer']==layer);check(tag+'/head_partition_matches_prior_actual_KVB',tuple(anchor['global_heads'])==plan.heads.head_ids)
        payload=load_o_proj_columns_cpu(reader,c,layer,plan.heads,contract);m=payload.manifest;r['primary_weight_bytes']+=m.weight_bytes;r['primary_channel_bytes']+=4096*8
        check(tag+'/correct_input_columns_all_output_rows',(m.first_column,m.stop_column)==((0,1024) if rank==0 else (15360,16384)) and payload.weight_nk.shape==(4096,1024))
        path,header,d=reader.descriptor(m.prefix+'.weight')
        # Read-only virtual mapping;copy ONLY selected columns,never whole matrix.
        with path.open('rb') as f:
            with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mapped:
                full_view=np.ndarray((4096,16384),dtype=np.int8,buffer=mapped,offset=header+d['data_offsets'][0]);selected=full_view[:,m.first_column:m.stop_column].copy();del full_view
        raw=selected.tobytes();r['independent_verification_bytes']+=len(raw)
        check(tag+'/independent_strided_raw_bytes',raw==payload.weight_nk.view(torch.uint8).numpy().tobytes())
        check(tag+'/signed_INT8_and_hash',payload.weight_nk.dtype==torch.int8 and bool((payload.weight_nk<0).any()) and hashlib.sha256(raw).hexdigest()==payload.weight_sha256)
        sidecar={}
        for suffix in ('weight_scale','weight_offset'):
            path,header,d=reader.descriptor(m.prefix+'.'+suffix)
            with path.open('rb') as f:f.seek(header+d['data_offsets'][0]);data=f.read(4096*4)
            r['independent_verification_bytes']+=len(data);sidecar[suffix]=torch.frombuffer(bytearray(data),dtype=torch.float32).reshape(4096,1)
            expected=payload.source_scale_sha256 if suffix=='weight_scale' else payload.source_offset_sha256
            check(tag+'/'+suffix+'_raw_digest',len(data)==4096*4 and hashlib.sha256(data).hexdigest()==expected)
        check(tag+'/full_output_channel_scale_BF16',payload.weight_scale.shape==(4096,) and payload.weight_scale.dtype==torch.bfloat16 and torch.equal(payload.weight_scale,sidecar['weight_scale'].flatten().bfloat16()))
        check(tag+'/symmetric_offset_and_positive_scale',not bool(sidecar['weight_offset'].any()) and bool(torch.isfinite(payload.weight_scale).all()) and bool((payload.weight_scale>0).all()))
        values=torch.arange(3*4*256).reshape(3,4,256).bfloat16();local=payload.prepare_input_cpu(values,plan=plan)
        check(tag+'/local_input_order_shape',local.shape==(3,1024) and torch.equal(local,values.reshape(3,1024)))
        check(tag+'/local_input_matches_weight_K',local.shape[1]==payload.weight_nk.shape[1] and payload.weight_nk.shape[0]==c.hidden_size)
        check(tag+'/payload_budgets_logical',m.checkpoint_bytes==4194304+32768 and m.runtime_bytes==4194304+8192)
        r['cases'].append({'layer':layer,'declared_TP':16,'rank':rank,'heads':plan.heads.head_ids,'input_columns':[m.first_column,m.stop_column],'weight_shape':list(payload.weight_nk.shape),'weight_dtype':str(payload.weight_nk.dtype),'weight_bytes':m.weight_bytes,'channel_F32_bytes':32768,'runtime_scale_BF16_bytes':8192,'weight_sha256':payload.weight_sha256,'source_scale_sha256':payload.source_scale_sha256,'source_offset_sha256':payload.source_offset_sha256,'descriptor_sha256':m.descriptor_sha256,'negative_codes':int((payload.weight_nk<0).sum()),'runtime_scale_sha256':hashlib.sha256(payload.weight_scale.view(torch.uint8).numpy().tobytes()).hexdigest()})
    check('two_bounded_input_column_payloads',len(r['cases'])==2 and r['primary_weight_bytes']==8388608 and r['primary_channel_bytes']==65536 and r['independent_verification_bytes']==8454144)
    check('no_native_backend_modules_imported',not any(n in sys.modules for n in ('torch_npu','fla_npu','_C_ascend')))
    r['status']='passed'
except BaseException as e:r.update(status='failed',failure=repr(e));raise
finally:
    r['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'oproj-payload-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
