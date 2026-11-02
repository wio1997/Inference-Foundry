# SPDX-License-Identifier: Apache-2.0
"""Actual-config/descriptor and pinned-source contracts with SYNTHETIC stubs.

Reuses earlier actual signed-byte evidence; reads no weight/channel payloads.
This does not execute native W8A8, quantization, INT8 math or TP collectives.
"""
import argparse,ast,datetime,hashlib,json,os,platform,sys
from dataclasses import replace
from pathlib import Path
import torch
from test_glm5_next_sfa_inputs import fixture
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.sfa_inputs import NoPESFAContract,SFAHeadPartition
from deepinfer.model_executor.models.glm5_next.oproj_payload import o_proj_column_manifest,PreparedOProjColumns
from deepinfer.model_executor.models.glm5_next.oproj_execution import execute_rank_oproj_cpu

p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','prior-evidence','output-dir'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);root=Path(__file__).resolve().parents[2];prior=json.loads(a.prior_evidence.read_text());prior_kvb=a.prior_evidence.parent.parent/'round37/sfa-weight-result.json';kvb=json.loads(prior_kvb.read_text());assert prior['status']==kvb['status']=='passed'
sha=lambda b:hashlib.sha256(b).hexdigest()
config_sha=sha((a.checkpoint/'config.json').read_bytes());assert prior['config_sha256']==kvb['config_sha256']==config_sha
assert sha(prior_kvb.read_bytes())==prior['prior_evidence_sha256']
assert not prior['NPU_execution'] and not prior['INT8_GEMM_or_dequantization']
c=Glm5NextConfig.from_checkpoint(a.checkpoint);contract=NoPESFAContract.from_config(c,reference_source_sha256=kvb['reference_source_sha256'])
reader=ModelSlimCheckpoint(a.checkpoint,max_read_bytes=8*1024**2)
def forbidden(*args,**kwargs):raise AssertionError('weight/channel payload read forbidden')
for name in ('tensor','row_slice','matrix','linear_payload','load_linear','materialize'):setattr(reader,name,forbidden)
files=['deepinfer/model_executor/models/glm5_next/oproj_execution.py','tests/ascend/test_glm5_next_oproj_execution.py','tests/ascend/probe_glm5_next_oproj_execution_cpu.py']
r={'status':'in_progress','native_round':39,'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,'source_sha256':{n:sha((root/n).read_bytes()) for n in files},'prior_evidence_sha256':sha(a.prior_evidence.read_bytes()),'kvb_evidence_sha256':sha(prior_kvb.read_bytes()),'weight_or_channel_payload_bytes':0,'synthetic_callback_payloads':True,'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'INT8_GEMM_or_dequantization':False,'activation_quantization':False,'TP_collective':False,'model_generation':False,'checks':{},'cases':[],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def check(name,ok):
    assert name not in r['checks']
    if not ok:raise AssertionError(name)
    r['checks'][name]={'passed':True}
try:
    check('actual_GLM_hidden_heads_value_dimensions',(c.hidden_size,c.mla_dimensions['num_attention_heads'],c.mla_dimensions['v_head_dim'])==(4096,64,256))
    spans=[SFAHeadPartition(64,16,rank,tuple(range(rank*4,rank*4+4))) for rank in range(16)]
    check('TP16_complete_disjoint_input_column_cover',tuple(h for v in spans for h in v.head_ids)==tuple(range(64)) and [(v.head_ids[0]*256,(v.head_ids[-1]+1)*256) for v in spans]==[(i*1024,(i+1)*1024) for i in range(16)])
    native_path=root/'deepinfer/npu_ops/w8a8.py';native_source=native_path.read_text();r['native_interface_source_sha256']=sha(native_path.read_bytes())
    native_cls=next(n for n in ast.parse(native_source).body if isinstance(n,ast.ClassDef) and n.name=='AscendW8A8Linear')
    ctor=next(n for n in native_cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');forward=next(n for n in native_cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    check('native_constructor_positional_NK_scale_source_only',[v.arg for v in ctor.args.args]==['self','weight_nk','scale'])
    calls=[n for n in ast.walk(forward) if isinstance(n,ast.Call)]
    quant=next(n for n in calls if ast.unparse(n.func)=='torch_npu.npu_dynamic_quant');mm=next(n for n in calls if ast.unparse(n.func)=='torch_npu.npu_quant_matmul')
    check('native_dynamic_rank_input_before_quant_matmul_source_only',[ast.unparse(v) for v in quant.args]==['flat'] and quant.lineno<mm.lineno and [ast.unparse(v) for v in mm.args]==['quantized','self.weight','self.weight_scale'])
    check('native_per_token_scale_BF16_output_keywords_source_only',{v.arg:ast.unparse(v.value) for v in mm.keywords}=={'pertoken_scale':'token_scale','output_dtype':'x.dtype'})
    tree=ast.parse((root/files[0]).read_text());imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    check('execution_seam_has_no_native_import_or_factory',not any('npu_ops' in v or 'torch_npu' in v for v in imports))
    for old in prior['cases']:
        layer=old['layer'];rank=old['rank'];tag=str(layer);plan,*_=fixture(start=3,count=3,rank=rank);plan=replace(plan,layer_id=layer,contract=contract,revision=39)
        m=o_proj_column_manifest(reader,c,layer,plan.heads,contract)
        check(tag+'/actual_descriptor_identity_reuses_prior_payload_receipt',m.descriptor_sha256==old['descriptor_sha256'] and (m.first_column,m.stop_column)==tuple(old['input_columns']))
        anchor=next(v for v in kvb['cases'] if v['layer']==layer)
        check(tag+'/head_partition_agrees_KVB_Oproj',tuple(anchor['global_heads'])==tuple(old['heads'])==plan.heads.head_ids)
        # SYNTHETIC codes/scales only; actual weight payload evidence stays in round38.
        w=torch.full((4096,1024),-7,dtype=torch.int8);scale=torch.ones(4096,dtype=torch.bfloat16)
        digest=lambda t:sha(t.contiguous().view(torch.uint8).numpy().tobytes())
        payload=PreparedOProjColumns(m,w,scale,digest(w),'a'*64,'b'*64);values=torch.arange(3*4*256).reshape(3,4,256).bfloat16();events=[]
        def apply(x,weight,channel,*,spec):
            events.append('rank_apply')
            check(tag+'/supplied_backend_receives_full_contract',spec.identity==plan.identity and spec.weight_sha256==digest(w) and spec.runtime_scale_sha256==digest(scale) and spec.descriptor_sha256==old['descriptor_sha256'] and spec.input_columns==tuple(old['input_columns']) and weight.dtype==torch.int8 and channel.shape==(4096,) and torch.equal(x,values.reshape(3,1024)))
            return torch.full((3,4096),rank+1,dtype=torch.bfloat16)
        def reduce(y,*,spec):
            events.append('sum_stub');check(tag+'/reducer_receives_full_output_after_rank_apply',events==['rank_apply','sum_stub'] and y.shape==(3,4096) and bool((y==rank+1).all()) and spec.output_shape==(3,4096));return torch.full_like(y,17)
        out=execute_rank_oproj_cpu(payload,values,plan=plan,expected_identity=plan.identity,expected_runtime_scale_sha256=digest(scale),quantization_policy='rank_local_dynamic_int8',bias_policy='absent',rank_apply=apply,reduce_sum=reduce)
        check(tag+'/owned_BF16_stub_output',out.dtype==torch.bfloat16 and out.shape==(3,4096) and bool((out==17).all()))
        r['cases'].append({'layer':layer,'rank':rank,'heads':plan.heads.head_ids,'input_columns':old['input_columns'],'actual_descriptor_sha256':m.descriptor_sha256,'reused_actual_weight_sha256':old['weight_sha256'],'executed_weight_data':'synthetic signedINT8 -7;NO GEMM','executed_scales':'synthetic BF16 ones','callbacks':'shape/copy/order stubs only;no quantization or TP collective','bias_policy':'caller-declared absent;not runtime proof'})
    check('no_native_modules_imported',not any(n in sys.modules for n in ('torch_npu','fla_npu','_C_ascend')))
    r['status']='passed'
except BaseException as exc:r.update(status='failed',failure=repr(exc));raise
finally:
    r['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'oproj-execution-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
