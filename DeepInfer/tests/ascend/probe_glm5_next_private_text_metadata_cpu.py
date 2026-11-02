# SPDX-License-Identifier: Apache-2.0
"""Actual45-layer METADATA preflight only; no model math/payload/native run."""
import argparse,datetime,hashlib,json,os,platform
from pathlib import Path
import torch
from torch import nn
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.glm5_next_plan import GlmBackbonePlan
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry
from deepinfer.model_executor.models.glm5_next.hybrid_decoder import PreparedDecoderStack

p=argparse.ArgumentParser(description=__doc__)
for n in ('checkpoint','prior-evidence','output-dir'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);config=Glm5NextConfig.from_checkpoint(a.checkpoint);prior=json.loads(a.prior_evidence.read_text());config_sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
src=Path(__file__).resolve().parents[2];files=['deepinfer/model_executor/models/glm5_next/'+n+'.py' for n in ('decoder','hybrid_decoder','private_request','text_request','private_text_request')]+['tests/ascend/test_glm5_next_private_text_request.py','tests/ascend/probe_glm5_next_private_text_metadata_cpu.py']
r={'status':'in_progress','native_round':34,'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,
   'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'prior_evidence_sha256':hashlib.sha256(a.prior_evidence.read_bytes()).hexdigest(),
   'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'actual_model_forward':False,'weights_read_bytes':0,'full_model_states_allocated':False,
   'scope':'actual45-layer metadata dispatch preflight;runtime integration tested separately with tiny injected KDA/DSA/DSA stack','checks':{},'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def check(n,v):
    if not v:raise AssertionError(n)
    r['checks'][n]={'passed':True}
class MetadataLayer(nn.Module):
    def __init__(self,s):super().__init__();self.layer_id,self.attention_kind,self.mlp_kind=s.layer_id,s.attention,s.mlp
    def forward(self,*a,**kw):raise AssertionError('actual metadata layer must never execute')
try:
    check('reused_round33_evidence_pins_config',prior['status']=='passed' and prior['config_sha256']==config_sha and not prior['NPU_execution'])
    plan=GlmBackbonePlan.from_config(config);layout=DSACacheLayout.from_config(config,logical_block_size=128)
    dsa=tuple(s.layer_id for s in plan.layers if s.attention=='deepseek_sparse_attention');kda=tuple(s.layer_id for s in plan.layers if s.attention=='linear_attention')
    check('actual45_contiguous_layers',tuple(s.layer_id for s in plan.layers)==tuple(range(45)))
    check('actual34_KDA_and11_DSA',len(kda)==34 and dsa==tuple(range(3,44,4)))
    check('actual_mhc_hidden_geometry',(plan.hidden_size,plan.streams)==(4096,4))
    check('actual_KDA_geometry',(plan.kda_heads,plan.kda_dim,plan.conv_width)==(64,128,4))
    check('actual_DSA_geometry',(layout.latent_dim,layout.index_dim,layout.pool_size)==(512,128,4))
    check('actual_nope_attention_geometry',tuple(config.mla_dimensions[k] for k in ('num_attention_heads','qk_nope_head_dim','v_head_dim','qk_rope_head_dim'))==(64,256,256,0))
    stack=PreparedDecoderStack(config,layout,{s.layer_id:MetadataLayer(s) for s in plan.layers});operations={i:lambda *x:None for i in dsa}
    # Valid full coverage reaches state preflight; never allocate actual layer states.
    try:stack(None,None,dsa_operations=operations)
    except ValueError as e:check('all11_operations_pass_map_preflight_then_reject_missing_state','complete request state' in str(e))
    else:raise AssertionError('missing state accepted')
    for layer in dsa:
        wrong=dict(operations);del wrong[layer]
        try:stack(None,None,dsa_operations=wrong)
        except ValueError as e:check(f'missing_layer{layer}_rejects_before_execution','cover every DSA layer' in str(e))
        else:raise AssertionError('incomplete operation map accepted')
    for name,wrong in [('extra_KDA',{**operations,0:lambda *a:None}),('bad_callable',{**operations,dsa[-1]:None})]:
        try:stack(None,None,dsa_operations=wrong)
        except ValueError as e:check(name+'_rejects_before_execution','cover every DSA layer' in str(e))
        else:raise AssertionError('invalid map accepted')
    geometries={}
    for B in (128,256):
        g=CANNCacheGeometry(DSACacheLayout.from_config(config,logical_block_size=B),9*256*4)
        raw_one=10*(g.main_page_bytes+g.index_page_bytes)+5*g.tail_page_bytes
        geometries[str(B)]={'fixture_raw_backing_bytes_per_DSA_layer':raw_one,'all11_fixture_raw_backing_bytes':11*raw_one}
        check(f'B{B}_11layer_raw_fixture_backing_within_default64MiB',11*raw_one<=64*1024**2)
    r['actual_layer_ids']={'KDA':kda,'DSA':dsa};r['arithmetic_only']=geometries
    r['budget_scope']='raw CPU main/index/tail backings per composite frame only;excludes KDA/state/stage/retained frames/callbacks/copy peak;no process/HBM claim'
    r['logical_state_bytes_arithmetic_only']=plan.logical_state_bytes(128)
    r['status']='passed'
except BaseException as e:r.update(status='failed',failure=repr(e));raise
finally:
    r['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'private-text-metadata-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
