# SPDX-License-Identifier: Apache-2.0
"""Actual config/metadata text binding only; no real backbone or weight loads."""
import argparse,datetime,hashlib,json,os,platform,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout
from deepinfer.model_executor.models.glm5_next.hybrid_decoder import PreparedDecoderStack
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState
from deepinfer.model_executor.models.glm5_next.text_boundary import GlmTextBoundary,GlmLogitsShard,text_boundary_manifest
from deepinfer.model_executor.models.glm5_next.text_request import PreparedTextRunner

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--prior-plan',type=Path,required=True)
p.add_argument('--prior-boundary',type=Path,required=True);p.add_argument('--prior-native',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2)
old=[json.loads(f.read_bytes()) for f in (a.prior_plan,a.prior_boundary,a.prior_native)]
config_sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
assert all(x['status']=='passed' and not x['NPU_execution'] and x['config_sha256']==config_sha for x in old)
src=Path(__file__).resolve().parents[2]
for previous in old:
 for name,sha in previous['source_sha256'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==sha
files=['deepinfer/model_executor/models/glm5_next/text_request.py','tests/ascend/test_glm5_next_text_request.py','tests/ascend/probe_glm5_next_text_request_cpu.py']
report={'status':'in_progress','native_round':26,'NPU_execution':False,'real_backbone_executed':False,'model_generation':False,
        'synthetic_component_binding':True,'weights_read_bytes':0,'architecture':platform.machine(),'torch':torch.__version__,
        'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config_sha256':config_sha,
        'reused_evidence_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.prior_plan,a.prior_boundary,a.prior_native)},
        'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'checks':{}}


def check(name,value):
    assert value,name
    report['checks'][name]={'passed':True}


def rejects(name,action):
    try:action()
    except ValueError:check(name,True)
    else:raise AssertionError(name)


class Reader(ModelSlimCheckpoint):
    def denied(self,*args,**kwargs):raise AssertionError('weight payload must not be read')
    tensor=matrix=linear_payload=raw_rows=row_slice=denied


class MetadataLayer(nn.Module):
    def __init__(self,spec,hidden,streams):
        super().__init__();self.layer_id=spec.layer_id;self.attention_kind=spec.attention;self.mlp_kind=spec.mlp
        self.attn_hc=self.ffn_hc=SimpleNamespace(hidden_size=hidden,streams=streams)
    def forward(self,*args):raise AssertionError('metadata layer is not executable')


try:
    c=Glm5NextConfig.from_checkpoint(a.checkpoint);reader=Reader(a.checkpoint,max_read_bytes=256*1024);m=text_boundary_manifest(reader,c)
    check('actual_text_geometry',(m.hidden_size,m.vocab_size,m.streams)==(4096,154880,4))
    check('actual_topology',len(c.layers)==45 and sum(s.attention=='linear_attention' for s in c.layers)==34 and sum(s.mlp=='sparse' for s in c.layers)==42)
    boundary=GlmTextBoundary(reader,m,torch.ones(4096,dtype=torch.bfloat16))
    head=GlmLogitsShard(torch.zeros(8,4096,dtype=torch.bfloat16),vocab_start=m.vocab_size-8,scale=m.logit_scale)
    stack=PreparedDecoderStack(c,DSACacheLayout.from_config(c,logical_block_size=128),{s.layer_id:MetadataLayer(s,m.hidden_size,m.streams) for s in c.layers})
    runner=PreparedTextRunner(stack,boundary,head)
    check('explicit_component_identity',runner.stack is stack and runner.boundary is boundary and runner.head is head)
    check('bounded_ownership_and_history',(runner.max_requests,runner.max_context_tokens)==(1,128))
    check('global_last_eight_head_range',(head.vocab_start,head.vocab_end)==(154872,154880))
    check('all_layer_stream_geometries',all(l.attn_hc.hidden_size==4096 and l.attn_hc.streams==4 for l in stack.layers))
    rejects('missing_stack_rejected',lambda:PreparedTextRunner(None,boundary,head))
    rejects('nonpositive_request_bound_rejected',lambda:PreparedTextRunner(stack,boundary,head,max_requests=0))
    rejects('boolean_context_bound_rejected',lambda:PreparedTextRunner(stack,boundary,head,max_context_tokens=True))
    rejects('incomplete_state_rejected_before_payload',lambda:runner.open_request(MixedRequestState('metadata-only',0,{})))
    head.scale=m.logit_scale+1
    rejects('wrong_head_scale_rejected',lambda:PreparedTextRunner(stack,boundary,head));head.scale=m.logit_scale
    head.vocab_end=m.vocab_size+1
    rejects('out_of_vocabulary_head_rejected',lambda:PreparedTextRunner(stack,boundary,head));head.vocab_end=m.vocab_size
    stack.layers[-1].ffn_hc=SimpleNamespace(hidden_size=4096,streams=3)
    rejects('last_layer_wrong_streams_rejected',lambda:PreparedTextRunner(stack,boundary,head))
    check('no_native_imports','torch_npu' not in sys.modules and 'fla_npu' not in sys.modules)
    check('config_file_unchanged',config_sha==hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest())
    report['synthetic_boundary_elements']=4096+8*4096
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'text-request-result.json').write_text(json.dumps(report,indent=2)+'\n')
