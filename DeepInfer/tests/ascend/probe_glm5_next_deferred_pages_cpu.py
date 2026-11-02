# SPDX-License-Identifier: Apache-2.0
"""Actual eleven-DSA CPU cache transitions; no backbone/model/native execution."""
import argparse,datetime,hashlib,json,os,platform,sys
from pathlib import Path
from unittest.mock import patch
import torch
from test_glm5_next_deferred_pages import expected_bytes,buffers
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout,DSAState,state_from_aligned_prefix
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry
from deepinfer.model_executor.models.glm5_next.deferred_pages import CPUPageBinding,capture_page_bindings
import deepinfer.model_executor.models.glm5_next.paged_commit as commit

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--prior-cache',type=Path,required=True)
p.add_argument('--prior-request',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);src=Path(__file__).resolve().parents[2];config_sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
rollback={'deepinfer/model_executor/models/glm5_next/paged_commit.py':a.output_dir/'paged_commit.py.before','deepinfer/model_executor/models/glm5_next/text_request.py':a.output_dir/'text_request.py.before'}
for f in (a.prior_cache,a.prior_request):
    old=json.loads(f.read_bytes());assert old['status']=='passed' and not old['NPU_execution'] and old['config_sha256']==config_sha
    for n,sha in old['source_sha256'].items():assert hashlib.sha256((rollback.get(n,src/n)).read_bytes()).hexdigest()==sha
files=['deepinfer/model_executor/models/glm5_next/deferred_pages.py','deepinfer/model_executor/models/glm5_next/paged_commit.py','deepinfer/model_executor/models/glm5_next/text_request.py','tests/ascend/test_glm5_next_deferred_pages.py','tests/ascend/probe_glm5_next_deferred_pages_cpu.py']
report={'status':'in_progress','native_round':28,'NPU_execution':False,'model_generation':False,'real_backbone_executed':False,'weights_read_bytes':0,
        'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reused_evidence_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.prior_cache,a.prior_request)},
        'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'checks':{}}


def check(name,value):
    assert value,name
    report['checks'][name]={'passed':True}


try:
    c=Glm5NextConfig.from_checkpoint(a.checkpoint);ids=[s.layer_id for s in c.layers if s.attention=='deepseek_sparse_attention']
    layout=DSACacheLayout.from_config(c,logical_block_size=128);g=CANNCacheGeometry(layout,6*2*layout.index_dim*4)
    check('actual_DSA_topology_and_dimensions',ids==list(range(3,44,4)) and (layout.latent_dim,layout.index_dim,layout.pool_size)==(512,128,4))
    bindings={};old_states={};new_states={}
    for layer in ids:
        bindings[layer]=CPUPageBinding(torch.full((2*g.main_page_bytes,),0xa5,dtype=torch.uint8),torch.full((2*g.index_page_bytes,),0xa5,dtype=torch.uint8),torch.full((2*g.tail_page_bytes,),0xa5,dtype=torch.uint8),g,2,2,torch.tensor([[1,0]]),1)
        old_states[layer]=state_from_aligned_prefix(torch.empty(0,512,dtype=torch.bfloat16),torch.empty(0,128,dtype=torch.bfloat16),layout)
        new_states[layer]=DSAState((torch.arange(7*512).reshape(7,512).float()/512+layer).bfloat16(),(torch.arange(128).reshape(1,128).float()/128+layer).bfloat16(),torch.arange(2*4*128).reshape(2,4,128).float()/128+layer,torch.tensor([4,5,6,3]),7)
    owned=capture_page_bindings(c,layout,bindings);check('all11_binding_identity',len(owned)==11 and all(owned[i] is bindings[i] for i in ids))
    before=buffers(bindings);proposals=tuple(bindings[i].stage(old_states[i],new_states[i]) for i in ids)
    check('all11_staging_does_not_write',buffers(bindings)==before)
    calls=[]
    def fail_last(target,rows,values):
        target.index_copy_(0,rows,values);calls.append(len(rows))
        if len(calls)==33:raise RuntimeError('last of11 layer tails failed')
    with patch.object(commit,'_write_rows',side_effect=fail_last):
        try:commit.commit_dsa_batches(proposals)
        except RuntimeError:pass
        else:raise AssertionError('last-layer write failure did not surface')
    check('all33_writes_attempted',len(calls)==33)
    check('all11_layers_exact_rollback',buffers(bindings)==before and all(p.summary()['state']=='aborted' for p in proposals))
    check('old_state_tokens_unpublished',all(s.tokens_seen==0 for s in old_states.values()))
    proposals=tuple(bindings[i].stage(old_states[i],new_states[i]) for i in ids);summaries=commit.commit_dsa_batches(proposals)
    check('all11_commit_terminal',len(summaries)==11 and all(s['state']=='committed' for s in summaries))
    check('all11_raw_scalar_bytes',all(buffers(bindings)[i]==expected_bytes(bindings[i],old_states[i],new_states[i],before[i]) for i in ids))
    for i in ids:bindings[i].require_snapshot(new_states[i])
    check('all11_prefixes_match_at7',True)
    following={}
    for i in ids:
        s=new_states[i];tail=s.tail.clone();tail[:,3]=i+7
        following[i]=DSAState(torch.cat((s.latent,torch.full((1,512),i+.5,dtype=torch.bfloat16))),torch.cat((s.compressed,torch.full((1,128),i+1.,dtype=torch.bfloat16))),tail,torch.tensor([4,5,6,7]),8)
    before=buffers(bindings);summaries2=commit.commit_dsa_batches(tuple(bindings[i].stage(new_states[i],following[i]) for i in ids))
    check('all11_continuation_bytes_at8',all(buffers(bindings)[i]==expected_bytes(bindings[i],new_states[i],following[i],before[i]) for i in ids))
    for i in ids:bindings[i].require_snapshot(following[i])
    check('all11_prefixes_match_at8',True)
    before=buffers(bindings);empty=commit.commit_dsa_batches(tuple(bindings[i].stage(following[i],following[i]) for i in ids))
    check('all11_empty_transition_noop',buffers(bindings)==before and all(s['payload_bytes']==0 for s in empty))
    b=bindings[ids[-1]];b.main_raw[g.main_page_bytes]^=1
    try:b.require_snapshot(following[ids[-1]])
    except ValueError:pass
    else:raise AssertionError('corrupt restored history accepted')
    b.main_raw[g.main_page_bytes]^=1;check('corrupt_restore_prefix_rejects',buffers(bindings)==before)
    try:capture_page_bindings(c,layout,bindings,(owned,))
    except ValueError:pass
    else:raise AssertionError('simultaneous owner accepted')
    check('cross_request_backing_ownership_rejects',True)
    check('empty_group_noop',commit.commit_dsa_batches(())==())
    check('native_imports_absent','torch_npu' not in sys.modules and 'fla_npu' not in sys.modules)
    report.update(layer_ids=ids,cache_source='synthetic DSA snapshots; compression/attention not computed',total_backing_bytes=sum(t.numel() for b in bindings.values() for t in (b.main_raw,b.index_raw,b.tail_raw)),initial_payload_bytes=sum(s['payload_bytes'] for s in summaries),continuation_payload_bytes=sum(s['payload_bytes'] for s in summaries2))
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'deferred-pages-result.json').write_text(json.dumps(report,indent=2)+'\n')
