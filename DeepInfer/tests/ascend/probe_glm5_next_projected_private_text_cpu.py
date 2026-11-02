# SPDX-License-Identifier: Apache-2.0
"""Actual config and prior receipts with synthetic CPU projected transactions.

No actual checkpoint weight/channel/shard reads, native calls or TP collective.
The8-layer control-flow fixture uses synthetic KDA/mHC/FFN/projection/head data.
"""
import argparse,datetime,hashlib,json,os,platform,sys
from pathlib import Path
from unittest.mock import patch
import torch
from test_glm5_next_projected_private_text import fixture,plans
from test_glm5_next_private_text_request import same_pages
from test_glm5_next_hybrid_decoder import equal_state
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.models.glm5_next.sfa_inputs import NoPESFAContract
from deepinfer.model_executor.models.glm5_next.private_request import PrivateDSARequest

p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','evidence-root','output-dir'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'));torch.set_num_threads(2)
sha=lambda b:hashlib.sha256(b).hexdigest();config_sha=sha((a.checkpoint/'config.json').read_bytes());c=Glm5NextConfig.from_checkpoint(a.checkpoint)
names={37:'sfa-weight-result.json',38:'oproj-payload-result.json',39:'oproj-execution-result.json',40:'rank-dsa-result.json'};receipts={i:json.loads((a.evidence_root/f'round{i}'/n).read_text()) for i,n in names.items()}
assert all(v['status']=='passed' and v['config_sha256']==config_sha and not v['NPU_execution'] for v in receipts.values());contract=NoPESFAContract.from_config(c,reference_source_sha256=receipts[37]['reference_source_sha256'])
root=Path(__file__).resolve().parents[2];files=['deepinfer/model_executor/models/glm5_next/projected_reader.py','deepinfer/model_executor/models/glm5_next/private_request.py','deepinfer/model_executor/models/glm5_next/private_text_request.py','tests/ascend/test_glm5_next_projected_private_text.py','tests/ascend/probe_glm5_next_projected_private_text_cpu.py']
r={'status':'in_progress','native_round':41,'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,'reference_source_sha256':contract.reference_source_sha256,'prior_receipts_sha256':{str(i):sha((a.evidence_root/f'round{i}'/n).read_bytes()) for i,n in names.items()},'source_sha256':{n:sha((root/n).read_bytes()) for n in files},'weight_channel_or_shard_payload_bytes':0,'synthetic_8_layer_control_flow':True,'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'INT8_GEMM_or_dequantization':False,'activation_quantization':False,'TP_collective':False,'model_generation':False,'checks':{},'plans':[],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def check(name,ok):
    assert name not in r['checks']
    if not ok:raise AssertionError(name)
    r['checks'][name]={'passed':True}
try:
    check('actual45layers_34KDA_11DSA',len(c.layers)==45 and len([v for v in c.layers if v.attention=='linear_attention'])==34 and tuple(v.layer_id for v in c.layers if v.attention=='deepseek_sparse_attention')==tuple(range(3,44,4)))
    check('actual_reader_geometry_NoPE4096_qr1536_64x256_latent512',(c.hidden_size,c.mla_dimensions['q_lora_rank'],c.mla_dimensions['num_attention_heads'],c.mla_dimensions['qk_nope_head_dim'],c.mla_dimensions['v_head_dim'],c.mla_dimensions['kv_lora_rank'],contract.scale)==(4096,1536,64,256,256,512,1/16))
    runner,initial,q,bindings,calls,kept,embedding=fixture(source_contract=contract)
    for layer in (3,7):
        reader=q._attentions[layer];reader.validate_config(c,layer)
        check(str(layer)+'/reader_matches_actual_layer_contract',reader.contract==contract and reader.heads.local_heads==4 and c.layers[layer].attention=='deepseek_sparse_attention')
    stamp=q.stamp;before=q.page_snapshot()
    with patch.object(PrivateDSARequest,'publish',side_effect=AssertionError('layer publication forbidden')):
        out=q.step(torch.tensor([1,2,3]),expected_stamp=stamp,page_plans=plans(q,3))
    check('first_step_one_composite_revision',q.stamp.revision==1 and q.stamp.tokens_seen==3 and q.stamp!=stamp)
    check('first_step_all8states_advance_together',{v.tokens_seen for v in q.snapshot().layers.values()}=={3})
    check('first_step_projected_output_and_selected_logits',out.hidden.shape==(3,4096) and out.logits.shape==(3,4) and out.hidden.dtype==out.logits.dtype==torch.bfloat16 and (out.vocab_start,out.vocab_end)==(1,5))
    check('no_finish_on_projected_layers',all(runner.stack.layers[i].attention.mla.finishes==0 for i in (3,7)))
    check('first_step_callback_order',[v[:2] for v in calls]==[('producer',3),('reader',3),('attention',3),('rank_apply',3),('sum_stub',3),('producer',7),('reader',7),('attention',7),('rank_apply',7),('sum_stub',7)])
    check('plan_request_revision_and_source_bound',len(kept)==2 and all(v.request_id==q.request_id and v.revision==0 and v.contract==contract and v.candidate.start==0 and v.candidate.tokens==3 for v in kept))
    check('one_attempt_nonce_distinct_layer_metadata',kept[0].identity._owner is kept[1].identity._owner and kept[0].identity!=kept[1].identity)
    check('caller_backings_and_old_snapshots_unchanged',all(not bool(t.any()) for b in bindings.values() for t in (b.main_raw,b.index_raw,b.tail_raw)) and all(v.state.tokens_seen==0 for v in before.values()))
    next_out=q.step(torch.tensor([4,5]),expected_stamp=q.stamp,page_plans=plans(q,2))
    check('continuation_uses_committed_prefix_revision_and_new_attempt',q.stamp.revision==2 and q.stamp.tokens_seen==5 and all(v.revision==1 and v.candidate.start==3 for v in kept[-2:]) and kept[-2].identity._owner is not kept[0].identity._owner)
    check('continuation_all_states_pages_match_position',{v.tokens_seen for v in q.snapshot().layers.values()}=={5} and {v.state.tokens_seen for v in q.page_snapshot().values()}=={5})
    saved=q.snapshot();pages=q.page_snapshot();stamp=q.stamp;runner.head.fail=True
    try:q.step(torch.tensor([6]),expected_stamp=stamp,page_plans=plans(q,1))
    except RuntimeError as exc:check('late_head_failure_reached','head failure' in str(exc))
    else:raise AssertionError('head failure not raised')
    failed=kept[-2:];equal_state(saved,q.snapshot());same_pages(pages,q.page_snapshot())
    check('head_failure_all_layers_pages_and_revision_unpublished',q.stamp==stamp and {v.tokens_seen for v in q.snapshot().layers.values()}=={5})
    runner.head.fail=False;q.step(torch.tensor([6]),expected_stamp=stamp,page_plans=plans(q,1));retry=kept[-2:]
    check('retry_same_metadata_fresh_attempt_identity',all(x.identity.metadata_sha256==y.identity.metadata_sha256 and x.identity!=y.identity and x.identity._owner is not y.identity._owner for x,y in zip(failed,retry)))
    check('retry_advances_exactly_once',q.stamp.revision==3 and q.stamp.tokens_seen==6 and {v.tokens_seen for v in q.snapshot().layers.values()}=={6})
    r['plans']=[{'layer':v.layer_id,'request':v.request_id,'revision':v.revision,'start':v.candidate.start,'tokens':v.candidate.tokens,'heads':v.heads.head_ids,'read_table':v.candidate.read_table,'identity_metadata_sha256':v.identity.metadata_sha256,'scope':'synthetic candidate metadata;not physical/native leases'} for v in kept]
    check('no_native_modules_imported',not any(n in sys.modules for n in ('torch_npu','fla_npu','_C_ascend')))
    q.close();r['status']='passed'
except BaseException as exc:r.update(status='failed',failure=repr(exc));raise
finally:
    r['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'projected-private-text-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
