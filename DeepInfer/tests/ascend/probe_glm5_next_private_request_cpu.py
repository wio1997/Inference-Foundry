# SPDX-License-Identifier: Apache-2.0
"""Actual GLM dimensions with synthetic producers/two-head CPU attention."""
import argparse,datetime,hashlib,json,os,platform
from dataclasses import replace
from pathlib import Path
import torch
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout,state_from_aligned_prefix,append_dsa
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry,view_tail,pack_tail_planes
from deepinfer.model_executor.models.glm5_next.private_request import PrivateDSARequest
from deepinfer.model_executor.models.glm5_next.absorbed_mla import AbsorbedMLAContract,split_kv_b_cpu,absorbed_sparse_mla_cpu

p=argparse.ArgumentParser(description=__doc__)
for n in ('checkpoint','prior-evidence','output-dir'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);torch.manual_seed(5333);config=Glm5NextConfig.from_checkpoint(a.checkpoint);prior=json.loads(a.prior_evidence.read_text());config_sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
assert prior['status']=='passed' and not prior['NPU_execution'] and prior['config_sha256']==config_sha
src=Path(__file__).resolve().parents[2];files=['deepinfer/model_executor/models/glm5_next/private_request.py','tests/ascend/test_glm5_next_private_request.py','tests/ascend/probe_glm5_next_private_request_cpu.py']
report={'status':'in_progress','native_round':33,'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'model_generation':False,'weights_read_bytes':0,
 'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':config_sha,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'prior_evidence_sha256':hashlib.sha256(a.prior_evidence.read_bytes()).hexdigest(),'checks':{},'cases':[]}
def check(name,value):
    if not value:raise AssertionError(name)
    report['checks'][name]={'passed':True}
def same(a,b):
    return a.stamp==b.stamp and all(torch.equal(x,y) for x,y in zip((a.main_raw,a.index_raw,a.tail_raw,a.state.latent,a.state.compressed,a.state.tail,a.state.tail_positions),(b.main_raw,b.index_raw,b.tail_raw,b.state.latent,b.state.compressed,b.state.tail,b.state.tail_positions)))
def append(state,count,g):
    l=g.layout;pos=torch.arange(state.tokens_seen,state.tokens_seen+count).float()[:,None]
    latent=(torch.sin(pos*.11+torch.arange(l.latent_dim)[None,:]*.03)*.2).bfloat16()
    keys=torch.cos(pos*.13+torch.arange(l.index_dim)[None,:]*.07)*.3;gates=torch.sin(pos*.19+torch.arange(l.index_dim)[None,:]*.05)*.1
    return append_dsa(latent,keys,gates,torch.zeros(l.pool_size,l.index_dim),state,l,start_position=state.tokens_seen)
def initial(start,block):
    l=DSACacheLayout.from_config(config,logical_block_size=block);g=CANNCacheGeometry(l,9*256*4)
    empty=state_from_aligned_prefix(torch.empty(0,512,dtype=torch.bfloat16),torch.empty(0,128,dtype=torch.bfloat16),l);state=append(empty,start,g)
    raw=[torch.full((10*g.main_page_bytes,),0x55,dtype=torch.uint8),torch.full((10*g.index_page_bytes,),0x33,dtype=torch.uint8),torch.full((5*g.tail_page_bytes,),0x66,dtype=torch.uint8)]
    table=(1,4,2)[:(start+block-1)//block]
    for logical,page in enumerate(table):
        first=logical*block;stop=min(start,first+block);raw[0].view(torch.bfloat16).reshape(10,block,512)[page,:stop-first]=state.latent[first:stop]
        first=logical*(block//4);stop=min(start//4,first+block//4);raw[1].view(torch.bfloat16).reshape(10,block//4,128)[page,:stop-first]=state.compressed[first:stop]
    view_tail(raw[2],5,g)[0]=pack_tail_planes(state.tail)
    return g,state,raw,table
def owner(name,g,state,raw,table):return PrivateDSARequest(name,state,*raw,table,0,geometry=g,num_main_pages=10,num_tail_pages=5,output_width=512,max_context_tokens=512)
def reserve(stamp,count,g):
    occupied=frozenset({1,2,4,5,*stamp.logical_table});pages=(stamp.tokens_seen+count-1)//g.layout.logical_block_size-stamp.tokens_seen//g.layout.logical_block_size+1 if count else 0
    ids=tuple(i for i in (8,3,9,6,7,0) if i not in occupied)[:pages];tails=frozenset({0,1,stamp.tail_page_id});tail=next(i for i in (3,4,2) if i not in tails) if count else None
    return ids,dict(tail_private=tail,occupied_main=occupied,occupied_tail=tails)
try:
    c=AbsorbedMLAContract.from_config(config,(0,63));weights=split_kv_b_cpu((torch.randn(1024,512)*.02).bfloat16(),c)
    check('actual_sample_attention_geometry',(c.total_heads,c.qk_dim,c.kv_rank,c.value_dim,c.scale)==(64,256,512,256,1/16))
    for case,(start,count,block) in enumerate(((0,3,128),(3,2,128),(127,5,128),(255,4,256))):
        tag=f'case{case}';g,state,raw,table=initial(start,block);r=owner(tag,g,state,raw,table);before=r.snapshot();raw_before=[t.clone() for t in raw]
        query=(torch.randn(count,2,256)*.1).bfloat16();calls=[]
        def attention(view):
            ids=torch.arange(view.tokens_seen);latent=view.gather(ids);positions=torch.tensor(view.positions,dtype=torch.int64);indices=torch.full((len(positions),2051),-1,dtype=torch.int32)
            for row,pos in enumerate(positions.tolist()):indices[row,:pos+1]=torch.arange(pos+1,dtype=torch.int32)
            first=int(positions[0])-start;out=absorbed_sparse_mla_cpu(query[first:first+len(positions)],latent,indices,positions,weights,sink_policy='absent').values.flatten(1)
            calls.append({'positions':view.positions,'table':view.logical_table});return out
        ids,kw=reserve(r.stamp,count,g);next_state=append(state,count,g)
        branch=r.prepare(next_state,ids,expected_stamp=r.stamp,attention=attention,**kw)
        check(tag+'/prepare_leaves_committed_frame',same(before,r.snapshot()))
        check(tag+'/candidate_attention_sees_new_rows',calls[-1]['positions']==tuple(range(start,start+count)) and calls[-1]['table']==branch.candidate_stamp.logical_table)
        # A later explicit validator failure must preserve state AND every byte.
        def reject(output):raise RuntimeError('injected downstream/head rejection')
        try:r.publish(branch,expected_stamp=before.stamp,validate=reject)
        except RuntimeError:check(tag+'/late_rejection_full_frame_stable',same(before,r.snapshot()) and branch.status=='rejected')
        else:raise AssertionError('validator rejection ignored')
        a_branch=r.prepare(next_state,ids,expected_stamp=r.stamp,attention=attention,**kw);stale=r.prepare(next_state,ids,expected_stamp=r.stamp,attention=attention,**kw)
        output=r.publish(a_branch,expected_stamp=r.stamp,validate=lambda x:None);committed=r.snapshot()
        check(tag+'/published_position_table_revision',committed.stamp.tokens_seen==start+count and committed.stamp.revision==1 and committed.stamp.logical_table==a_branch.candidate_stamp.logical_table)
        check(tag+'/published_state_exact',all(torch.equal(x,y) for x,y in zip((committed.state.latent,committed.state.compressed,committed.state.tail,committed.state.tail_positions),(next_state.latent,next_state.compressed,next_state.tail,next_state.tail_positions))))
        try:r.publish(stale,expected_stamp=before.stamp,validate=lambda x:None)
        except ValueError:check(tag+'/stale_branch_rejected',same(committed,r.snapshot()))
        else:raise AssertionError('stale branch accepted')
        try:r.prepare(append(next_state,1,g),(),expected_stamp=replace(r.stamp,logical_table=()),attention=attention,tail_private=None,occupied_main=frozenset(),occupied_tail=frozenset())
        except ValueError:check(tag+'/stale_table_rejects_before_attention',same(committed,r.snapshot()))
        else:raise AssertionError('stale table accepted')
        segmented=owner(tag+'-seg',g,state,raw,table);outs=[];s=state;offset=0
        for n in (1,count-1):
            if not n:continue
            dest,reservation=reserve(segmented.stamp,n,g);s=append(s,n,g)
            b=segmented.prepare(s,dest,expected_stamp=segmented.stamp,attention=attention,**reservation)
            outs.append(segmented.publish(b,expected_stamp=segmented.stamp,validate=lambda x:None));offset+=n
        final=segmented.snapshot();check(tag+'/segmented_output_exact',torch.equal(torch.cat(outs),output))
        check(tag+'/segmented_state_exact',all(torch.equal(x,y) for x,y in zip((final.state.latent,final.state.compressed,final.state.tail,final.state.tail_positions),(committed.state.latent,committed.state.compressed,committed.state.tail,committed.state.tail_positions))))
        current=segmented.snapshot();empty=segmented.prepare(current.state,(),expected_stamp=segmented.stamp,attention=lambda v:(_ for _ in ()).throw(AssertionError('empty callback')),tail_private=None,occupied_main=frozenset(current.stamp.logical_table),occupied_tail=frozenset({current.stamp.tail_page_id}))
        empty_out=segmented.publish(empty,expected_stamp=segmented.stamp,validate=lambda v:(_ for _ in ()).throw(AssertionError('empty validator')))
        check(tag+'/empty_no_revision_or_state_change',same(current,segmented.snapshot()) and empty_out.shape==(0,512))
        check(tag+'/caller_source_backings_unchanged',all(torch.equal(x,y) for x,y in zip(raw,raw_before)))
        check(tag+'/old_snapshot_survives_publication',before.stamp.tokens_seen==start and torch.equal(before.state.latent,state.latent))
        report['cases'].append({'start':start,'tokens':count,'block_size':block,'output_shape':list(output.shape),'candidate_table':committed.stamp.logical_table,'frame_backing_bytes':sum(t.numel()*t.element_size() for t in raw),'attention_calls':len(calls)})
        r.close();segmented.close()
    report['status']='passed'
except BaseException as exc:report.update(status='failed',failure=repr(exc));raise
finally:
    a.output_dir.mkdir(parents=True,exist_ok=True);report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'private-request-result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
