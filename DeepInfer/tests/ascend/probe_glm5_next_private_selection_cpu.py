# SPDX-License-Identifier: Apache-2.0
"""Bounded actual-dimension CPU page/selection/MLA bridge,synthetic producers."""
import argparse,datetime,hashlib,json,os,platform
from dataclasses import replace
from pathlib import Path
import torch
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout,state_from_aligned_prefix,append_dsa
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry,view_tail,pack_tail_planes
from deepinfer.model_executor.models.glm5_next.private_request import PrivateDSARequest
from deepinfer.model_executor.models.glm5_next.private_selection import private_absorbed_mla_cpu
from deepinfer.model_executor.models.glm5_next.absorbed_mla import AbsorbedMLAContract,split_kv_b_cpu,absorbed_sparse_mla_cpu
from deepinfer.model_executor.models.glm5_next.sparse_reference import select_pools_reference

p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','prior-evidence','output-dir'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);torch.manual_seed(5335);config=Glm5NextConfig.from_checkpoint(a.checkpoint);prior=json.loads(a.prior_evidence.read_text());sha=hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest();assert prior['status']=='passed' and prior['config_sha256']==sha and not prior['NPU_execution']
src=Path(__file__).resolve().parents[2];files=['deepinfer/model_executor/models/glm5_next/'+n+'.py' for n in ('private_pages','private_request','private_selection')]+['tests/ascend/test_glm5_next_private_selection.py','tests/ascend/probe_glm5_next_private_selection_cpu.py']
r={'status':'in_progress','native_round':35,'architecture':platform.machine(),'torch':torch.__version__,'config_sha256':sha,'prior_evidence_sha256':hashlib.sha256(a.prior_evidence.read_bytes()).hexdigest(),
 'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},'NPU_execution':False,'native_imports':False,'native_ABI_validated':False,'model_generation':False,'full_model_forward':False,'weights_read_bytes':0,
 'scope':'one DSA layer at actual dimensions;synthetic completed producer/query/selected-head weights;no actualINT8/full45-layer execution','checks':{},'cases':[],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def check(name,value):
    if not value:raise AssertionError(name)
    r['checks'][name]={'passed':True}
def append(state,count,g):
    pos=torch.arange(state.tokens_seen,state.tokens_seen+count)[:,None].float();l=g.layout
    latent=(torch.sin(pos*.11+torch.arange(l.latent_dim)[None,:]*.03)*.2).bfloat16();keys=torch.cos(pos*.13+torch.arange(l.index_dim)[None,:]*.07)*.3;gates=torch.zeros_like(keys)
    result=append_dsa(latent,keys,gates,torch.zeros(l.pool_size,l.index_dim),state,l,start_position=state.tokens_seen)
    # Explicit synthetic completed compression rows isolate page visibility/ranking.
    compressed=result.compressed.clone()
    for pool in range(state.tokens_seen//l.pool_size,result.tokens_seen//l.pool_size):compressed[pool]=10+pool/1024
    return replace(result,compressed=compressed)
def initial(start,block):
    l=DSACacheLayout.from_config(config,logical_block_size=block);g=CANNCacheGeometry(l,9*256*4);aligned=start//4*4
    pos=torch.arange(aligned)[:,None].float();latent=(torch.sin(pos*.11+torch.arange(512)[None,:]*.03)*.2).bfloat16()
    comp=(torch.arange(aligned//4)[:,None].remainder(97).float()/128).expand(-1,128).clone().bfloat16()
    if len(comp):comp[0]=-100
    state=state_from_aligned_prefix(latent,comp,l)
    if start>aligned:state=append(state,start-aligned,g)
    raw=[torch.zeros(24*g.main_page_bytes,dtype=torch.uint8),torch.zeros(24*g.index_page_bytes,dtype=torch.uint8),torch.full((5*g.tail_page_bytes,),0x55,dtype=torch.uint8)]
    table=tuple(range(1,(start+block-1)//block+1))
    for logical,page in enumerate(table):
        first=logical*block;last=min(start,first+block);raw[0].view(torch.bfloat16).reshape(24,block,512)[page,:last-first]=state.latent[first:last]
        first=logical*(block//4);last=min(start//4,first+block//4);raw[1].view(torch.bfloat16).reshape(24,block//4,128)[page,:last-first]=state.compressed[first:last]
    view_tail(raw[2],5,g)[0]=pack_tail_planes(state.tail)
    return g,state,raw,table
def owner(name,g,state,raw,table):return PrivateDSARequest(name,state,*raw,table,0,geometry=g,num_main_pages=24,num_tail_pages=5,output_width=512,max_context_tokens=4096)
def reserve(stamp,count,g):
    used=frozenset(stamp.logical_table);n=(stamp.tokens_seen+count-1)//g.layout.logical_block_size-stamp.tokens_seen//g.layout.logical_block_size+1
    private=tuple(i for i in range(24) if i not in used)[:n];tail=next(i for i in range(5) if i!=stamp.tail_page_id)
    return private,dict(tail_private=tail,occupied_main=used,occupied_tail=frozenset({stamp.tail_page_id}))
def same(a,b):
    return a.stamp==b.stamp and all(torch.equal(getattr(a,n),getattr(b,n)) for n in ('main_raw','index_raw','tail_raw')) and all(torch.equal(getattr(a.state,n),getattr(b.state,n)) for n in ('latent','compressed','tail','tail_positions'))
try:
    contract=AbsorbedMLAContract.from_config(config,(0,63));weights=split_kv_b_cpu((torch.randn(1024,512)*.02).bfloat16(),contract)
    check('actual_attention_geometry_and_scale',(contract.total_heads,contract.qk_dim,contract.kv_rank,contract.value_dim,contract.scale)==(64,256,512,256,1/16))
    check('actual_index_heads',config.indexer_config['index_n_heads']==32)
    for case,(start,count,block) in enumerate(((127,5,128),(255,4,256),(2051,4,128))):
        tag=f'case{case}';g,state,raw,table=initial(start,block);q=owner(tag,g,state,raw,table);before=q.snapshot();next_state=append(state,count,g);observed=[]
        def attention(view):
            positions=torch.tensor(view.positions);iq=torch.ones(len(positions),32,128,dtype=torch.bfloat16);hw=torch.full((len(positions),32),1/32,dtype=torch.bfloat16);query=torch.full((len(positions),2,256),1/16,dtype=torch.bfloat16)
            actual=private_absorbed_mla_cpu(view,iq,hw,query,weights,sink_policy='absent')
            check(tag+'/candidate_compressed_exact/'+str(int(positions[0])),torch.equal(view.gather_compressed(torch.arange(view.tokens_seen//4)),next_state.compressed[:view.tokens_seen//4]))
            expected=select_pools_reference(iq,hw,next_state.compressed,positions,g.layout)
            check(tag+'/dense_selection_exact/'+str(int(positions[0])),torch.equal(actual.selection.pool_ids,expected.pool_ids) and torch.equal(actual.selection.pool_scores,expected.pool_scores) and torch.equal(actual.selection.token_indices,expected.token_indices))
            oracle=absorbed_sparse_mla_cpu(query,next_state.latent,expected.token_indices,positions,weights,sink_policy='absent')
            check(tag+'/dense_absorption_exact/'+str(int(positions[0])),torch.equal(actual.attention.values,oracle.values) and torch.equal(actual.attention.lse,oracle.lse))
            check(tag+'/per_query_pool_visibility/'+str(int(positions[0])),torch.equal(actual.selection.visible_pools,(positions+1)//4))
            ids=actual.selection.token_indices
            check(tag+'/tokens_causal/'+str(int(positions[0])),bool(((ids<0)|(ids<=positions[:,None])).all()))
            if start==2051:
                check(tag+'/new_pool512_selected_and_pool0_dropped/'+str(int(positions[0])),bool((actual.selection.pool_ids==512).any(1).all()) and not bool((actual.selection.pool_ids==0).any()))
            observed.append(actual.selection);return actual.attention.values.flatten(1)
        private,kw=reserve(q.stamp,count,g);branch=q.prepare(next_state,private,expected_stamp=q.stamp,attention=attention,**kw)
        check(tag+'/prepare_leaves_full_committed_frame',same(before,q.snapshot()))
        # A stale completed-cache surrogate misses appended pool rows and changes ranking.
        iq=torch.ones(count,32,128,dtype=torch.bfloat16);hw=torch.full((count,32),1/32,dtype=torch.bfloat16);positions=torch.arange(start,start+count)
        stale=torch.cat((state.compressed,torch.full((next_state.tokens_seen//4-state.tokens_seen//4,128),-1000,dtype=torch.bfloat16)))
        wrong=select_pools_reference(iq,hw,stale,positions,g.layout)
        check(tag+'/stale_compressed_rows_change_selection',not torch.equal(wrong.pool_ids,observed[0].pool_ids))
        output=q.publish(branch,expected_stamp=q.stamp,validate=lambda x:None);check(tag+'/publish_state_exact',torch.equal(q.snapshot().state.compressed,next_state.compressed) and q.stamp.tokens_seen==start+count)
        segmented=owner(tag+'-seg',g,state,raw,table);current=state;outs=[]
        for n in (1,count-1):
            current=append(current,n,g);p,kw=reserve(segmented.stamp,n,g);b=segmented.prepare(current,p,expected_stamp=segmented.stamp,attention=attention,**kw);outs.append(segmented.publish(b,expected_stamp=segmented.stamp,validate=lambda x:None))
        check(tag+'/segmented_output_bitwise',torch.equal(torch.cat(outs),output));check(tag+'/segmented_final_state_bitwise',all(torch.equal(getattr(segmented.snapshot().state,n),getattr(next_state,n)) for n in ('latent','compressed','tail','tail_positions')))
        r['cases'].append({'start':start,'tokens':count,'block_size':block,'completed_pools':next_state.tokens_seen//4,'sample_head_ids':[0,63],'index_heads':32,'output_shape':list(output.shape),'raw_frame_backing_bytes':sum(v.numel()*v.element_size() for v in raw),'candidate_pool_first_ids':observed[0].pool_ids[:,0].tolist()})
        q.close();segmented.close()
    r['status']='passed'
except BaseException as e:r.update(status='failed',failure=repr(e));raise
finally:
    r['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'private-selection-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
