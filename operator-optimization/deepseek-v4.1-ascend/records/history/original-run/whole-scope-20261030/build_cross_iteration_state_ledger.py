#!/usr/bin/env python3
"""Persist explicit local value/state edges; every global consumer gate stays open."""
import datetime
import hashlib
import json
from pathlib import Path

W=Path(__file__).resolve().parent;R=W.parent
read=lambda p:json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
files=read(W/'coverage-ledger-v2/source-files.json')
units=read(W/'coverage-ledger-v2/source-units.json')
def ref(suffix,line):
    found=[x for x in files if x['path'].startswith('vllm-ascend/') and x['path'].endswith(suffix)]
    assert len(found)==1,(suffix,len(found))
    f=found[0];assert hashlib.sha256(Path(f['local_path']).read_bytes()).hexdigest()==f['sha256']
    owners=[x for x in units if x['source']==f['path'] and x['kind'] in ('function','async_function') and x['line']<=line<=x['end_line']]
    assert owners,(suffix,line)
    owner=min(owners,key=lambda x:x['end_line']-x['line'])
    return {'source':f['path'],'source_sha256':f['sha256'],'line':line,'containing_unit_id':owner['id'],'owner':owner['qualified_name']}
worker='worker/model_runner_v1.py';base='spec_decode/llm_base_proposer.py';dspark='spec_decode/dspark_proposer.py';kernel='ops/triton/spec_decode/utils.py'
# Producer/consumer descriptions name versions. Lists are deliberately local subsets.
definitions=[
 ('XI01','sampler_output.sampled_token_ids[t]','padded next-token preparation[t]',worker,2033,'current→next draft','original is retained; helper clones before discard masking'),
 ('XI02','request history/req_ids and CPU num_tokens_no_spec[t]','backup_next_token_ids CPU→GPU[t]',base,2212,'current request mapping','host mapping and uploaded fallback; validity and reuse not proven'),
 ('XI03','sampled clone + discard indices + vocab_size[t]','valid mask and valid_count[t]',base,2231,'same step','clone prevents mutation of sampled output; mask intermediates local'),
 ('XI04','valid_count + sampled clone + backup[t]','next_token_ids[t]',base,2246,'same step','where result; count-1 gather assumes valid prefix as original code'),
 ('XI05','valid_count device[t]','side-stream D2H + count_event[t]',worker,1884,'current→next host mirror','wait current stream; nonblocking copy; record event'),
 ('XI06','valid_count device[t]','valid_sampled_token_count_gpu for next _prepare_inputs[t+1]',worker,1893,'cross iteration','same tensor retained when async spec; producer completion and lifetime required'),
 ('XI07','next_token_ids[t]','input_batch.prev_sampled_token_ids[t]',worker,1896,'cross iteration','unsqueeze view; request ordering and storage lifetime required'),
 ('XI08','cu draft counts + valid_count + query_start_loc[t]','rejected_count + token_indices_to_sample[t]',kernel,54,'same step','existing fused prepare_inputs_padded_kernel; inclusive cumulative counts'),
 ('XI09','next_token_ids[t]','persistent _dspark_seed_buffer[t]',dspark,242,'next draft','copy prefix and zero remainder; not merely an alias'),
 ('XI10','selected target auxiliary hidden[t]','_dflash_hidden_states context buffer[t]',dspark,259,'cross model','assignment copies selected context prefix; dtype/capacity and all readers remain open'),
 ('XI11','per-group block tables/slots + rejected_count[t]','query/context positions + per-group slots[t]',dspark,274,'same draft','shared output buffers plus distinct per-group slot buffers; exact group identity open'),
 ('XI12','seq_len + rejected_count[t]','draft effective seq_len[t]',dspark,315,'same draft','subtract rejected then add query width; host mirror follows separate branch'),
 ('XI13','persistent seed[t]','draft_buffer[:,0][t]',base,1584,'same draft','copy seed into persistent buffer, later Markov loop writes successive columns'),
 ('XI14','draft_buffer[:,i] + base_logits[:,i][t]','draft_buffer[:,i+1][t]',base,1587,'cross draft position','Markov embed→bias→inplace add→argmax/sample; alias and arithmetic barriers matter'),
 ('XI15','draft_buffer[:,1:][t]','worker self._draft_token_ids[t]',base,1643,'cross function/iteration','returned view remains backed by persistent buffer; all last reads not closed'),
 ('XI16','worker draft token view[t]','conditional CPU draft copy[t]',worker,2175,'same/next host schedule','async fast guard can return; otherwise stream/event and actual K width apply'),
 ('XI17','prev device count + request mapping + previous positions/drafts[t]','corrected num_computed_tokens[t+1]',worker,1522,'cross iteration','GPU correction; same request identity and lifetime required'),
 ('XI18','host count copy + count_event[t]','corrected optimistic_seq_lens_cpu[t+1]',worker,1867,'cross iteration','synchronize precedes host read; actual wait duration UNKNOWN'),
 ('XI19','corrected num_computed_tokens[t+1]','positions/seq_lens/block slot mapping[t+1]',worker,1608,'next target','downstream row ownership and physical KV state remain open'),
]
edges=[]
for ident,producer,consumer,suffix,line,domain,alias in definitions:
    edges.append({'id':ident,'producer_version':producer,'reviewed_consumer_subset':consumer,'domain':domain,
        'alias_or_order_contract':alias,'source_evidence':ref(suffix,line),'fact_kind':'DOCUMENTED_LOCAL_SOURCE_CONTRACT',
        'runtime_tensor_event_stream_identity':'UNKNOWN','all_consumers_closed':False,'global_coverage_closed':False})
assert len({x['id'] for x in edges})==len(edges)
data={'status':'PARTIAL_LOCAL_STATE_GRAPH','scope':'frozen Ascend V1 padded path / DSpark; conditions and global consumers remain open',
    'edges':edges,'candidate':'WS-NEXT-01','candidate_status':'CONDITIONAL_UNIMPLEMENTED_NO_MEASURED_GAIN',
    'overlaps':['F31','F33','WS-REJECT-01 for acceptance preparation boundaries'],
    'all_modules_closed':False,'overall_complete':False,'review':'CROSS-ITERATION-REVIEW.md'}
save(W/'cross-iteration-state-ledger.json',data)

# Attach a local partial review while keeping all acceptance gates open.
for name in ['STATE.json','module-coverage-ledger.json','PROGRESS.md','recovery-checkpoint-verification.json']:
    source=W/name;destination=W/(name+'.before-cross-iteration')
    assert not destination.exists();destination.write_bytes(source.read_bytes())
module=read(W/'module-coverage-ledger.json')
for m in module['surfaces']:
    if m['id'] in ('M04','M17','M21','M22','M24','M25'):
        m.setdefault('local_review_evidence',[]).extend(['CROSS-ITERATION-REVIEW.md','cross-iteration-state-ledger.json'])
        m['mandatory_gates']['cross_stream_rank_iteration_dependencies']['status']='PARTIAL_LOCAL_VERSIONED_STATE_EDGES; GLOBAL_AND_RUNTIME_OPEN'
for row in module['rule_coverage']:
    if row['surface'] in ('M17','M21','M24') and row['rule'] in ('R01','R04','R08','R10'):
        row['status']='PARTIAL_LOCAL_CONDITIONAL_REVIEW; OTHER_BOUNDARIES_OPEN'
        row.setdefault('evidence',[]).append('CROSS-ITERATION-REVIEW.md')
assert all(not m['complete'] for m in module['surfaces']) and not module['overall_complete']
save(W/'module-coverage-ledger.json',module)
state=read(W/'STATE.json');state['candidate_local_reviews'].append('WS-NEXT-01')
state['local_state_edge_ledger']='cross-iteration-state-ledger.json'
state['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
state['next_action']='继续检查M24全消费者/alias与请求重排闭包，关联同次事件和host等待；同时按真实import补M00/M01/M05的初始化/前端范围。MoE有效行/容量、Indexer长度与分层流量证据仍待补。'
assert state['goal_status']=='active' and not state['overall_complete'] and state['whole_modules_closed']==0
save(W/'STATE.json',state)
with (W/'PROGRESS.md').open('a') as out:
    out.write('\n追加跨迭代实质审阅：已按冻结源码展开19条局部版本化状态边，覆盖拒绝输出、next token/count、seed、持久draft view、side-stream复制、CPU镜像和下一轮device修正。见[局部状态图](CROSS-ITERATION-REVIEW.md)。WS-NEXT-01为条件性重组，未测收益；全部消费者/真实事件仍OPEN，0个整模块关闭。\n')

catalog=read(R/'RESEARCH-CATALOG.json')
existing={x['relative_path']:x for x in catalog['indexed_files']}
for name in ['STATE.json','module-coverage-ledger.json','PROGRESS.md','CROSS-ITERATION-REVIEW.md','cross-iteration-state-ledger.json']:
    p=W/name;relative=str(p.relative_to(R));b=p.read_bytes()
    item=existing.get(relative,{})
    item.update(relative_path=relative,local_path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),
        current_local_record=True,server_scope='Updated local checkpoint; this revision has not been synchronized to remote/public copies.')
    existing[relative]=item
catalog['indexed_files']=list(existing.values());catalog['continuation_updated_at_utc']=state['updated_at_utc']
save(R/'RESEARCH-CATALOG.json',catalog)
save(W/'cross-iteration-checkpoint-verification.json',{'status':'PASS_SOURCE_REFERENCES_AND_OPEN_STATE_ONLY','local_state_edges':len(edges),
    'selected_local_branch_contracts_unchanged':23,'whole_modules_closed':0,'overall_goal_complete':False})
print(json.dumps({'status':'PARTIAL_LOCAL_STATE_GRAPH_PERSISTED','local_edges':len(edges),'whole_modules_closed':0,'overall_goal_complete':False}))
