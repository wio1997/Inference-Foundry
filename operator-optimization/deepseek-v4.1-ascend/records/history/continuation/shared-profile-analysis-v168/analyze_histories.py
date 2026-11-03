"""CPU-only review of actual GPU histories;no inference,profile,or graph replay."""
import collections,hashlib,json,statistics,time
from pathlib import Path
import torch
ROOT=Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/joint168')
OUT=ROOT/'derived';OUT.mkdir(exist_ok=True);SENTINEL=-2147483648
torch.set_num_threads(1)
def stats(xs):return {'n':len(xs),'min':min(xs),'median':statistics.median(xs),'max':max(xs),'mean':statistics.mean(xs)} if xs else {'n':0}
def one(p):
 meta=json.loads(p.read_text());path=p.parent/'history.pt';data=torch.load(path,map_location='cpu',weights_only=True);history=data['history'];ids=data['ids']
 assert history.dtype==torch.int32 and ids.dtype==torch.int64 and history.ndim==3 and history.shape[2]==26
 assert history.shape[0]==len(meta['banks']) and history.shape[1]==meta['history_last_index']+1
 events=[json.loads(line) for line in (p.parent/'events.jsonl').read_text().splitlines()];expected=set();eager={}
 for e in events:
  if e['kind']=='eager_mc2_binding':eager[(e['bank_slot'],e['history_slot'])]=e;expected.add((e['bank_slot'],e['history_slot']))
  elif e['kind']=='replay_scope' and e['armed'] and e['root'] and e['history_epoch']:
   expected.update((b['slot'],e['history_epoch']) for b in meta['banks'] if b['graph_token']==e['graph_token'] and b['capture_epoch']==e['capture_epoch'])
 rows=[];seen=set();issues=[];selected_ids=[]
 for b in meta['banks']:
  E=b['E_local'];h=history[b['slot']];filled=(h[:,E+1]!=SENTINEL).nonzero().flatten().tolist()
  for epoch in filled:
   if epoch==0:continue
   v=h[epoch];counts=v[:E].tolist();valid=int(v[E]);stamp=int(v[E+1]);source=eager.get((b['slot'],epoch));M=source['M'] if source else b['M_at_creation'];K=source['K'] if source else b['K_at_creation'];seen.add((b['slot'],epoch))
   good=stamp==epoch and all(x>=0 for x in counts) and 0<=valid<=M
   if not good:issues.append({'bank':b['slot'],'epoch':epoch,'device_epoch':stamp,'valid':valid,'M':M,'counts':counts})
   n=sum(counts);capacity=b['capacity_max_output_size'];executed=min(n,capacity)
   r={'pid':meta['pid'],'process_uuid':meta['process_uuid'],'bank_slot':b['slot'],'history_epoch':epoch,'device_epoch':stamp,'role':b['role'],'layer_name':b['layer_name'],'target_layer_observation_order':b['target_layer_observation_order'],'execution_kind':b['execution_kind'],
    'graph_token':b['graph_token'],'capture_epoch':b['capture_epoch'],'capture_site':b['capture_site'],'E_local':E,'M':M,'K':K,'valid_sender_rows':valid,'receiver_counts':counts,'receiver_rows':n,'empty_local_experts':sum(x==0 for x in counts),'capacity':capacity,'prefix_clipped_executed_rows_if_source_contract':executed,
    'conditional_target_direct_MAC':35389440*executed if b['role']=='target' else None,'ep_rank':b['ep_rank'],'dp_rank':b['dp_rank'],'tp_rank':b['tp_rank'],'row_values_valid':good}
   rows.append(r)
   if b['id_slot'] is not None:
    raw=ids[b['id_slot'],epoch,:M*K].reshape(M,K);distinct=all(len(set(row))==K for row in raw.tolist());inrange=bool(((raw>=0)&(raw<b['E_global'])).all())
    selected_ids.append({'bank_slot':b['slot'],'history_epoch':epoch,'M':M,'K':K,'valid_sender_rows':valid,'all_physical_rows_distinct':distinct,'all_physical_ids_in_range':inrange,'active_duplicate_status':'NO_ACTIVE_DUPLICATES_IF_MASK_BOOL' if distinct else ('ACTIVE_DUPLICATE_IF_MASK_BOOL_AND_ALL_ROWS_ACTIVE' if valid==M else 'UNKNOWN_ACTIVE_ROWS_WITHOUT_MASK_POSITIONS'),'ids':raw.tolist()})
 outpath=OUT/(meta['process_uuid']+'-rows.jsonl')
 with outpath.open('w') as f:
  for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
 (OUT/(meta['process_uuid']+'-selected-ids.json')).write_text(json.dumps(selected_ids,separators=(',',':'))+'\n')
 byrole={}
 for role in ['target','draft']:
  rs=[r for r in rows if r['role']==role];byrole[role]={'records':len(rs),'layers':sorted({r['layer_name'] for r in rs}),'M_histogram':dict(collections.Counter(r['M'] for r in rs)),'valid_rows':stats([r['valid_sender_rows'] for r in rs]),'receiver_rows':stats([r['receiver_rows'] for r in rs]),'empty_local_experts':stats([r['empty_local_experts'] for r in rs]),'raw_rows_above_capacity':sum(r['receiver_rows']>r['capacity'] for r in rs),'first_MoE_receiver_rows':stats([r['receiver_rows'] for r in rs if r['target_layer_observation_order']==0]),'comparison_MoE_receiver_rows':stats([r['receiver_rows'] for r in rs if r['target_layer_observation_order']==20])}
 return {'pid':meta['pid'],'process_uuid':meta['process_uuid'],'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'epochs':meta['epochs'],'overflow':meta['overflow'],'banks':len(meta['banks']),'rows':len(rows),'invalid_value_rows':issues[:32],'invalid_value_row_count':len(issues),'expected_pairs':len(expected),'observed_pairs':len(seen),'missing_pairs':len(expected-seen),'unexpected_pairs':len(seen-expected),'missing_pair_sample':sorted(expected-seen)[:16],'unexpected_pair_sample':sorted(seen-expected)[:16],'by_role':byrole,'selected_id_records':len(selected_ids),'selected_all_physical_rows_distinct':sum(x['all_physical_rows_distinct'] for x in selected_ids),'selected_all_physical_ids_in_range':sum(x['all_physical_ids_in_range'] for x in selected_ids),'rows_path':str(outpath)}
if __name__=='__main__':
 status=json.loads((ROOT/'STATUS.json').read_text());assert status['state'] in ['ORIGINAL_SERVICE_RESTORED_REVIEW_PENDING','ACQUISITION_FAILED_ORIGINAL_SERVICE_RESTORED'],'Do not analyze mutable live histories'
 finals=sorted((ROOT/'workers').glob('*/FINAL.json'));workers=[one(p) for p in finals]
 result={'status':'ACTUAL_GPU_HISTORY_REVIEW','generated_unix':time.time(),'workers':workers,'workers_count':len(workers),'rows':sum(w['rows'] for w in workers),'invalid_value_rows':sum(w['invalid_value_row_count'] for w in workers),'missing_history_pairs':sum(w['missing_pairs'] for w in workers),'unexpected_history_pairs':sum(w['unexpected_pairs'] for w in workers),'overflows':sum(w['overflow'] for w in workers),
  'scope':['Device epoch stamp must equal the history row;source and native invocation join are separate DB steps.','Root replay/source graph tokens and process incarnation do not provide a global EP operation ordinal.','MC2 counts and mask sums are new observer reads;not evidence that baseline buffers had those readers.','Target MAC formula remains conditional on matched actual native shapes and executed-prefix source contract;not a physical resource upper.','Detailed IDs cover selected small-row sites;partial valid count does not identify active mask positions.'],'overall_complete':False}
 (OUT/'HISTORY-REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='workers'}))
