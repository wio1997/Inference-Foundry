"""Check that each graph's packed counter copy occurs before the next writer."""
import bisect,collections,concurrent.futures,json,os,sqlite3,time,traceback
from pathlib import Path
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168');O=P.parent/'joint169'
review=json.loads((O/'MC2-NATIVE-JOINT-REVIEW.json').read_text())
def one(item):
 uid=item['process_uuid'];c=sqlite3.connect('file:'+item['database']+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row;s=dict(c.execute('select id,value from STRING_IDS'))
 cat_ids=[];target_ids=[];types=set()
 for r in c.execute('select * from COMPUTE_TASK_INFO'):
  name=s[r['name']];shape=s.get(r['outputShapes'],'').strip('"');inp=s.get(r['inputShapes'],'').strip('"').split(';');dtype=s.get(r['outputDataTypes'],'')
  if shape=='26' and dtype=='INT32' and [a for a in inp if a not in ('','0')]==['24','1','1']:cat_ids.append(r['globalTaskId'])
  if name=='DispatchFFNCombine':
   types.add(s[r['inputDataTypes']])
   if len(inp)>1 and inp[1]=='24,5120,4608':target_ids.append(r['globalTaskId'])
 assert cat_ids and target_ids
 ph=lambda n:','.join('?' for _ in range(n))
 cats=[dict(r) for r in c.execute('select rowid TASK_ROW,* from TASK where modelId!=4294967295 and globalTaskId in ('+ph(len(cat_ids))+') order by startNs',cat_ids)]
 writers=[dict(r) for r in c.execute('select rowid TASK_ROW,* from TASK where globalTaskId in ('+ph(len(target_ids))+') order by startNs',target_ids)]
 joins=[json.loads(x) for x in (O/(uid+'-MC2-JOINS.jsonl')).read_text().splitlines()];replays=json.loads((O/(uid+'-REPLAYS.json')).read_text());envs=sorted(replays,key=lambda e:e['start_ns']);starts=[e['start_ns'] for e in envs];cat_by_env=collections.defaultdict(list)
 for cat in cats:
  i=bisect.bisect_right(starts,cat['startNs'])-1
  assert i>=0 and envs[i]['start_ns']<=cat['startNs']<=cat['endNs']<=envs[i]['end_ns'] and cat['modelId']==envs[i]['native_model_id']
  cat_by_env[envs[i]['launch_TASK_ROW']].append(cat)
 rows_by_env=collections.defaultdict(list)
 for j in joins:rows_by_env[j['evidence']['native_launch_TASK_ROW']].append(j)
 writer_pos={r['TASK_ROW']:i for i,r in enumerate(writers)};issues=[];checked=0
 for env in envs:
  rs=sorted(rows_by_env[env['launch_TASK_ROW']],key=lambda r:r['native_start_ns']);cs=cat_by_env[env['launch_TASK_ROW']]
  if len(rs)!=len(cs):issues.append({'kind':'COPY_COUNT_MISMATCH','launch':env['launch_TASK_ROW'],'writers':len(rs),'copies':len(cs)});continue
  for r,cat in zip(rs,cs):
   i=writer_pos[r['native_TASK_ROW']];next_start=writers[i+1]['startNs'] if i+1<len(writers) else env['end_ns']
   good=r['native_end_ns']<=cat['startNs']<=cat['endNs']<=min(env['end_ns'],next_start) and r['native_stream_id']==cat['streamId']
   if not good:issues.append({'kind':'COUNTER_COPY_ORDER_NOT_PROVED','TASK_ROW':r['native_TASK_ROW'],'copy_TASK_ROW':cat['TASK_ROW']})
   else:checked+=1
 c.close();return {'pid':item['pid'],'process_uuid':uid,'database_sha256_reused':item['database_sha256'],'target_native_writers':len(writers),'graph_packed_copies':len(cats),'checked':checked,'issues':issues[:16],'issues_count':len(issues),'native_input_datatypes':sorted(types),'all_mask_types_BOOL':bool(types) and all(t.split(';')[-1]=='BOOL' for t in types)}
if __name__=='__main__':
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,review['profiles']))
  out={'status':'ACTUAL_NATIVE_COUNTER_SNAPSHOT_ORDER_AUDIT','profiles':rows,'checked':sum(r['checked'] for r in rows),'issues_count':sum(r['issues_count'] for r in rows),'all_mask_types_BOOL':all(r['all_mask_types_BOOL'] for r in rows),'domain':'Dedicated INT32 Cat(24,1,1)->26 payload in this controlled observer;source Tensor identity comes from pinnedhook andshared expert_token_nums. Eachpack materializes after itsnativewriter andbeforethenexttargetwriter,onthatwriterstream;notbaseline-reader evidence.','no_new_device_work':True,'overall_complete':False}
  (O/'OBSERVER-SNAPSHOT-ORDER-AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='profiles'}))
 except BaseException:(O/'OBSERVER-SNAPSHOT-AUDIT-ERROR.json').write_text(json.dumps({'error':traceback.format_exc()},indent=2)+'\n');raise
