"""Exact native replay/eager API links plus explicitly scoped graph-order mapping.

CPU-only. Uses indexes and disjoint intervals;never a task x replay sweep.
"""
import bisect,collections,concurrent.futures,hashlib,json,os,sqlite3,statistics,time,traceback
from pathlib import Path
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168')
OUT=P.parent/'joint169';OUT.mkdir(exist_ok=True)
UINT=4294967295
FINAL={}
for p in (P/'workers').glob('*/FINAL.json'):
 d=json.loads(p.read_text());FINAL[d['pid']]=(p,d)
def stat(xs):return {'n':len(xs),'min':min(xs),'median':statistics.median(xs),'max':max(xs),'mean':statistics.mean(xs)} if xs else {'n':0}
def one(db):
 c=sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
 s=dict(c.execute('select id,value from STRING_IDS'));ids=lambda value:[k for k,v in s.items() if v==value]
 fids=ids('DispatchFFNCombine');assert len(fids)==1
 native=[dict(r) for r in c.execute('select t.rowid TASK_ROW,t.*,x.inputShapes,x.outputShapes from TASK t join COMPUTE_TASK_INFO x on t.globalTaskId=x.globalTaskId where x.name=? order by t.startNs,t.endNs,t.rowid',(fids[0],))]
 pids={r['globalPid'] for r in native};assert len(pids)==1;pidi=next(iter(pids));mp,meta=FINAL[pidi];uid=meta['process_uuid']
 cache=OUT/(uid+'-SUMMARY.json')
 if cache.exists():c.close();return json.loads(cache.read_text())
 ev=[json.loads(x) for x in (mp.parent/'events.jsonl').read_text().splitlines()]
 eager={e['source_marker']:e for e in ev if e['kind']=='eager_mc2_binding'}
 replays={e['marker']:e for e in ev if e['kind']=='replay_scope' and e['armed'] and e['root']}
 hist={}
 for line in (P/'derived'/(uid+'-rows.jsonl')).read_text().splitlines():
  r=json.loads(line);hist[(r['bank_slot'],r['history_epoch'])]=r
 banks={b['slot']:b for b in meta['banks']};marks=[dict(r,message_text=s[r['message']]) for r in c.execute('select * from MSTX_EVENTS') if s[r['message']] in eager or s[r['message']] in replays]
 mark_by_message=collections.defaultdict(list)
 for m in marks:mark_by_message[m['message_text']].append(m)
 assert all(len(v)==1 for v in mark_by_message.values()),'MSTXmessage not unique'
 # Fetch only launch/model APIs,then connect via official native connectionID.
 api_names=ids('launch')+ids('aclmdlRIExecuteAsync')
 apis=[dict(r,name_text=s[r['name']]) for r in c.execute('select rowid API_ROW,* from CANN_API where name in ('+','.join('?' for _ in api_names)+')',api_names)]
 by_conn=collections.defaultdict(list);model_by_tid=collections.defaultdict(list)
 for a in apis:
  by_conn[a['connectionId']].append(a)
  if a['name_text']=='aclmdlRIExecuteAsync':model_by_tid[a['globalTid']].append(a)
 for rows in model_by_tid.values():rows.sort(key=lambda a:a['startNs'])
 launchtypes=ids('MODEL_EXECUTE');waittypes=ids('MODEL_WAIT_COMPLETE');assert len(launchtypes)==len(waittypes)==1
 modeltasks=[dict(r) for r in c.execute('select rowid TASK_ROW,* from TASK where taskType in (?,?)',(launchtypes[0],waittypes[0]))]
 launches=collections.defaultdict(list);waits=collections.defaultdict(list)
 for r in modeltasks:
  key=tuple(r[k] for k in ['connectionId','deviceId','globalPid','streamId','modelId'])
  (launches if r['taskType']==launchtypes[0] else waits)[key].append(r)
 models_by_conn=collections.defaultdict(list)
 for key,ls in launches.items():
  ws=waits.get(key,[])
  if len(ls)==len(ws)==1:models_by_conn[key[0]].append((ls[0],ws[0]))
 envelopes=[];issues=[]
 for message,e in replays.items():
  ms=mark_by_message.get(message,[])
  if len(ms)!=1:issues.append({'kind':'REPLAY_MSTX_NOT_UNIQUE','message':message,'matches':len(ms)});continue
  m=ms[0];candidates=[a for a in model_by_tid.get(m['globalTid'],[]) if m['startNs']<=a['startNs']<=a['endNs']<=m['endNs']]
  if len(candidates)!=1:issues.append({'kind':'REPLAY_API_NOT_UNIQUE','message':message,'matches':len(candidates)});continue
  a=candidates[0];parents=models_by_conn.get(a['connectionId'],[])
  if len(parents)!=1:issues.append({'kind':'REPLAY_NATIVE_PARENT_NOT_UNIQUE','connection':a['connectionId'],'matches':len(parents)});continue
  first,last=parents[0];assert first['globalPid']==meta['pid'] and first['modelId']!=UINT and first['startNs']<=last['endNs']
  envelopes.append({'marker':message,'source_event':e['event'],'graph_token':e['graph_token'],'capture_epoch':e['capture_epoch'],'history_epoch':e['history_epoch'],'native_model_id':first['modelId'],'device_id':first['deviceId'],'native_pid':first['globalPid'],'native_parent_stream':first['streamId'],'native_connection':a['connectionId'],'start_ns':first['startNs'],'end_ns':last['endNs'],'launch_TASK_ROW':first['TASK_ROW'],'wait_TASK_ROW':last['TASK_ROW'],'CANN_API_ROW':a['API_ROW'],'MSTX_globalTid':m['globalTid'],'F21':[]})
 env_by_key=collections.defaultdict(list)
 for e in envelopes:env_by_key[(e['native_pid'],e['device_id'],e['native_model_id'])].append(e)
 for es in env_by_key.values():
  es.sort(key=lambda e:e['start_ns']);assert all(a['end_ns']<=b['start_ns'] for a,b in zip(es,es[1:])), 'Overlapping model envelopes need different index'
 starts={k:[e['start_ns'] for e in es] for k,es in env_by_key.items()}
 eager_marks=collections.defaultdict(list)
 for m in marks:
  if m['message_text'] in eager:eager_marks[m['globalTid']].append(m)
 for ms in eager_marks.values():
  ms.sort(key=lambda m:m['startNs']);assert all(a['endNs']<=b['startNs'] for a,b in zip(ms,ms[1:])), 'Nested eager source scopes'
 eager_starts={k:[m['startNs'] for m in ms] for k,ms in eager_marks.items()}
 joins=[];unbound=[]
 def add(r,h,kind,evidence):
  shape=s[r['inputShapes']].strip('"').split(';');outshape=s[r['outputShapes']].strip('"').split(';')
  expected_x=f"{h['M']},5120";expected_w1=f"{h['E_local']},5120,4608";expected_w2=f"{h['E_local']},2304,5120"
  check=shape[:3]==[expected_x,expected_w1,expected_w2] and outshape[:2]==[expected_x,str(h['E_local'])]
  if not check:issues.append({'kind':'NATIVE_SOURCE_SHAPE_MISMATCH','TASK_ROW':r['TASK_ROW'],'expected':[expected_x,expected_w1,expected_w2],'input':shape,'output':outshape});return
  joins.append({**h,'native_start_ns':r['startNs'],'native_end_ns':r['endNs'],'native_duration_ns':r['endNs']-r['startNs'],'native_TASK_ROW':r['TASK_ROW'],'native_globalTaskId':r['globalTaskId'],'native_model_id':r['modelId'],'native_stream_id':r['streamId'],'native_task_id':r['taskId'],'binding':kind,'evidence':evidence,'native_shape_matched':True})
 for r in native:
  if r['modelId']!=UINT:
   key=(r['globalPid'],r['deviceId'],r['modelId']);i=bisect.bisect_right(starts.get(key,[]),r['startNs'])-1
   if i>=0:
    e=env_by_key[key][i]
    if e['start_ns']<=r['startNs']<=r['endNs']<=e['end_ns']:e['F21'].append(r);continue
   unbound.append({'TASK_ROW':r['TASK_ROW'],'reason':'NO_UNIQUE_NATIVE_ENVELOPE'});continue
  matches=[]
  for a in by_conn.get(r['connectionId'],[]):
   ms=eager_marks.get(a['globalTid'],[]);i=bisect.bisect_right(eager_starts.get(a['globalTid'],[]),a['startNs'])-1
   if i>=0 and ms[i]['startNs']<=a['startNs']<=a['endNs']<=ms[i]['endNs']:matches.append((a,ms[i]))
  if len(matches)!=1:unbound.append({'TASK_ROW':r['TASK_ROW'],'reason':'NO_UNIQUE_EXACT_EAGER_API_SOURCE','matches':len(matches)});continue
  a,m=matches[0];source=eager[m['message_text']];h=hist.get((source['bank_slot'],source['history_slot']))
  if h is None:issues.append({'kind':'EAGER_HISTORY_ABSENT','source':source});continue
  add(r,h,'EXACT_CONNECTION_AND_SAME_THREAD_SCOPE',{'source_marker':m['message_text'],'source_event':source['event'],'CANN_API_ROW':a['API_ROW'],'native_connection':r['connectionId'],'source_globalTid':m['globalTid']})
 for e in envelopes:
  rs=sorted(e['F21'],key=lambda r:r['startNs']);bs=sorted([b for b in meta['banks'] if b['graph_token']==e['graph_token'] and b['capture_epoch']==e['capture_epoch']],key=lambda b:b['capture_site'])
  if len(rs)!=len(bs) or not rs:issues.append({'kind':'GRAPH_SITE_NATIVE_CARDINALITY_MISMATCH','epoch':e['history_epoch'],'native':len(rs),'source':len(bs)});continue
  streams={r['streamId'] for r in rs};serial=all(a['endNs']<=b['startNs'] for a,b in zip(rs,rs[1:]))
  if len(streams)!=1 or not serial:issues.append({'kind':'GRAPH_MC2_NOT_SINGLE_SERIAL_STREAM','epoch':e['history_epoch']});continue
  for r,b in zip(rs,bs):
   h=hist.get((b['slot'],e['history_epoch']))
   if h is None:issues.append({'kind':'GRAPH_HISTORY_ABSENT','bank':b['slot'],'epoch':e['history_epoch']});continue
   add(r,h,'UNIQUE_ORDERED_MAPPING_IN_SAME_CAPTURED_GRAPH',{'source_graph_token':e['graph_token'],'capture_epoch':e['capture_epoch'],'source_capture_site':b['capture_site'],'source_event':e['source_event'],'replay_marker':e['marker'],'native_model_id':e['native_model_id'],'native_launch_TASK_ROW':e['launch_TASK_ROW'],'native_wait_TASK_ROW':e['wait_TASK_ROW'],'native_replay_connection':e['native_connection']})
 ecount=collections.Counter(r['binding'] for r in joins);capt=[r for r in joins if r['binding']=='UNIQUE_ORDERED_MAPPING_IN_SAME_CAPTURED_GRAPH'];zeros=[r for r in capt if r['receiver_rows']==0]
 # All claim domains stay explicit;same graph order is not a CPU creator connection.
 summary={'status':'ACTUAL_NATIVE_MC2_HISTORY_ASSOCIATION_REVIEW','pid':pidi,'process_uuid':uid,'database':str(db),'database_sha256':hashlib.sha256(db.read_bytes()).hexdigest(),'native_MC2_executions':len(native),'history_rows':len(hist),'joins':len(joins),'binding_counts':dict(ecount),'native_replays':len(envelopes),'native_F21_per_replay':dict(collections.Counter(len(e['F21']) for e in envelopes)),'unbound_count':len(unbound),'unbound_samples':unbound[:8],'issues':issues[:16],'issues_count':len(issues),'graph_kernel_streams':sorted({r['native_stream_id'] for r in capt}),'captured_R_us':stat([r['native_duration_ns']/1000 for r in capt]),'zero_receiver_captured_R_us':stat([r['native_duration_ns']/1000 for r in zeros]),'captured_R_over1ms':len([r for r in capt if r['native_duration_ns']>=1000000]),'over1ms_source_layers':dict(collections.Counter(r['layer_name'] for r in capt if r['native_duration_ns']>=1000000)),
 'graph_order_domain':['Same process/strongly held Python graph token/capture epoch,official same-thread RuntimeAPI/native execute-wait identity.','All source fused_experts calls use pinned nonMegaMC2 branch,one DispatchFFNCombine launch each in this current implementation contract.','Captured source sites and nativeMC2 tasks have equal cardinality,one native stream,serial order,and identical M/E/K weight/output shapes.','Given the graph-preserves-current-stream-order contract,there is exactly one order-preserving bijection. This is not a direct per-kernel CPU connection and not an all-reader or independence proof.'],
 'not_proved':['Native global communicator/iteration identity','All baseline readers or future fusion legality','Production timing,physical MFU or speedup'],'overall_complete':False}
 with (OUT/(uid+'-MC2-JOINS.jsonl')).open('w') as f:
  for r in joins:f.write(json.dumps(r,separators=(',',':'))+'\n')
 (OUT/(uid+'-REPLAYS.json')).write_text(json.dumps([{k:v for k,v in e.items() if k!='F21'} for e in envelopes],indent=2)+'\n')
 cache.write_text(json.dumps(summary,indent=2)+'\n');c.close();return summary
if __name__=='__main__':
 start=time.time();dbs=sorted((P/'runtime').glob('*_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db'));assert len(dbs)==16
 state={'state':'RUNNING','pid':os.getpid(),'start_ticks':Path('/proc/'+str(os.getpid())+'/stat').read_text().rsplit(')',1)[1].split()[19],'started_unix':start,'CPU_only':True,'completed':0};(OUT/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
 try:
  rows=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for r in pool.map(one,dbs):rows.append(r);state.update(completed=len(rows));(OUT/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
  result={'status':'NATIVE_MC2_HISTORY_ASSOCIATION_RETURNED','profiles':rows,'native_MC2_executions':sum(r['native_MC2_executions'] for r in rows),'history_rows':sum(r['history_rows'] for r in rows),'joins':sum(r['joins'] for r in rows),'unbound_count':sum(r['unbound_count'] for r in rows),'issues_count':sum(r['issues_count'] for r in rows),'native_replays':sum(r['native_replays'] for r in rows),'historical_native_or_phase_promotions':0,'overall_complete':False};(OUT/'MC2-NATIVE-JOINT-REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');state.update(state='RETURNED',elapsed_seconds=time.time()-start,result=str(OUT/'MC2-NATIVE-JOINT-REVIEW.json'));(OUT/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
 except BaseException:
  state.update(state='ERROR',error=traceback.format_exc());(OUT/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n');raise
