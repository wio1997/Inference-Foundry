"""Read-only correlation of actual scheduler ranges, CANN replay, and device tasks."""
from pathlib import Path
import bisect,collections,datetime,json,os,socket,sqlite3,time
assert socket.gethostname()=='<ROLE_HOSTNAME>'
ROOT=Path('/PRIVATE_TASK_ROOT');CPU=ROOT/'native_profile_CPU_g838';OUT=CPU/'logical_graph_map_g860';OUT.mkdir(exist_ok=False);os.environ['TMPDIR']=str(OUT);os.sched_setaffinity(0,set(range(172,184)));started=time.monotonic()
db=next((ROOT/'vllm_pd_g727/offline_db_rank0_g826').rglob('ascend_pytorch_profiler_0.db'));con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON');con.execute('PRAGMA temp_store=MEMORY');con.set_progress_handler(lambda:int(time.monotonic()-started>600),100000)
strings=dict(con.execute('SELECT id,value FROM STRING_IDS'));annotations=[]
for begin,end,tid,name in con.execute("SELECT p.startNs,p.endNs,p.globalTid,s.value FROM PYTORCH_API p JOIN STRING_IDS s ON p.name=s.id WHERE s.value LIKE 'ds41.scheduler:%'"):
 annotations.append({'start_ns':int(begin),'end_ns':int(end),'globalTid':tid,'logical':json.loads(name.split(':',1)[1])})
annotations.sort(key=lambda x:x['start_ns']);starts=[x['start_ns'] for x in annotations];assert len(annotations)==913
graph_calls=[]
for begin,end,cid,tid in con.execute("SELECT a.startNs,a.endNs,a.connectionId,a.globalTid FROM CANN_API a JOIN STRING_IDS s ON a.name=s.id WHERE s.value='aclmdlRIExecuteAsync' ORDER BY a.startNs"):
 i=bisect.bisect_right(starts,begin)-1;ann=annotations[i] if i>=0 and annotations[i]['start_ns']<=begin<=end<=annotations[i]['end_ns'] and annotations[i]['globalTid']==tid else None
 graph_calls.append({'start_ns':begin,'end_ns':end,'connectionId':cid,'annotation_iteration':ann['logical']['iteration'] if ann else None,'logical_query_tokens':ann['logical']['scheduled_query_tokens'] if ann else None,'logical_requests':len(ann['logical']['requests']) if ann else None})
assert len(graph_calls)==1812
connections={x['connectionId']:x for x in graph_calls};direct=collections.defaultdict(collections.Counter);model_types=collections.Counter();task_count=0
for cid,model,typ in con.execute('SELECT connectionId,modelId,taskType FROM TASK'):
 task_count+=1;model_types[(model,strings.get(typ,str(typ)))]+=1
 if cid in connections:direct[cid][model]+=1
for call in graph_calls:call['direct_device_model_counts']={str(k):v for k,v in direct[call['connectionId']].items()}
by_iteration=collections.defaultdict(list)
for call in graph_calls:by_iteration[call['annotation_iteration']].append(call)
pair_counts=collections.Counter();query_pairs=collections.Counter()
for iteration,calls in by_iteration.items():
 pair=tuple(tuple(sorted(c['direct_device_model_counts'])) for c in calls);pair_counts[pair]+=1
 if calls:query_pairs[(calls[0]['logical_query_tokens'],pair)]+=1
ops=[]
for model,opid,count,duration in con.execute('SELECT t.modelId,c.opType,COUNT(*),SUM(t.endNs-t.startNs) FROM TASK t JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId GROUP BY t.modelId,c.opType'):
 ops.append({'model_id':model,'op_type':strings[opid],'count':count,'sum_duration_ns_not_walltime':duration})
shapes=[]
for model,shapeid,count in con.execute("SELECT t.modelId,c.inputShapes,COUNT(*) FROM TASK t JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId JOIN STRING_IDS s ON c.opType=s.id WHERE s.value LIKE '%GroupedMatmul%' OR s.value LIKE '%SparseFlash%' GROUP BY t.modelId,c.inputShapes"):
 shapes.append({'model_id':model,'input_shapes':strings[shapeid],'count':count})
report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'database':str(db),'CPU_annotations':len(annotations),'nonzero_logical_steps':sum(x['logical']['scheduled_query_tokens']>0 for x in annotations),'RI_execute_calls':len(graph_calls),'RI_calls_enclosed_by_same_thread_scheduler_range':sum(x['annotation_iteration'] is not None for x in graph_calls),'RI_calls_directly_correlated_with_device_model':sum(bool(x['direct_device_model_counts']) for x in graph_calls),'task_rows_scanned':task_count,'calls_per_nonzero_iteration':dict(collections.Counter(len(v) for k,v in by_iteration.items() if k is not None)),'ordered_model_pair_counts':[{'ordered_models':k,'scheduler_iterations':v} for k,v in pair_counts.items()],'query_to_ordered_model_pair_counts':[{'logical_query_tokens':k[0],'ordered_models':k[1],'scheduler_iterations':v} for k,v in query_pairs.items()],'device_model_task_types':[{'model_id':k[0],'type':k[1],'count':v} for k,v in model_types.items()],'compute_ops':ops,'selected_compute_input_shapes':shapes,'elapsed_s':time.monotonic()-started,'scope':'ActualD0fullcapture/read-onlySDKDB. CPUlogicalinput anddevicegraphreplayarecorrelatedbyCPUenclosure/samethread/CANNconnection, notcrosshosttime. Target/draftnamesandphysicalpaddingrequireadditionalpinnedsourceandshapeinterpretation; opdurationscontainoverlapandarenotwalltime/gain.'}
(OUT/'graph_calls.json').write_text(json.dumps(graph_calls,indent=2)+'\n');(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'result':str(OUT/'result.json'),'elapsed_s':report['elapsed_s'],'RI_calls':report['RI_execute_calls'],'RI_enclosed':report['RI_calls_enclosed_by_same_thread_scheduler_range'],'RI_device_correlated':report['RI_calls_directly_correlated_with_device_model']}),flush=True)
