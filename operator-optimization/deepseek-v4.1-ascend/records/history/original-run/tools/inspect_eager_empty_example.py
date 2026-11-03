"""Read-only exact profiler chain for one attention-missing eager window."""
from pathlib import Path
import json,sqlite3
R=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces');T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
rank=next(R.glob('*ep2_*_ascend_pt'));db=next((rank/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_2.db'))
c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);c.execute('pragma query_only=ON');c.execute('pragma temp_store=MEMORY')
task=c.execute('select taskId,streamId,connectionId,globalTaskId from TASK where streamId=47 and taskId=44214').fetchall();assert len(task)==1
id,stream,conn,gid=task[0];assert (id,conn,gid)==(44214,101727,5038)
op=c.execute('select s.value,ss.value from COMPUTE_TASK_INFO ci left join STRING_IDS s on s.id=ci.opType left join STRING_IDS ss on ss.id=ci.inputShapes where ci.globalTaskId=?',(gid,)).fetchall();assert op==[('ZerosLike','"3,64,512"')]
links=c.execute('select id,connectionId from CONNECTION_IDS where connectionId=?',(conn,)).fetchall();assert len(links)==1;pyid=links[0][0]
a=c.execute('select startNs,endNs,globalTid,connectionId from PYTORCH_API where connectionId=?',(pyid,)).fetchall();assert len(a)==1
start,end,tid,_=a[0]
q='select a.startNs,a.endNs,s.value,i.value from PYTORCH_API a join STRING_IDS s on s.id=a.name left join STRING_IDS i on i.id=a.inputShapes where a.globalTid=? and cast(a.startNs as integer)<=? and cast(a.endNs as integer)>=? order by cast(a.startNs as integer)'
frames=c.execute(q,(tid,int(start),int(end))).fetchall();frame_names=[x[2] for x in frames];assert any('dsa_v41_forward' in x for x in frame_names) and 'aten::new_zeros' in frame_names and 'aclnnInplaceZero' in frame_names
x={'status':'EXACT_ZERO_FILL_CALL_CHAIN_NOT_QUERY_LENGTH_PROOF','rank':rank.name,'database':str(db),'task':{'task_id':id,'stream_id':stream,'connection_id':conn,'global_task_id':gid,'type':op[0][0],'input_shape':op[0][1]},'connection_mapping':links,'pytorch_api':{'start_ns':start,'end_ns':end,'thread_id':tid,'connection_id':pyid},'containing_frames':[{'start_ns':a,'end_ns':b,'name':n,'input_shapes':sh} for a,b,n,sh in frames], 'source':'sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py:350-355,432-437,704-719','limitations':['Output zero-fill shape is not original query-hidden or num_actual_tokens.','Full containment and connection IDs establish one call chain, not direct proof of the empty-query branch.','The example cannot establish behavior for all 720 windows.'],'no_device_work':True}
c.close();p=T/'evidence/eager-empty-query-example.json';assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'status':x['status'],'frames':frame_names,'task':x['task']}))
