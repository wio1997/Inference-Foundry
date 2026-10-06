"""Read-only existing profiler database relations; CPU only, no torch import."""
import gzip
import hashlib
import json
import pathlib
import re
import sqlite3
import sys

root, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dest.mkdir(exist_ok=True, parents=True)
index = []
for db in sorted(root.glob('*_rank*_*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db')):
    rank = int(re.search(r'_rank(\d+)_',str(db)).group(1))
    with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True) as cx:
        cols = ['op_name','group_name','device_start_ns','device_end_ns','connection','rank_size','count','dtype','host_launch_start_ns','host_launch_end_ns','host_function','host_tid']
        comm = cx.execute('''select s.value,g.value,c.startNs,c.endNs,c.connectionId,c.rankSize,c.count,c.dataType,a.startNs,a.endNs,n.value,a.globalTid
          from COMMUNICATION_OP c join STRING_IDS s on s.id=c.opName join STRING_IDS g on g.id=c.groupName
          left join CANN_API a on a.connectionId=c.connectionId left join STRING_IDS n on n.id=a.name''').fetchall()
        tasks = cx.execute('''select t.startNs,t.endNs,t.streamId,t.taskId,t.connectionId,t.taskType,coalesce(s.value,ss.value),a.startNs,a.endNs,n.value,a.globalTid
          from TASK t left join COMPUTE_TASK_INFO k on t.globalTaskId=k.globalTaskId left join STRING_IDS s on s.id=k.name
          left join COMMUNICATION_SCHEDULE_TASK_INFO sk on t.globalTaskId=sk.globalTaskId left join STRING_IDS ss on ss.id=sk.name
          left join CANN_API a on a.connectionId=t.connectionId left join STRING_IDS n on n.id=a.name''').fetchall()
        py = cx.execute('select globalPid from TASK limit 1').fetchone()[0]
        tid = (py<<32)|py
        cond = '' if rank in (13,15) else " and (s.value like 'vllm::%' or s.value in ('aten::embedding','aten::_unique2','Event::synchronize'))"
        cpu = cx.execute('''select p.startNs,p.endNs,s.value,p.connectionId,p.type from PYTORCH_API p join STRING_IDS s on s.id=p.name where p.globalTid=?'''+cond,(tid,)).fetchall()
        # The parser's CPU link mapping is not assumed equivalent to CANN IDs.
        conn = cx.execute('select id,connectionId from CONNECTION_IDS').fetchall() if rank in (13,15) else []
    assert len(comm)>=1901 and tasks
    result = {'rank':rank,'db':str(db),'db_sha256':hashlib.sha256(db.read_bytes()).hexdigest(),
              'comm_columns':cols,'comm':comm,
              'task_columns':['start_ns','end_ns','stream','task','connection','task_type','name','host_start_ns','host_end_ns','host_function','host_tid'],'tasks':tasks,
              'cpu_columns':['start_ns','end_ns','name','connection','type'],'cpu':cpu,
              'connection_ids':conn,'clock':'All D ranks share host167 realtime domain. No P/D cross-host subtraction.'}
    with gzip.open(dest/f'db_rank{rank}.json.gz','wt') as f:json.dump(result,f)
    row={'rank':rank,'comm':len(comm),'tasks':len(tasks),'cpu':len(cpu),'db_sha256':result['db_sha256']}
    index.append(row);print(json.dumps(row),flush=True)
(dest/'index.json').write_text(json.dumps(index,indent=2)+'\n')
