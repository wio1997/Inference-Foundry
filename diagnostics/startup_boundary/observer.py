"""Host-only ordinary-execute boundary packet; never a formal timing result."""
import json,os,time
from pathlib import Path

def enabled():
    root=os.getenv('EXTREME_STARTUP_BOUNDARY_DIR')
    return bool(root and (Path(root)/'arm').exists())

from threading import Lock
_core_lock=Lock()
_flush_lock=Lock()
_core_rows=[]
_core_seq=0
_core_time_namespace=os.readlink('/proc/self/ns/time')

def core_event(kind,**fields):
    if not enabled():return
    global _core_seq
    row=dict(kind=kind,t_ns=time.monotonic_ns(),pid=os.getpid(),time_namespace=_core_time_namespace,**fields)
    with _core_lock:
        if len(_core_rows)>=100000:raise RuntimeError('boundary buffer exceeded diagnostic bound')
        _core_seq+=1;row['observer_seq']=_core_seq
        _core_rows.append(row)

def core_flush():
    # Called only after output enqueue; no per-request file writes.
    # Serialize flushers, while producers can append to the next batch.
    with _flush_lock:
        begin=time.monotonic_ns()
        with _core_lock:
            rows=list(_core_rows);_core_rows.clear()
        if not rows:return
        root=Path(os.environ['EXTREME_STARTUP_BOUNDARY_DIR'])
        try:
            raw=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows).encode()
            fd=os.open(root/'core.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
            try:
                if os.write(fd,raw)!=len(raw):raise RuntimeError('short core boundary flush')
            finally:os.close(fd)
        except BaseException:
            # A failed run is invalid; retain rows in memory for diagnosis.
            with _core_lock:_core_rows[:0]=rows
            raise
        end=time.monotonic_ns()
        with (root/'core_flush.jsonl').open('a') as f:
            f.write(json.dumps(dict(begin_ns=begin,end_ns=end,count=len(rows),first_seq=rows[0]['observer_seq'],last_seq=rows[-1]['observer_seq']))+'\n')

def worker_event(owner,kind,**fields):
    rows=getattr(owner,'_ordinary_boundary_rows',None)
    if rows is None:owner._ordinary_boundary_rows=rows=[]
    if len(rows)>=10000:raise RuntimeError('worker boundary buffer exceeded diagnostic bound')
    rows.append(dict(kind=kind,t_ns=time.monotonic_ns(),**fields))

def flush(owner,rank,cohort):
    root=Path(os.environ['EXTREME_STARTUP_BOUNDARY_DIR'])
    rows=getattr(owner,'_ordinary_boundary_rows',[])
    begin=time.monotonic_ns()
    data=dict(rank=rank,cohort=cohort,pid=os.getpid(),time_namespace=os.readlink('/proc/self/ns/time'),events=rows,scope='Host timestamps only; no device readiness or removable-wall claim')
    path=root/f'rank{rank}_cohort{cohort}.json'
    with path.open('x') as f:json.dump(data,f,separators=(',',':'))
    end=time.monotonic_ns()
    with (root/f'rank{rank}_flush.jsonl').open('a') as f:
        f.write(json.dumps(dict(cohort=cohort,begin_ns=begin,end_ns=end))+'\n')
    owner._ordinary_boundary_rows=[]

def metadata_scalars(obj):
    rows=[];seen=set()
    def visit(value,depth):
        if depth>6 or id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,dict):
            for v in value.values():visit(v,depth+1)
        elif isinstance(value,(list,tuple)):
            for v in value:visit(v,depth+1)
        elif 'DSA' in type(value).__name__ or 'SFA' in type(value).__name__:
            row={'type':type(value).__name__}
            for name in ('num_decodes','num_prefills','num_decode_tokens','num_prefill_tokens','num_actual_tokens','num_input_tokens','num_reqs','max_query_len'):
                v=getattr(value,name,None)
                # Restrict to Python scalars: never materialize a Tensor.
                if type(v) in (int,bool):row[name]=v
            row['attn_state']=getattr(getattr(value,'attn_state',None),'name',None)
            rows.append(row)
    visit(obj,0)
    return rows
