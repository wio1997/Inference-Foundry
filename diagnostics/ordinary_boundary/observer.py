"""Host-only ordinary-execute boundary packet; never a formal timing result."""
import json,os,time
from pathlib import Path

def enabled():
    root=os.getenv('EXTREME_ORDINARY_BOUNDARY_DIR')
    return bool(root and (Path(root)/'arm').exists())

def core_event(kind,**fields):
    if not enabled():return
    row=dict(kind=kind,t_ns=time.monotonic_ns(),pid=os.getpid(),**fields)
    raw=(json.dumps(row,separators=(',',':'))+'\n').encode()
    path=Path(os.environ['EXTREME_ORDINARY_BOUNDARY_DIR'])/'core.jsonl'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    try:
        if os.write(fd,raw)!=len(raw):raise RuntimeError('short boundary record')
    finally:os.close(fd)

def worker_event(owner,kind,**fields):
    rows=getattr(owner,'_ordinary_boundary_rows',None)
    if rows is None:owner._ordinary_boundary_rows=rows=[]
    rows.append(dict(kind=kind,t_ns=time.monotonic_ns(),**fields))

def flush(owner,rank,cohort):
    root=Path(os.environ['EXTREME_ORDINARY_BOUNDARY_DIR'])
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
