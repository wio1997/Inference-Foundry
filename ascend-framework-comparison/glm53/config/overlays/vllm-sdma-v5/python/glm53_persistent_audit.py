"""Append project audit events outside rotating Docker/SDK logs."""
import json,os,pathlib,threading,time
_lock=threading.RLock()
def persist(kind,fields):
    row={'kind':kind,'pid':os.getpid(),'thread_id':threading.get_ident(),'audit_monotonic_ns':time.monotonic_ns(),**fields}
    data=(json.dumps(row,sort_keys=True)+'\n').encode()
    root=pathlib.Path('/run/glm53/audit');root.mkdir(exist_ok=True)
    with _lock:
        fd=os.open(root/f'events-{os.getpid()}.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try:
            if os.write(fd,data)!=len(data):raise RuntimeError('Incomplete GLM53 audit write')
        finally:os.close(fd)
