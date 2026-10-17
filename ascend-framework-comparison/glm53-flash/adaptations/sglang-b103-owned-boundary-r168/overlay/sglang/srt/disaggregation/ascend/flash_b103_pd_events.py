# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostic event writer for serialized Flash PD qualification."""
import json,os,pathlib,threading,time
_LOCK=threading.Lock()
_ROOT=pathlib.Path('/run-evidence/pd-events')
def tag():
    from sglang.srt.managers.flash_pd_request_context import transfer_tag
    return transfer_tag()

def append(kind,**fields):
    v={'kind':kind,'pid':os.getpid(),'monotonic_ns':time.monotonic_ns(),**fields}
    data=(json.dumps(v,sort_keys=True)+'\n').encode()
    if len(data)>256*1024:raise ValueError('Flash PD diagnostic event too large')
    with _LOCK:
        _ROOT.mkdir(exist_ok=True)
        p=_ROOT/(str(os.getpid())+'.jsonl')
        if p.exists() and p.stat().st_size>64*1024*1024:
            raise ValueError('Flash PD diagnostic evidence bound exceeded')
        fd=os.open(p,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
        try:
            written=os.write(fd,data)
            if written!=len(data):raise IOError('Incomplete diagnostic append')
            os.fsync(fd)
        finally:os.close(fd)
    return v
