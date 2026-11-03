"""Pure heartbeat rules and conservative Mac process-group identity helpers."""
import ctypes
import os
import signal
import sys
import time

if sys.flags.optimize:raise RuntimeError('optimized Python forbidden')

# Long enough for 40-minute model loading, 40-minute capture and recovery.
PHASE_LIMITS={'watchdog_ready':60,'starting':1000,'supervising':6000,'recovering':3200}
MAX_STALENESS=120


def check_heartbeat(h,run_id,now):
    if not isinstance(h,dict) or h.get('run_id')!=run_id:return 'INVALID'
    phase=h.get('phase');start=h.get('phase_start');updated=h.get('updated');deadline=h.get('deadline')
    if phase not in PHASE_LIMITS or any(type(x) not in (int,float) for x in (now,start,updated,deadline)):
        return 'INVALID'
    def birth_ok(x):return isinstance(x,list) and len(x)==3 and all(type(v) is int and v>=0 for v in x)
    if type(h.get('owner_pid')) is not int or h['owner_pid']<=0 or not birth_ok(h.get('owner_birth')):return 'INVALID'
    child=h.get('child_pid');child_birth=h.get('child_birth')
    if not ((child is None and child_birth is None) or (type(child) is int and child>0 and birth_ok(child_birth))):return 'INVALID'
    if not start<=updated<=now:return 'INVALID'
    if deadline!=start+PHASE_LIMITS[phase] or now>deadline:return 'DEADLINE'
    if now-updated>MAX_STALENESS:return 'STALE'
    return 'FRESH'


def validate_takeover_snapshot(h,last,run_id,owner,now):
    """Authenticate latest snapshot against last authenticated owner, even if stale."""
    state=check_heartbeat(h,run_id,now)
    if state not in ('FRESH','STALE','DEADLINE'):return False
    if h['deadline']!=h['phase_start']+PHASE_LIMITS[h['phase']]:return False
    if not isinstance(last,dict) or h['owner_pid']!=owner or h['owner_birth']!=last.get('owner_birth') or h['updated']<last.get('updated',float('inf')):return False
    try:
        if identity(owner)!=h['owner_birth'] or os.getsid(owner)!=owner or os.getpgid(owner)!=owner:return False
        child=h['child_pid']
        if child is not None:
            info=process_info(child)
            if info['birth']!=h['child_birth'] or info['ppid']!=owner or os.getsid(child)!=child or os.getpgid(child)!=child:return False
        return True
    except (OSError,RuntimeError):return False


class ProcBsdInfo(ctypes.Structure):
    _fields_=[('flags',ctypes.c_uint32),('status',ctypes.c_uint32),('xstatus',ctypes.c_uint32),
       ('pid',ctypes.c_uint32),('ppid',ctypes.c_uint32),('uid',ctypes.c_uint32),('gid',ctypes.c_uint32),
       ('ruid',ctypes.c_uint32),('rgid',ctypes.c_uint32),('svuid',ctypes.c_uint32),('svgid',ctypes.c_uint32),
       ('rfu',ctypes.c_uint32),('comm',ctypes.c_char*16),('name',ctypes.c_char*32),
       ('nfiles',ctypes.c_uint32),('pgid',ctypes.c_uint32),('pjobc',ctypes.c_uint32),('tdev',ctypes.c_uint32),
       ('tpgid',ctypes.c_uint32),('nice',ctypes.c_int32),('start_sec',ctypes.c_uint64),('start_usec',ctypes.c_uint64)]


def process_info(pid):
    """Read UID and microsecond process birth via macOS PROC_PIDTBSDINFO."""
    if sys.platform!='darwin' or type(pid) is not int or pid<=0:raise RuntimeError('unsupported process identity')
    lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
    fn=lib.proc_pidinfo;fn.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_int];fn.restype=ctypes.c_int
    info=ProcBsdInfo();n=fn(pid,3,0,ctypes.byref(info),ctypes.sizeof(info))
    if n!=ctypes.sizeof(info) or info.pid!=pid or info.uid!=os.getuid() or not info.start_sec:
        raise RuntimeError('cannot authenticate process identity')
    return {'birth':[int(info.start_sec),int(info.start_usec),int(info.uid)],'ppid':int(info.ppid),'status':int(info.status)}


def identity(pid):return process_info(pid)['birth']


def signal_owned_group(pid,birth,sig,expected_parent=None):
    """Signal only same-UID session/group leader with matching birth identity.

    A residual lookup-to-signal race remains; on uncertain identity do nothing.
    """
    info=process_info(pid)
    if info['birth']!=birth or (expected_parent is not None and info['ppid']!=expected_parent) or os.getpgid(pid)!=pid or os.getsid(pid)!=pid:return False
    os.killpg(pid,sig)
    return True


def wait_owner_stopped(pid,birth,timeout=5):
    """Authenticate each observation; Darwin sys/proc.h SSTOP == 4."""
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            info=process_info(pid)
            if info['birth']!=birth or os.getpgid(pid)!=pid or os.getsid(pid)!=pid:return False
            if info['status']==4:return True
        except (OSError,RuntimeError,KeyError):return False
        time.sleep(0.05)
    return False


def group_absent(pid):
    try:os.killpg(pid,0);return False
    except ProcessLookupError:return True
    except OSError:return False


def owner_absent(pid):
    # Both leader PID and group must be absent. Reuse or permission failure is UNKNOWN.
    try:os.kill(pid,0);return False
    except ProcessLookupError:return group_absent(pid)
    except OSError:return False


def snapshot_bound(h,last,run_id,owner,now):
    state=check_heartbeat(h,run_id,now)
    return (state in ('FRESH','STALE','DEADLINE') and
            h['deadline']==h['phase_start']+PHASE_LIMITS[h['phase']] and
            h['owner_pid']==owner and h['owner_birth']==last.get('owner_birth') and
            h['updated']>=last.get('updated',float('inf')))


def stop_owned_group(pid,birth,expected_parent=None):
    """Kill verified local session directly, including stopped groups; never resume it."""
    if pid is None and birth is None:return 'NO_CHILD_RECORDED'
    if pid is None or birth is None:return 'UNKNOWN'
    if group_absent(pid):return 'EXIT_CONFIRMED'
    try:
        if not signal_owned_group(pid,birth,signal.SIGKILL,expected_parent):return 'UNKNOWN'
        time.sleep(1)
        return 'EXIT_CONFIRMED' if group_absent(pid) else 'UNKNOWN'
    except (OSError,RuntimeError):return 'UNKNOWN'
