"""Bounded supplementary evidence, independent of Torch/CANN.

Object observations and version counters are not allocator generations or GPU
completion. Native joins need an explicit same-origin API connection witness.
"""
import atexit,contextvars,json,os,re,threading,time,uuid
from contextlib import contextmanager
from collections import Counter

class EvidenceSink:
    def __init__(self, directory, run_id, max_records=65536, max_bytes=64*1024**2):
        if not os.path.isabs(directory) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',run_id):
            raise ValueError('Absolute evidence directory and bounded run ID required')
        if not 8<=max_records<=65536 or not 4096<=max_bytes<=64*1024**2:
            raise ValueError('Supplementary evidence budget outside hard limits')
        self.pid=os.getpid();self.process_uuid=uuid.uuid4().hex;self.run_id=run_id
        self.limit=max_records;self.byte_limit=max_bytes;self.bytes=0;self.written=0
        self.sequence=0;self.closed=False;self.failures=Counter();self.lock=threading.RLock()
        self.exhausted=False
        self.scope_stack=contextvars.ContextVar('wio_evidence_scopes',default=())
        self.path=os.path.join(directory,'supplement-'+self.process_uuid+'.jsonl')
        os.makedirs(directory,mode=0o700,exist_ok=True)
        self.fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_CLOEXEC|getattr(os,'O_NOFOLLOW',0),0o600)
        self.emit('header',schema='TARGETED_EXTRA_FIELDS_V1',
            identity_scope='run UUID + process UUID + event ordinal; numeric handles alone are not global identities',
            storage_generations_and_native_task_bridges='UNKNOWN_UNLESS_SEPARATELY_ATTESTED',
            observed_deallocation_is_not_GPU_read_retirement=True)
        atexit.register(self.close)

    def next_id(self):
        if os.getpid()!=self.pid:return None
        with self.lock:
            self.sequence+=1;return self.sequence

    def wants_records(self):
        if os.getpid()!=self.pid:return False
        with self.lock:
            if self.closed or self.exhausted or self.written>=self.limit:
                self.failures['supplementary_calls_suppressed_after_budget']+=1
                return False
            return True

    def _write(self, blob):
        at=0
        while at<len(blob):
            try:n=os.write(self.fd,blob[at:])
            except InterruptedError:continue
            if n<=0:raise OSError('Incomplete evidence write')
            at+=n
        self.bytes+=len(blob)

    def emit(self, kind, **data):
        # Do not enter an inherited mutex or file in a forked child.
        if os.getpid()!=self.pid:return False
        with self.lock:
            if self.closed:return False
            if self.written>=self.limit:
                self.exhausted=True;self.failures['record_limit']+=1;return False
            row=dict(kind=kind,run_id=self.run_id,process_uuid=self.process_uuid,
                pid=self.pid,tid=threading.get_native_id(),monotonic_ns=time.monotonic_ns(),
                realtime_ns=time.time_ns(),scopes=list(self.scope_stack.get()),**data)
            try:blob=(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
            except (TypeError,ValueError):self.failures['encoding_error']+=1;return False
            if len(blob)>self.byte_limit-1024-self.bytes:
                self.exhausted=True;self.failures['byte_limit']+=1;return False
            try:self._write(blob)
            except OSError:self.failures['IO_error']+=1;return False
            self.written+=1;return True

    @contextmanager
    def scope(self, name, **identity):
        sid=self.next_id();token=self.scope_stack.set(self.scope_stack.get()+(sid,))
        self.emit('scope_begin',scope_id=sid,name=name,identity=identity)
        status='returned'
        try:yield sid
        except BaseException:
            status='original_call_raised';raise
        finally:
            self.emit('scope_end',scope_id=sid,name=name,status=status,
                scope_end_is_not_GPU_completion=True)
            self.scope_stack.reset(token)

    def close(self):
        if os.getpid()!=self.pid:return
        with self.lock:
            if self.closed:return
            row=dict(kind='footer',run_id=self.run_id,process_uuid=self.process_uuid,pid=self.pid,
                written=self.written,bytes_before_footer=self.bytes,limits=dict(records=self.limit,bytes=self.byte_limit),
                failures=dict(self.failures),source_coverage='SELECTED_PYTHON_DISPATCH_AND_GRAPH_BOUNDARIES_ONLY',
                complete_all_native_readers=False,allocation_generation_attested=False,
                missing_footer_or_drop_requires_unknown_scope=True)
            blob=(json.dumps(row,separators=(',',':'))+'\n').encode()
            try:
                if self.bytes+len(blob)<=self.byte_limit:self._write(blob)
            except OSError:pass
            finally:os.close(self.fd);self.closed=True

def safe_host_value(value, tensor_predicate, tensor_extra, depth=0):
    """Only existing host literals and Tensor metadata; never repr/copy/read Tensor values."""
    if tensor_predicate(value):return {'tensor_extra':tensor_extra(value)}
    if value is None or type(value) in (bool,int):return value
    if type(value) is float:
        import math
        return value if math.isfinite(value) else {'unknown':'nonfinite scalar'}
    if type(value) is str:return value[:192]
    if depth<3 and type(value) in (tuple,list):
        if len(value)>64:return {'unknown':'host sequence over64','length':len(value)}
        return [safe_host_value(v,tensor_predicate,tensor_extra,depth+1) for v in value]
    if depth<3 and type(value) is dict:
        if len(value)>64:return {'unknown':'host mapping over64','length':len(value)}
        return {str(k)[:96]:safe_host_value(v,tensor_predicate,tensor_extra,depth+1)
            for k,v in value.items() if type(k) in (str,int)}
    return {'unknown_type':type(value).__module__+'.'+type(value).__qualname__}

def join_native_candidates(native, tasks):
    """No timestamp matching; fail closed on absent identities, reuse or ambiguity.

    api_witness is supplied only by an independently verified exact connection
    bridge. Merely adding equal strings is not evidence of such a bridge.
    """
    required=['run_uuid','process_incarnation','device_namespace','context_epoch','api_witness']
    missing=[k for k in required+['stream_id','task_id','kernel_name','num_blocks'] if native.get(k) is None or native.get(k)=='']
    if native.get('task_id_status')!='QUERIED_SUCCESS':missing.append('queried task ID')
    if native.get('stream_id_status')!='QUERIED_SUCCESS':missing.append('queried stream ID')
    if native.get('api_witness_status')!='VERIFIED_EXACT_CONNECTION':missing.append('verified API witness')
    if any(type(native.get(k)) is not int or native[k]<0 for k in ['stream_id','task_id']):missing.append('valid integer task/stream IDs')
    if missing:return dict(status='UNKNOWN',reason='Missing explicit bridge facts',missing=missing,candidates=[])
    keys=required+['stream_id','task_id']
    rows=[r for r in tasks if all(k in r and r[k]==native[k] for k in keys)]
    if len(rows)!=1:return dict(status='UNKNOWN',reason='No unique full-identity match',candidates=rows)
    row=rows[0]
    if row.get('kernel_name')!=native.get('kernel_name') or row.get('num_blocks')!=native.get('num_blocks'):
        return dict(status='UNKNOWN',reason='Matched IDs contradict kernel signature',candidates=rows)
    return dict(status='MATCHED_IN_ATTESTED_CONNECTION_DOMAIN',task=row,
        does_not_attest_parameter_roles_aliasing_readers_or_completion=True)
