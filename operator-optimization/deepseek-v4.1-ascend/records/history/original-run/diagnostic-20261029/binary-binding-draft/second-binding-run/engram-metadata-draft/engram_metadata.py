"""Opt-in metadata observer; install only in an isolated diagnostic process.

No tensor contents, raw addresses or request data are written. This module does
not import the model or torch, open a device, or patch anything on import.
"""
import atexit
import functools
import json
import os
import re
import secrets
import stat
import sys
import threading
import time
import weakref
import fcntl

MAX_BYTES = 1024 * 1024
RESERVE = 512
MAX_IDS = 512
MAX_DIMS = 8
_STATE = None
_STATE_MU = threading.Lock()
_INIT_DROPS = 0
_INSTALLED = set()
_INSTALL_MU = threading.Lock()
_FORK_REGISTERED = False

class _Observer:
    def __init__(self, path, directory):
        self.path=path;self.directory=directory;self.pid=os.getpid()
        self.process=secrets.token_hex(12);self.mu=threading.Lock()
        self.fd=-1;self.ids=[];self.next_id=0;self.call_seq=0
        self.stats={'gate_unknown':0,'metadata_error':0,'state_lock_drop':0,'log_lock_drop':0,'write_error':0,'identity_evicted':0,'identity_unavailable':0,'truncated_metadata':0,'calls_seen':0,'cap_reached':0}
        self.cap=False
    def gate(self):
        try:
            s=os.stat(os.path.join(self.directory,'PROFILE_ACTIVE'))
            if stat.S_ISREG(s.st_mode):return True
            self.stats['gate_unknown']+=1;return False
        except FileNotFoundError:
            try:
                if stat.S_ISDIR(os.stat(self.directory).st_mode):return False
            except OSError:pass
        except OSError:pass
        self.stats['gate_unknown']+=1;return False
    def identity(self, obj, kind):
        # Identity comparison only; never obtain object or storage addresses.
        for existing_kind,ref,token in self.ids:
            if existing_kind==kind and ref() is obj:return token
        try:ref=weakref.ref(obj)
        except TypeError:
            self.stats['identity_unavailable']+=1;return None
        if len(self.ids)>=MAX_IDS:
            dead=next((i for i,(_,r,_) in enumerate(self.ids) if r() is None),None)
            if dead is not None:self.ids.pop(dead)
            else:
                self.ids.pop(0)
                self.stats['identity_evicted']+=1
        self.next_id+=1;token=self.next_id;self.ids.append((kind,ref,token))
        return token
    def metadata(self, tensor):
        torch=sys.modules.get('torch')
        # Subclasses can override methods or descriptors. Skip them conservatively.
        if torch is None or type(tensor) is not getattr(torch,'Tensor',None):
            self.stats['metadata_error']+=1;return None
        shape=list(tensor.shape);stride=list(tensor.stride());offset=tensor.storage_offset()
        if len(shape)>MAX_DIMS or len(stride)>MAX_DIMS or len(shape)!=len(stride) or any(type(x) is not int or not (-2**63<=x<2**63) for x in shape+stride) or type(offset) is not int or not (-2**63<=offset<2**63):
            self.stats['truncated_metadata']+=1;return None
        dtype=str(tensor.dtype);device_obj=tensor.device;device=getattr(device_obj,'type',None);device_index=getattr(device_obj,'index',None)
        if device_index is not None and (type(device_index) is not int or not 0<=device_index<65536):
            self.stats['truncated_metadata']+=1;return None
        if not re.fullmatch(r'[A-Za-z0-9_.]{1,40}',dtype) or device not in ('cpu','npu','cuda','meta'):
            self.stats['truncated_metadata']+=1;return None
        value={'shape':shape,'stride':stride,'storage_offset':offset,'dtype':dtype,'device_type':device,'device_index':device_index,'is_contiguous':bool(tensor.is_contiguous()),'tensor_instance':self.identity(tensor,'tensor')}
        try:
            # A weakly observed Python storage wrapper is not a physical-storage
            # lifetime ID; wrappers can disappear and be recreated.
            storage=tensor.untyped_storage()
            value['storage_wrapper_instance']=self.identity(storage,'storage_wrapper')
        except Exception:
            self.stats['identity_unavailable']+=1;value['storage_wrapper_instance']=None
        return value
    def write_locked(self, event, fields):
        if self.cap:return
        try:
            if self.fd<0:
                fd=os.open(self.path,os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_CLOEXEC|getattr(os,'O_NOFOLLOW',0),0o600)
                s=os.fstat(fd)
                if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:os.close(fd);self.stats['write_error']+=1;return
                self.fd=fd
            try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except (BlockingIOError,OSError):self.stats['log_lock_drop']+=1;return
            try:
                st=os.fstat(self.fd)
                if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_size<0:self.stats['write_error']+=1;return
                # Cooperating writers stop once any process writes the shared cap marker.
                n=min(st.st_size,RESERVE);tail=os.pread(self.fd,n,st.st_size-n)
                if b'\"event\": \"trace_cap\"' in tail:self.cap=True;return
                record={'v':1,'event':event,'pid':self.pid,'process_instance':self.process,'tid':threading.get_native_id(),'monotonic_ns':time.monotonic_ns(),'state_incomplete':bool(_INIT_DROPS or any(v for k,v in self.stats.items() if k!='calls_seen')),**fields}
                payload=(json.dumps(record,separators=(',',':'),ensure_ascii=True)+'\n').encode()
                if len(payload)>4096:self.stats['write_error']+=1;return
                if st.st_size+len(payload)>MAX_BYTES-RESERVE:
                    marker=(json.dumps({'v':1,'event':'trace_cap','max_bytes':MAX_BYTES})+'\n').encode()
                    if st.st_size+len(marker)<=MAX_BYTES:
                        if os.write(self.fd,marker)!=len(marker):self.stats['write_error']+=1
                    self.stats['cap_reached']+=1;self.cap=True;return
                if os.write(self.fd,payload)!=len(payload):self.stats['write_error']+=1
            finally:fcntl.flock(self.fd,fcntl.LOCK_UN)
        except Exception:self.stats['write_error']+=1
    def begin(self, tensor, num_tokens):
        if not self.gate():return None
        if not self.mu.acquire(False):self.stats['state_lock_drop']+=1;return None
        try:
            self.stats['calls_seen']+=1;self.call_seq+=1;call=self.call_seq
            try:meta=self.metadata(tensor)
            except Exception:self.stats['metadata_error']+=1;meta=None
            n=num_tokens if type(num_tokens) is int and 0<=num_tokens<2**63 else None
            self.write_locked('embed_begin',{'call':call,'num_tokens':n,'tensor':meta,'identity_scope':'python_weak_object_only'})
            return call
        finally:self.mu.release()
    def end(self, call, ok):
        if call is None:return
        if not self.mu.acquire(False):self.stats['state_lock_drop']+=1;return
        try:self.write_locked('embed_end',{'call':call,'outcome':'returned' if ok else 'raised'})
        finally:self.mu.release()
    def finish(self):
        if not self.mu.acquire(False):return
        try:
            # No normal-exit record after cap/write failure is proof of completeness.
            self.write_locked('observer_stats',{'stats':{**self.stats,'observer_initialization_drops':_INIT_DROPS},'identity_scope':'python_weak_object_only'})
            if self.fd>=0:os.close(self.fd);self.fd=-1
        except Exception:pass
        finally:self.mu.release()

def _after_fork():
    global _STATE,_STATE_MU,_INIT_DROPS,_INSTALL_MU
    # Only the child copy is reset; it opens its own descriptor lazily.
    if _STATE is not None:
        try:
            if _STATE.fd>=0:os.close(_STATE.fd)
        except Exception:pass
    _STATE=None;_STATE_MU=threading.Lock();_INIT_DROPS=0;_INSTALL_MU=threading.Lock()

def _observer():
    global _STATE,_INIT_DROPS
    if _STATE is not None and _STATE.pid==os.getpid():return _STATE
    if not _STATE_MU.acquire(False):_INIT_DROPS+=1;return None
    try:
        if _STATE is not None and _STATE.pid==os.getpid():return _STATE
        path=os.environ.get('WIO_ENGRAM_META_PATH','');directory=os.environ.get('WIO_DIAG_DIR','')
        if not path.startswith('/') or not directory.startswith('/') or os.path.normpath(path)!=os.path.join(os.path.normpath(directory),'engram-metadata.jsonl'):return None
        _STATE=_Observer(path,directory);return _STATE
    finally:_STATE_MU.release()

def install(cls):
    """Install for a supplied class only when WIO_ENGRAM_META_ENABLE=1.

    Returns True only if a wrapper was installed. The caller must ensure the
    supplied class and method are those from the reviewed image.
    """
    global _FORK_REGISTERED
    if os.environ.get('WIO_ENGRAM_META_ENABLE')!='1':return False
    if not _INSTALL_MU.acquire(False):return False
    try:
        original=cls.__dict__.get('embed_gathered')
        if not callable(original) or getattr(original,'_wio_engram_metadata',False):return False
        @functools.wraps(original)
        def wrapped(self,gathered,num_tokens,*args,**kwargs):
            observer=None;call=None
            try:
                observer=_observer()
                if observer is not None:call=observer.begin(gathered,num_tokens)
            except Exception:pass
            try:result=original(self,gathered,num_tokens,*args,**kwargs)
            except BaseException:
                try:
                    if observer is not None:observer.end(call,False)
                except Exception:pass
                raise
            try:
                if observer is not None:observer.end(call,True)
            except Exception:pass
            return result
        wrapped._wio_engram_metadata=True
        if not _FORK_REGISTERED and hasattr(os,'register_at_fork'):
            os.register_at_fork(after_in_child=_after_fork);_FORK_REGISTERED=True
        if cls not in _INSTALLED:
            atexit.register(lambda: _STATE.finish() if _STATE is not None else None)
        setattr(cls,'embed_gathered',wrapped)
        _INSTALLED.add(cls)
        return True
    except Exception:return False
    finally:_INSTALL_MU.release()
