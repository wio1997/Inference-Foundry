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
MAX_WORKERS = 64
STATUS_MAX_BYTES = 4096
POLL_SECONDS = 0.5
_STATE = None
_STATE_MU = threading.Lock()
_INIT_DROPS = 0
_INIT_LOSS = False
_INSTALLED = set()
_INSTALL_MU = threading.Lock()
_FORK_REGISTERED = False

class _Observer:
    def __init__(self, path, directory):
        self.path=path;self.directory=directory;self.pid=os.getpid()
        self.run_id=os.environ.get('WIO_ENGRAM_RUN_ID','');self.installed={}
        self.loss=False;self.stop=threading.Event();self.worker=None;self.start_mu=threading.Lock();self.open_window=None;self.closed_window=None;self.inflight=0
        self.process=secrets.token_hex(12);self.mu=threading.Lock()
        self.fd=-1;self.ids=[];self.next_id=0;self.call_seq=0
        self.stats={'gate_unknown':0,'metadata_error':0,'state_lock_drop':0,'log_lock_drop':0,'write_error':0,'identity_evicted':0,'identity_unavailable':0,'truncated_metadata':0,'calls_seen':0,'cap_reached':0,'begin_written':0,'end_written':0,'status_drop':0}
        self.cap=False;self.finished=False
    def gate(self):
        path=os.path.join(self.directory,'PROFILE_ACTIVE')
        try:
            fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|getattr(os,'O_NOFOLLOW',0))
            try:
                st=os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_size>512:raise ValueError('invalid gate')
                v=json.loads(os.read(fd,513))
            finally:os.close(fd)
            if not isinstance(v,dict) or v.get('run_id')!=self.run_id or not re.fullmatch('[0-9a-f]{32}',v.get('window_id','')):raise ValueError('invalid gate')
            return v['window_id']
        except FileNotFoundError:
            try:
                if stat.S_ISDIR(os.stat(self.directory).st_mode):return ''
            except OSError:pass
        except (OSError,ValueError,TypeError):pass
        self.stats['gate_unknown']+=1;self.loss=True;return None
    def checkpoint(self, state, window=None):
        # Independent per-process status remains available when event log caps.
        if not self.mu.acquire(False):self.loss=True;self.stats['status_drop']+=1;return False
        try:
            facts={'version':1,'run_id':self.run_id,'process_instance':self.process,'pid':self.pid,'status':state,'window_id':window,
              'monotonic_ns':time.monotonic_ns(),'closed_window':self.closed_window,'closure_snapshot':'CONDITIONAL_NOT_PROOF','rank_environment_unverified':{k:(os.environ.get(k) if re.fullmatch('[0-9]{1,5}',os.environ.get(k,'')) else None) for k in ('RANK','LOCAL_RANK','WORLD_SIZE')},
              'installation':self.installed,'calls_seen':self.stats['calls_seen'],'inflight':self.inflight,'begin_written':self.stats['begin_written'],'end_written':self.stats['end_written'],
              'stats':dict(self.stats),'init_drops_lower_bound':_INIT_DROPS,'state_incomplete':bool(self.loss or _INIT_LOSS or _INIT_DROPS or any(v for k,v in self.stats.items() if k not in ('calls_seen','begin_written','end_written'))),'graph_coverage':'UNKNOWN'}
            raw=(json.dumps(facts,separators=(',',':'))+'\n').encode()
            if len(raw)>STATUS_MAX_BYTES:raise ValueError('status cap')
            lock=os.open(os.path.join(self.directory,'observer-status.lock'),os.O_RDWR|os.O_CREAT|os.O_CLOEXEC|getattr(os,'O_NOFOLLOW',0),0o600)
            if not _private_file(os.fstat(lock)):os.close(lock);raise ValueError('bad lock')
            try:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except (BlockingIOError,OSError):self.loss=True;self.stats['status_drop']+=1;return False
                try:
                    count=0
                    with os.scandir(self.directory) as it:
                        for x in it:
                            if re.fullmatch(r'observer-[0-9a-f]{24}\.json',x.name):count+=1
                            if count>MAX_WORKERS:break
                    name='observer-'+self.process+'.json';target=os.path.join(self.directory,name)
                    if count>=MAX_WORKERS and not os.path.exists(target):raise ValueError('worker status cap')
                    tmp=os.path.join(self.directory,'.observer-'+self.process+'.tmp')
                    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|getattr(os,'O_NOFOLLOW',0),0o600)
                    try:
                        if os.write(fd,raw)!=len(raw):raise OSError('short status write')
                        os.fsync(fd)
                    finally:os.close(fd)
                    os.replace(tmp,target)
                finally:fcntl.flock(lock,fcntl.LOCK_UN)
            finally:os.close(lock)
            return True
        except Exception:
            self.loss=True;self.stats['status_drop']+=1
            try:os.unlink(os.path.join(self.directory,'.observer-'+self.process+'.tmp'))
            except OSError:pass
            return False
        finally:self.mu.release()
    def watch(self):
        self.checkpoint('INSTALLED')
        while not self.stop.wait(POLL_SECONDS):
            current=self.gate()
            if current is None:continue
            if current and current!=self.open_window:
                if self.open_window is not None:self.loss=True;self.stats['gate_unknown']+=1
                if self.checkpoint('WINDOW_OPEN',current):self.open_window=current;self.closed_window=None
            elif not current and self.open_window and self.closed_window!=self.open_window:
                # Do not claim closure while any observed call is in flight.
                if self.inflight==0:
                    self.closed_window=self.open_window
                    if not self.checkpoint('WINDOW_CLOSED',self.open_window):self.closed_window=None
    def start(self,installed):
        if not self.start_mu.acquire(False):self.loss=True;return
        try:
            if self.worker is not None:return
            self.installed=installed
            self.worker=threading.Thread(target=self.watch,name='wio-engram-observer',daemon=True);self.worker.start()
        finally:self.start_mu.release()
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
        if self.cap:self.loss=True;return False
        try:
            if self.fd<0:
                fd=os.open(self.path,os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_CLOEXEC|getattr(os,'O_NOFOLLOW',0),0o600)
                s=os.fstat(fd)
                if not _private_file(s):os.close(fd);self.stats['write_error']+=1;self.loss=True;return False
                self.fd=fd
            try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except (BlockingIOError,OSError):self.stats['log_lock_drop']+=1;self.loss=True;return False
            try:
                st=os.fstat(self.fd)
                if not _private_file(st) or st.st_size<0:self.stats['write_error']+=1;self.loss=True;return False
                # Cooperating writers stop once any process writes the shared cap marker.
                n=min(st.st_size,RESERVE);tail=os.pread(self.fd,n,st.st_size-n)
                if b'\"event\": \"trace_cap\"' in tail:self.cap=True;self.loss=True;return False
                record={'v':1,'event':event,'pid':self.pid,'process_instance':self.process,'tid':threading.get_native_id(),'monotonic_ns':time.monotonic_ns(),'state_incomplete':bool(self.loss or _INIT_LOSS or _INIT_DROPS or any(v for k,v in self.stats.items() if k not in ('calls_seen','begin_written','end_written'))),**fields}
                payload=(json.dumps(record,separators=(',',':'),ensure_ascii=True)+'\n').encode()
                if len(payload)>4096:self.stats['write_error']+=1;self.loss=True;return False
                if st.st_size+len(payload)>MAX_BYTES-RESERVE:
                    marker=(json.dumps({'v':1,'event':'trace_cap','max_bytes':MAX_BYTES})+'\n').encode()
                    if st.st_size+len(marker)<=MAX_BYTES:
                        if os.write(self.fd,marker)!=len(marker):self.stats['write_error']+=1
                    self.stats['cap_reached']+=1;self.cap=True;self.loss=True;return False
                if os.write(self.fd,payload)!=len(payload):self.stats['write_error']+=1;self.loss=True;return False
                return True
            finally:fcntl.flock(self.fd,fcntl.LOCK_UN)
        except Exception:self.stats['write_error']+=1;self.loss=True;return False
    def begin(self, tensor, num_tokens):
        window=self.gate()
        if not window:return None
        if not self.mu.acquire(False):self.stats['state_lock_drop']+=1;self.loss=True;return None
        try:
            if self.gate()!=window:self.loss=True;self.stats['gate_unknown']+=1;return None
            self.stats['calls_seen']+=1;self.call_seq+=1;call=self.call_seq;self.inflight+=1
            try:meta=self.metadata(tensor)
            except Exception:self.stats['metadata_error']+=1;self.loss=True;meta=None
            n=num_tokens if type(num_tokens) is int and 0<=num_tokens<2**63 else None
            if self.write_locked('embed_begin',{'window_id':window,'call':call,'num_tokens':n,'tensor':meta,'identity_scope':'python_weak_object_only'}):self.stats['begin_written']+=1
            return (call,window)
        finally:self.mu.release()
    def end(self, call, ok):
        if call is None:return
        if not self.mu.acquire(False):self.stats['state_lock_drop']+=1;self.loss=True;return
        try:
            if self.write_locked('embed_end',{'window_id':call[1],'call':call[0],'outcome':'returned' if ok else 'raised'}):self.stats['end_written']+=1
            self.inflight-=1
        finally:self.mu.release()
    def finish(self):
        if self.finished:return
        self.finished=True;self.stop.set()
        if self.worker is not None and self.worker is not threading.current_thread():
            self.worker.join(timeout=2)
            if self.worker.is_alive():self.loss=True
        self.checkpoint('PROCESS_EXIT',self.open_window)
        if not self.mu.acquire(False):return
        try:
            # No normal-exit record after cap/write failure is proof of completeness.
            self.write_locked('observer_stats',{'stats':{**self.stats,'observer_initialization_drops':_INIT_DROPS},'identity_scope':'python_weak_object_only'})
            if self.fd>=0:os.close(self.fd);self.fd=-1
        except Exception:pass
        finally:self.mu.release()

def _after_fork():
    global _STATE,_STATE_MU,_INIT_DROPS,_INIT_LOSS,_INSTALL_MU
    # Only the child copy is reset; it opens its own descriptor lazily.
    if _STATE is not None:
        try:
            if _STATE.fd>=0:os.close(_STATE.fd)
        except Exception:pass
    _STATE=None;_STATE_MU=threading.Lock();_INIT_DROPS=0;_INIT_LOSS=False;_INSTALL_MU=threading.Lock()

def _private_file(st):
    return stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_uid==os.geteuid() and not (stat.S_IMODE(st.st_mode)&0o077)

def _private_directory(directory, path):
    # This rejects symlink components, including the destination itself.
    # The dedicated parent tree must also be controlled by the operator;
    # this check is not a defence against concurrent hostile renames.
    if os.path.realpath(directory)!=os.path.normpath(directory):return False
    st=os.stat(directory,follow_symlinks=False)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid!=os.geteuid() or stat.S_IMODE(st.st_mode)&0o077:return False
    try:st=os.stat(path,follow_symlinks=False)
    except FileNotFoundError:return True
    return _private_file(st)

def _observer():
    global _STATE,_INIT_DROPS,_INIT_LOSS
    if _STATE is not None and _STATE.pid==os.getpid():return _STATE
    if not _STATE_MU.acquire(False):_INIT_DROPS+=1;_INIT_LOSS=True;return None
    try:
        if _STATE is not None and _STATE.pid==os.getpid():return _STATE
        path=os.environ.get('WIO_ENGRAM_META_PATH','');directory=os.environ.get('WIO_DIAG_DIR','')
        if not re.fullmatch('[0-9a-f]{32}',os.environ.get('WIO_ENGRAM_RUN_ID','')) or not path.startswith('/') or not directory.startswith('/') or os.path.normpath(path)!=os.path.join(os.path.normpath(directory),'engram-metadata.jsonl'):return None
        try:
            if not _private_directory(directory,path):return None
        except OSError:return None
        _STATE=_Observer(path,directory);return _STATE
    finally:_STATE_MU.release()

def install(cls, verified):
    """Install for a supplied class only when WIO_ENGRAM_META_ENABLE=1.

    Returns True only if a wrapper was installed. The caller must ensure the
    supplied class and method are those from the reviewed image.
    """
    global _FORK_REGISTERED
    if os.environ.get('WIO_ENGRAM_META_ENABLE')!='1':return False
    if not _INSTALL_MU.acquire(False):return False
    original=None;wrapped=None;obs=None;worker_before=None;installed=False
    try:
        original=cls.__dict__.get('embed_gathered')
        if not callable(original) or getattr(original,'_wio_engram_metadata',False):return False
        @functools.wraps(original)
        def wrapped(self,gathered,num_tokens,*args,**kwargs):
            observer=None;call=None
            try:
                observer=_observer()
                if observer is not None:
                    if observer.worker is None:observer.start({'status':'inherited_wrapper_unverified_after_fork'})
                    call=observer.begin(gathered,num_tokens)
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
        # Obtain the observer before changing the class. Missing or failing
        # initialization must not leave a transparent-looking partial patch.
        obs=_observer()
        if obs is None:raise RuntimeError('observer initialization failed')
        worker_before=obs.worker
        if cls not in _INSTALLED:
            atexit.register(lambda: _STATE.finish() if _STATE is not None else None)
        setattr(cls,'embed_gathered',wrapped)
        if cls.__dict__.get('embed_gathered') is not wrapped or getattr(wrapped,'__wrapped__',None) is not original:
            raise RuntimeError('wrapper mismatch')
        obs.start(verified)
        if obs.worker is None or not obs.worker.is_alive():raise RuntimeError('observer start failed')
        _INSTALLED.add(cls);installed=True
        return True
    except Exception:
        # Do not overwrite a method replaced independently by another party.
        if wrapped is not None and cls.__dict__.get('embed_gathered') is wrapped:
            try:setattr(cls,'embed_gathered',original)
            except Exception:pass
        if obs is not None:
            obs.loss=True
            if worker_before is None:obs.stop.set()
        if not installed:_INSTALLED.discard(cls)
        return False
    finally:_INSTALL_MU.release()
