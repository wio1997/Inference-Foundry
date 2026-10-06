"""Narrow uprobes with per-sample scheduled task-clock group counter.

No torch, signals, NPU APIs, service/source mutation, or global tracing enable.
Own event names; exact offsets only; group CPU counter is proven by selfcheck.
"""
import ctypes, fcntl, hashlib, json, mmap, os, struct, time
from pathlib import Path


class Ring:
    def __init__(self, event_id, pid, leader, atomic):
        self.fd=-1;self.mapping=None;self.tail=0
        self.fd=event_open(2,event_id,pid,leader,1046,11)
        page=os.sysconf('SC_PAGE_SIZE')
        try:
            self.mapping=mmap.mmap(self.fd,page*65,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ|mmap.PROT_WRITE)
            base=ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))
            self.head_ptr,self.tail_ptr=base+1024,base+1032
            self.offset,self.size=struct.unpack_from('<QQ',self.mapping,1040)
            assert self.offset==page and self.size==page*64
            self.load=atomic.perf_head_acquire;self.load.argtypes=[ctypes.c_void_p];self.load.restype=ctypes.c_uint64
            self.store=atomic.perf_tail_release;self.store.argtypes=[ctypes.c_void_p,ctypes.c_uint64];self.store.restype=None
        except BaseException:self.close();raise
    def read(self):
        head=self.load(self.head_ptr);assert head-self.tail<=self.size,'ring overrun'
        rows=[]
        while self.tail<head:
            def copy(n):
                pos=self.tail%self.size;first=min(n,self.size-pos)
                return self.mapping[self.offset+pos:self.offset+pos+first]+self.mapping[self.offset:self.offset+n-first]
            typ,misc,size=struct.unpack('<IHH',copy(8));assert size>=8 and self.tail+size<=head
            raw=copy(size)
            if typ==9:
                pid,tid,wall,nr,enabled,running=struct.unpack_from('<IIQQQQ',raw,8)
                values=struct.unpack_from('<'+'Q'*nr,raw,48)
                rawsize=struct.unpack_from('<I',raw,48+nr*8)[0]
                payload=raw[52+nr*8:52+nr*8+rawsize]
                assert len(payload)==rawsize and enabled==running,'counter multiplexing'
                rows.append(dict(pid=pid,tid=tid,wall_ns=wall,task_cpu_ns=values[0],group_values=list(values),raw=payload.hex(),enabled=enabled,running=running))
            elif typ==2:
                ident,lost=struct.unpack_from('<QQ',raw,8);rows.append(dict(lost=lost,id=ident))
            else:rows.append(dict(other_record_type=typ,bytes=size))
            self.tail+=size
        self.store(self.tail_ptr,self.tail);return rows
    def close(self):
        if self.mapping is not None:self.mapping.close();self.mapping=None
        if self.fd>=0:os.close(self.fd);self.fd=-1


def event_open(typ,config,pid,group=-1,sample_type=0,read_format=11):
    header=Path('/usr/include/asm-generic/unistd.h').read_text();nr=int(next(x for x in header.splitlines() if x.startswith('#define __NR_perf_event_open ')).split()[-1])
    attr=bytearray(128);flags=1|(1<<25)
    struct.pack_into('<IIQQQQQ',attr,0,typ,128,config,1 if sample_type else 0,sample_type,read_format,flags)
    struct.pack_into('<I',attr,48,1);struct.pack_into('<i',attr,92,1)
    buf=ctypes.create_string_buffer(bytes(attr));libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    fd=int(libc.syscall(nr,ctypes.byref(buf),pid,-1,group,8))
    if fd<0:
        err=ctypes.get_errno();raise OSError(err,os.strerror(err),dict(type=typ,config=config,pid=pid,group=group))
    return fd


def function_offsets(path,selected):
    b=Path(path).read_bytes();assert b[:6]==b'\x7fELF\x02\x01'
    h=struct.unpack_from('<16sHHIQQQIHHHHHH',b);loads=[struct.unpack_from('<IIQQQQQQ',b,h[5]+i*h[9]) for i in range(h[10])];sections=[struct.unpack_from('<IIQQQQIIQQ',b,h[6]+i*h[11]) for i in range(h[12])]
    out={}
    for sec in sections:
        if sec[1] not in (2,11):continue
        names=sections[sec[6]];names=b[names[4]:names[4]+names[5]]
        for at in range(sec[4],sec[4]+sec[5],sec[9]):
            n,info,_,shndx,va,size=struct.unpack_from('<IBBHQQ',b,at);name=names[n:names.find(b'\0',n)].decode(errors='replace')
            if name not in selected or info&15!=2 or not shndx or not size:continue
            seg=next(x for x in loads if x[0]==1 and x[3]<=va<va+size<=x[3]+x[5]);out[name]=dict(va=va,offset=seg[2]+va-seg[3],size=size)
    assert set(out)==set(selected),(selected,out)
    return out,hashlib.sha256(b).hexdigest()


class Observer:
    def __init__(self,probes,tids,atomic_path,prefix):
        self.trace=Path('/sys/kernel/tracing');self.group=prefix+'_'+str(os.getpid());self.probes=probes;self.registered=[];self.leaders={};self.rings={};self.events=[]
        self.atomic=ctypes.CDLL(str(atomic_path))
        try:
            current=(self.trace/'uprobe_events').read_text();assert self.group+'/' not in current
            for name,row in probes.items():
                command=('r' if row['return'] else 'p')+':'+self.group+'/'+name+' '+row['path']+':'+hex(row['offset'])+'\n'
                with (self.trace/'uprobe_events').open('a') as f:f.write(command)
                self.registered.append(name)
                ident=int((self.trace/'events'/self.group/name/'id').read_text())
                for tid in tids:
                    if tid not in self.leaders:self.leaders[tid]=event_open(1,1,tid)
                    self.rings[(tid,name)]=Ring(ident,tid,self.leaders[tid],self.atomic)
        except BaseException:self.close();raise
    def enable(self):
        for fd in self.leaders.values():fcntl.ioctl(fd,0x2403,1);fcntl.ioctl(fd,0x2400,1)
    def drain(self):
        for (tid,name),ring in self.rings.items():
            for row in ring.read():row.update(event=name,target_tid=tid);self.events.append(row)
    def disable(self):
        for fd in self.leaders.values():fcntl.ioctl(fd,0x2401,1)
        self.drain()
    def close(self):
        errors=[]
        for fd in self.leaders.values():
            try:fcntl.ioctl(fd,0x2401,1)
            except OSError as e:errors.append(repr(e))
        for r in self.rings.values():r.close()
        self.rings={}
        for fd in self.leaders.values():os.close(fd)
        self.leaders={}
        for name in reversed(self.registered):
            try:
                with (self.trace/'uprobe_events').open('a') as f:f.write('-:'+self.group+'/'+name+'\n')
            except OSError as e:errors.append(repr(e))
        self.registered=[]
        assert not errors,errors
