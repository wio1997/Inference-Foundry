#define _GNU_SOURCE
#include <acl/acl_rt.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/if_alg.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

/* Diagnostic-only interposer: log metadata and digests, never input bytes. */
#define MAX_BINARY (128u * 1024u * 1024u)
#define MAX_NAME 256
#define MAX_RECORDS 100000
#ifndef BINDING_MAX_LOG_BYTES
#define BINDING_MAX_LOG_BYTES (16u * 1024u * 1024u)
#endif
#define CAP_RESERVE 512u
#if BINDING_MAX_LOG_BYTES <= CAP_RESERVE + 2048
#error BINDING_MAX_LOG_BYTES too small
#endif
static _Thread_local int in_hook;
static _Thread_local unsigned long long call_seq;
static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t log_lock = PTHREAD_MUTEX_INITIALIZER;
static int log_fd = -1;
static _Atomic unsigned long long seq;
static _Atomic unsigned long long dropped;
static _Atomic int cap_seen;

static void fork_prepare(void) { pthread_mutex_lock(&log_lock); }
static void fork_parent(void) { pthread_mutex_unlock(&log_lock); }
static void fork_child(void) { atomic_store(&seq, 0); atomic_store(&dropped, 0); atomic_store(&cap_seen, 0); in_hook=0; pthread_mutex_unlock(&log_lock); }
static void init_log(void) {
    const char *p = getenv("BINDING_TRACE_PATH");
    if (!p || p[0] != '/') return;
    int fd = open(p, O_RDWR|O_CREAT|O_APPEND|O_CLOEXEC|O_NOFOLLOW, 0600);
    struct stat st;
    if (fd < 0) return;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_nlink != 1) { close(fd); return; }
    if (pthread_atfork(fork_prepare, fork_parent, fork_child)) { close(fd); return; }
    log_fd = fd;
}
static int logging(void) { pthread_once(&init_once, init_log); return log_fd >= 0 && !atomic_load(&cap_seen); }
/* Linux advisory lock is scoped to an inode and process; nonblocking on contention.
 * Assumes this fresh task-owned log is written only by cooperating recorder processes. */
static int file_lock(int fd,short type) {
    struct flock f={.l_type=type,.l_whence=SEEK_SET,.l_start=0,.l_len=1};
    return fcntl(fd,F_SETLK,&f);
}
static void emit(const char *event, const char *fields) {
    if (log_fd < 0 || atomic_load(&cap_seen)) return;
    struct timespec t; if(clock_gettime(CLOCK_MONOTONIC,&t)) return;
    char buf[2048];
    if(pthread_mutex_trylock(&log_lock)) {atomic_fetch_add(&dropped,1);return;}
    if(file_lock(log_fd,F_WRLCK)) {atomic_fetch_add(&dropped,1);pthread_mutex_unlock(&log_lock);return;}
    struct stat st;
    if(fstat(log_fd,&st) || !S_ISREG(st.st_mode) || st.st_nlink!=1 || st.st_size<0) {atomic_fetch_add(&dropped,1);goto done;}
    /* A previous process may already have written a cap marker. */
    char tail[CAP_RESERVE+1];size_t tailn=(size_t)st.st_size<CAP_RESERVE?(size_t)st.st_size:CAP_RESERVE;
    ssize_t got=pread(log_fd,tail,tailn,st.st_size-(off_t)tailn);
    if(got<0) { atomic_fetch_add(&dropped,1); goto done; }
    tail[got]=0;
    if(strstr(tail,"\"event\":\"trace_cap\"")) { atomic_store(&cap_seen,1);goto done; }
    if((uint64_t)st.st_size >= (uint64_t)BINDING_MAX_LOG_BYTES-CAP_RESERVE) {
        int caplen=snprintf(buf,sizeof buf,"{\"v\":1,\"pid\":%ld,\"event\":\"trace_cap\",\"max_bytes\":%u}\n",(long)getpid(),(unsigned)BINDING_MAX_LOG_BYTES);
        if(caplen>0 && (uint64_t)st.st_size+(unsigned)caplen<=BINDING_MAX_LOG_BYTES) (void)write(log_fd,buf,(size_t)caplen);
        atomic_store(&cap_seen,1);goto done;
    }
    unsigned long long n=atomic_fetch_add(&seq,1)+1;
    if(n>MAX_RECORDS)goto done;
    unsigned long long missed=atomic_exchange(&dropped,0);
    int len=snprintf(buf,sizeof buf,
        "{\"v\":1,\"pid\":%ld,\"tid\":%ld,\"seq\":%llu,\"monotonic_ns\":%lld,\"event\":\"%s\",%s,\"dropped_since_last\":%llu%s}\n",
        (long)getpid(),(long)syscall(SYS_gettid),n,(long long)t.tv_sec*1000000000LL+t.tv_nsec,
        event,fields,missed,n==MAX_RECORDS?",\"record_cap_reached\":true":"");
    if(len<=0 || (size_t)len>=sizeof buf || (uint64_t)st.st_size+(unsigned)len>(uint64_t)BINDING_MAX_LOG_BYTES-CAP_RESERVE) {
        /* Move directly to bounded cap marker. No partial record is written. */
        int caplen=snprintf(buf,sizeof buf,"{\"v\":1,\"pid\":%ld,\"event\":\"trace_cap\",\"max_bytes\":%u}\n",(long)getpid(),(unsigned)BINDING_MAX_LOG_BYTES);
        if(caplen>0 && (uint64_t)st.st_size+(unsigned)caplen<=BINDING_MAX_LOG_BYTES) (void)write(log_fd,buf,(size_t)caplen);
        atomic_store(&cap_seen,1);goto done;
    }
    if(write(log_fd,buf,(size_t)len)!=len) atomic_fetch_add(&dropped,missed+1);
done:
    (void)file_lock(log_fd,F_UNLCK);
    pthread_mutex_unlock(&log_lock);
}
static ssize_t safe_copy(void *dst,const void *src,size_t n) {
    struct iovec l={dst,n}, r={(void*)src,n};
    return process_vm_readv(getpid(),&l,1,&r,1,0);
}
static void hex(const unsigned char *in,size_t n,char *out) {
    static const char h[]="0123456789abcdef";
    for(size_t i=0;i<n;i++){out[2*i]=h[in[i]>>4];out[2*i+1]=h[in[i]&15];}
    out[2*n]=0;
}
static int hash_start(void) {
    int tfm=socket(AF_ALG,SOCK_SEQPACKET|SOCK_CLOEXEC,0); if(tfm<0)return -1;
    struct sockaddr_alg a={.salg_family=AF_ALG};
    strcpy((char*)a.salg_type,"hash"); strcpy((char*)a.salg_name,"sha256");
    if(bind(tfm,(struct sockaddr*)&a,sizeof a)) { close(tfm); return -1; }
    int op=accept4(tfm,NULL,NULL,SOCK_CLOEXEC); close(tfm); return op;
}
static int hash_finish(int fd,char out[65]) {
    unsigned char digest[32]; ssize_t n;
    if(send(fd,"",0,0)<0) {close(fd);return -1;}
    do {n=read(fd,digest,sizeof digest);} while(n<0 && errno==EINTR);
    close(fd); if(n!=32)return -1; hex(digest,32,out); return 0;
}
static int hash_memory(const void *p,size_t length,char out[65]) {
    if(!p || length==0 || length>MAX_BINARY || (uintptr_t)p>UINTPTR_MAX-length) return -1;
    int fd=hash_start(); if(fd<0)return -1;
    unsigned char chunk[8192]; size_t off=0;
    while(off<length) {
        size_t n=length-off<sizeof chunk?length-off:sizeof chunk;
        if(safe_copy(chunk,(const char*)p+off,n)!=(ssize_t)n || send(fd,chunk,n,MSG_MORE)!=(ssize_t)n) { close(fd); return -1; }
        off+=n;
    }
    return hash_finish(fd,out);
}
static int safe_string(const char *p,char *out,size_t cap,size_t *len) {
    if(!p || cap<2 || (uintptr_t)p>UINTPTR_MAX-cap)return -1;
    for(size_t i=0;i<cap;i++){
        if(safe_copy(out+i,(const void*)((uintptr_t)p+i),1)!=1)return -1;
        if(!out[i]) {*len=i;return 0;}
    }
    return -1;
}
/* This is a pre-call candidate digest. The runtime may open a different file. */
static int hash_file(const char *path,char out[65]) {
    char local[4096]; size_t len;
    if(safe_string(path,local,sizeof local,&len) || !len)return -1;
    int fd=open(local,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK); if(fd<0)return -1;
    struct stat a,b;
    if(fstat(fd,&a)||!S_ISREG(a.st_mode)||a.st_size<=0||(uint64_t)a.st_size>MAX_BINARY){close(fd);return -1;}
    int hash=hash_start(); if(hash<0){close(fd);return -1;}
    char chunk[8192]; off_t total=0;
    for(;;){ssize_t n=read(fd,chunk,sizeof chunk); if(n<0&&errno==EINTR)continue;
      if(n<0){close(fd);close(hash);return -1;} if(!n)break;
      total+=n; if(total>a.st_size||send(hash,chunk,(size_t)n,MSG_MORE)!=n){close(fd);close(hash);return -1;}}
    int invalid=fstat(fd,&b)||total!=a.st_size||a.st_dev!=b.st_dev||a.st_ino!=b.st_ino||a.st_size!=b.st_size||a.st_mtim.tv_sec!=b.st_mtim.tv_sec||a.st_mtim.tv_nsec!=b.st_mtim.tv_nsec;
    close(fd); if(invalid){close(hash);return -1;} return hash_finish(hash,out);
}
#define REAL(type,name) int enter_errno=errno; static _Atomic(void*) slot; type real=(type)atomic_load(&slot); \
    if(!real){real=(type)dlsym(RTLD_NEXT,#name); if(real) atomic_store(&slot,(void*)real);} \
    if(!real){errno=ENOSYS;return (aclError)-1;} errno=enter_errno;

aclError aclrtBinaryLoadFromData(const void* data,size_t len,const aclrtBinaryLoadOptions* opt,aclrtBinHandle* h) {
    typedef aclError(*F)(const void*,size_t,const aclrtBinaryLoadOptions*,aclrtBinHandle*);
    REAL(F,aclrtBinaryLoadFromData); if(in_hook)return real(data,len,opt,h); in_hook=1;
    int saved=errno; int enabled=logging(); char digest[65]=""; int ok=enabled&&!hash_memory(data,len,digest); errno=saved;
    in_hook=0; aclError rc=real(data,len,opt,h); int ret_errno=errno; in_hook=1; aclrtBinHandle v=NULL;
    if(enabled){if(rc==0 && h && safe_copy(&v,h,sizeof v)!=sizeof v) v=NULL;
      char f[512];snprintf(f,sizeof f,"\"rc\":%d,\"size\":%zu,\"handle\":\"%p\",\"pre_call_sha256\":\"%s\",\"hash_status\":\"%s\"",(int)rc,len,v,rc==0&&ok?digest:"",rc==0&&ok?"ok":"unknown");emit("load_data",f);}
    in_hook=0;errno=ret_errno;return rc;
}
aclError aclrtBinaryLoadFromFile(const char *path,aclrtBinaryLoadOptions *opt,aclrtBinHandle *h) {
    typedef aclError(*F)(const char*,aclrtBinaryLoadOptions*,aclrtBinHandle*);
    REAL(F,aclrtBinaryLoadFromFile); if(in_hook)return real(path,opt,h); in_hook=1;
    int saved=errno;int enabled=logging();char digest[65]="";int ok=enabled&&!hash_file(path,digest);errno=saved;
    in_hook=0;aclError rc=real(path,opt,h);int ret_errno=errno;in_hook=1;aclrtBinHandle v=NULL;
    if(enabled){if(rc==0&&h&&safe_copy(&v,h,sizeof v)!=sizeof v)v=NULL;
      char f[512];snprintf(f,sizeof f,"\"rc\":%d,\"handle\":\"%p\",\"pre_call_sha256\":\"%s\",\"hash_status\":\"%s\"",(int)rc,v,rc==0&&ok?digest:"",rc==0&&ok?"ok":"unknown");emit("load_file",f);}
    in_hook=0;errno=ret_errno;return rc;
}
aclError aclrtBinaryGetFunction(const aclrtBinHandle bin,const char *name,aclrtFuncHandle *h) {
    typedef aclError(*F)(aclrtBinHandle,const char*,aclrtFuncHandle*);
    REAL(F,aclrtBinaryGetFunction); if(in_hook)return real(bin,name,h);in_hook=1;
    int saved=errno,enabled=logging();char s[MAX_NAME],encoded[2*MAX_NAME+1]="";size_t n=0;int ok=enabled&&!safe_string(name,s,sizeof s,&n);if(ok)hex((unsigned char*)s,n,encoded);errno=saved;
    in_hook=0;aclError rc=real(bin,name,h);int ret_errno=errno;in_hook=1;aclrtFuncHandle v=NULL;
    if(enabled){if(rc==0&&h&&safe_copy(&v,h,sizeof v)!=sizeof v)v=NULL;
      char f[1024];snprintf(f,sizeof f,"\"rc\":%d,\"bin\":\"%p\",\"func\":\"%p\",\"name_hex\":\"%s\",\"name_status\":\"%s\"",(int)rc,bin,v,ok?encoded:"",ok?"ok":"unknown");emit("get_function",f);}
    in_hook=0;errno=ret_errno;return rc;
}
aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle bin,uint64_t entry,aclrtFuncHandle *h) {
    typedef aclError(*F)(aclrtBinHandle,uint64_t,aclrtFuncHandle*);
    REAL(F,aclrtBinaryGetFunctionByEntry); if(in_hook)return real(bin,entry,h);in_hook=1;
    int saved=errno,enabled=logging();errno=saved;in_hook=0;aclError rc=real(bin,entry,h);int ret_errno=errno;in_hook=1;aclrtFuncHandle v=NULL;
    if(enabled){if(rc==0&&h&&safe_copy(&v,h,sizeof v)!=sizeof v)v=NULL;
      char f[300];snprintf(f,sizeof f,"\"rc\":%d,\"bin\":\"%p\",\"func\":\"%p\",\"entry\":%llu",(int)rc,bin,v,(unsigned long long)entry);emit("get_function_entry",f);}
    in_hook=0;errno=ret_errno;return rc;
}
aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle func,uint32_t blocks,aclrtStream stream,aclrtLaunchKernelCfg *cfg,void *args,size_t size,aclrtPlaceHolderInfo *ph,size_t phn) {
    typedef aclError(*F)(aclrtFuncHandle,uint32_t,aclrtStream,aclrtLaunchKernelCfg*,void*,size_t,aclrtPlaceHolderInfo*,size_t);
    REAL(F,aclrtLaunchKernelWithHostArgs); if(in_hook)return real(func,blocks,stream,cfg,args,size,ph,phn);in_hook=1;
    int saved=errno,enabled=logging();unsigned long long call=++call_seq;
    if(enabled){char f[200];snprintf(f,sizeof f,"\"call\":%llu,\"func\":\"%p\",\"stream\":\"%p\"",call,func,stream);emit("launch_begin",f);}errno=saved;
    in_hook=0;aclError rc=real(func,blocks,stream,cfg,args,size,ph,phn);int ret_errno=errno;in_hook=1;
    if(enabled){char f[300];snprintf(f,sizeof f,"\"call\":%llu,\"rc\":%d,\"func\":\"%p\",\"stream\":\"%p\",\"blocks\":%u",call,(int)rc,func,stream,blocks);emit("launch_host_args",f);}
    in_hook=0;errno=ret_errno;return rc;
}
aclError aclrtBinaryUnLoad(aclrtBinHandle bin) {
    typedef aclError(*F)(aclrtBinHandle);REAL(F,aclrtBinaryUnLoad);if(in_hook)return real(bin);in_hook=1;
    int saved=errno,enabled=logging();errno=saved;in_hook=0;aclError rc=real(bin);int ret_errno=errno;in_hook=1;
    if(enabled){char f[160];snprintf(f,sizeof f,"\"rc\":%d,\"bin\":\"%p\"",(int)rc,bin);emit("unload",f);}
    in_hook=0;errno=ret_errno;return rc;
}

/* Passive wrappers: observe only calls already made by the program. */
aclError aclrtStreamGetId(aclrtStream stream,int32_t *id) {
    typedef aclError(*F)(aclrtStream,int32_t*);REAL(F,aclrtStreamGetId);if(in_hook)return real(stream,id);in_hook=1;
    int saved=errno,enabled=logging();errno=saved;in_hook=0;aclError rc=real(stream,id);int ret_errno=errno;in_hook=1;int32_t value=0;
    int ok=enabled&&rc==0&&id&&safe_copy(&value,id,sizeof value)==sizeof value;
    if(enabled){char f[192];snprintf(f,sizeof f,"\"rc\":%d,\"stream\":\"%p\",\"id_valid\":%s,\"stream_id\":%d",(int)rc,stream,ok?"true":"false",value);emit("stream_get_id",f);}
    in_hook=0;errno=ret_errno;return rc;
}
aclError aclrtGetThreadLastTaskId(uint32_t *id) {
    typedef aclError(*F)(uint32_t*);REAL(F,aclrtGetThreadLastTaskId);if(in_hook)return real(id);in_hook=1;
    int saved=errno,enabled=logging();errno=saved;in_hook=0;aclError rc=real(id);int ret_errno=errno;in_hook=1;uint32_t value=0;
    int ok=enabled&&rc==0&&id&&safe_copy(&value,id,sizeof value)==sizeof value;
    if(enabled){char f[128];snprintf(f,sizeof f,"\"rc\":%d,\"id_valid\":%s,\"task_id\":%u",(int)rc,ok?"true":"false",value);emit("thread_last_task_id",f);}
    in_hook=0;errno=ret_errno;return rc;
}
aclError aclmdlRICaptureGetInfo(aclrtStream stream,aclmdlRICaptureStatus *status,aclmdlRI *model) {
    typedef aclError(*F)(aclrtStream,aclmdlRICaptureStatus*,aclmdlRI*);REAL(F,aclmdlRICaptureGetInfo);if(in_hook)return real(stream,status,model);in_hook=1;
    int saved=errno,enabled=logging();errno=saved;in_hook=0;aclError rc=real(stream,status,model);int ret_errno=errno;in_hook=1;aclmdlRICaptureStatus st=0;aclmdlRI m=NULL;
    int ok=enabled&&rc==0&&status&&model&&safe_copy(&st,status,sizeof st)==sizeof st&&safe_copy(&m,model,sizeof m)==sizeof m;
    if(enabled){char f[256];snprintf(f,sizeof f,"\"rc\":%d,\"stream\":\"%p\",\"result_valid\":%s,\"capture_status\":%d,\"model_ri\":\"%p\"",(int)rc,stream,ok?"true":"false",(int)st,m);emit("capture_get_info",f);}
    in_hook=0;errno=ret_errno;return rc;
}
aclError aclmdlRIExecuteAsync(aclmdlRI model,aclrtStream stream) {
    typedef aclError(*F)(aclmdlRI,aclrtStream);REAL(F,aclmdlRIExecuteAsync);if(in_hook)return real(model,stream);in_hook=1;
    int saved=errno,enabled=logging();unsigned long long call=++call_seq;if(enabled){char f[180];snprintf(f,sizeof f,"\"call\":%llu,\"model_ri\":\"%p\",\"stream\":\"%p\"",call,model,stream);emit("graph_execute_begin",f);}errno=saved;
    in_hook=0;aclError rc=real(model,stream);int ret_errno=errno;in_hook=1;
    if(enabled){char f[192];snprintf(f,sizeof f,"\"call\":%llu,\"rc\":%d,\"model_ri\":\"%p\",\"stream\":\"%p\"",call,(int)rc,model,stream);emit("graph_execute_end",f);}
    in_hook=0;errno=ret_errno;return rc;
}
