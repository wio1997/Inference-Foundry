#define _GNU_SOURCE
#include "host_packet_snapshot.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef int (*launch_fn)(void*,uint32_t,void*,void*,void*,size_t,wio_placeholder*,size_t);
typedef int (*name_fn)(const void*,uint32_t,char*);
typedef int (*last_fn)(uint32_t*);
typedef int (*stream_fn)(void*,int32_t*);
int aclrtLaunchKernelWithHostArgs(void*,uint32_t,void*,void*,void*,size_t,wio_placeholder*,size_t);
static launch_fn real_launch;
static name_fn real_name;
static last_fn real_last;
static stream_fn real_stream;
static void *backend_ref;
static pthread_once_t once=PTHREAD_ONCE_INIT;
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static pid_t owner;
static int enabled,query_ids,fd=-1,io_failed,initialized;
static unsigned limit=56,reserved,written;
static size_t byte_limit=1048576,bytes_written;
static uint64_t duration_ns=60000000000ULL,armed_ns;
static char directory[1024],run_id[65],arm_file[1024];
static _Thread_local unsigned inside,reentered;
static atomic_ullong seen,non_target,name_failed,reentrant_skips,quota_skips,time_skips,bad_packet,byte_skips,fork_skips;

static uint64_t now_ns(clockid_t clock) {
    struct timespec t;if(clock_gettime(clock,&t))return 0;
    return (uint64_t)t.tv_sec*1000000000ULL+(uint64_t)t.tv_nsec;
}
static unsigned long long parse_limit(const char *key,unsigned long long value,unsigned long long max) {
    const char *s=getenv(key);if(!s||!*s)return value;
    char *end;errno=0;unsigned long long n=strtoull(s,&end,10);
    if(errno||*end||n>max)return value;
    return n;
}
static void *next_symbol(const char *name) {
    void *p=dlsym(RTLD_NEXT,name);
    if(p)return p;
    const char *path=getenv("WIO_HOST_TRACE_REAL_LIBRARY");
    if(path&&path[0]=='/') {
        if(!backend_ref)backend_ref=dlopen(path,RTLD_LAZY|RTLD_LOCAL|RTLD_NOLOAD);
        if(backend_ref)p=dlsym(backend_ref,name);
    }
    return p;
}
static void init(void) {
    real_launch=(launch_fn)next_symbol("aclrtLaunchKernelWithHostArgs");
    if(real_launch==aclrtLaunchKernelWithHostArgs)real_launch=NULL;
    const char *e=getenv("WIO_HOST_TRACE_ENABLE"),*d=getenv("WIO_HOST_TRACE_DIR"),*r=getenv("WIO_HOST_TRACE_RUN_ID");
    enabled=e&&strcmp(e,"1")==0&&d&&d[0]=='/'&&strlen(d)<sizeof(directory)&&r&&*r&&strlen(r)<sizeof(run_id);
    if(enabled) {
        for(const char *p=r;*p;p++)if(!((*p>='a'&&*p<='z')||(*p>='A'&&*p<='Z')||(*p>='0'&&*p<='9')||*p=='_'||*p=='-'))enabled=0;
    }
    if(enabled) {
        strcpy(directory,d);strcpy(run_id,r);
        const char *a=getenv("WIO_HOST_TRACE_ARM_FILE");
        if(a&&strlen(a)<sizeof(arm_file))strcpy(arm_file,a);
        limit=(unsigned)parse_limit("WIO_HOST_TRACE_MAX_RECORDS",56,256);
        byte_limit=(size_t)parse_limit("WIO_HOST_TRACE_MAX_BYTES",1048576,16777216);
        if(byte_limit<1024)enabled=0;
        duration_ns=parse_limit("WIO_HOST_TRACE_MAX_SECONDS",60,3600)*1000000000ULL;
        query_ids=getenv("WIO_HOST_TRACE_QUERY_IDS")&&strcmp(getenv("WIO_HOST_TRACE_QUERY_IDS"),"1")==0;
        real_name=(name_fn)next_symbol("aclrtGetFunctionName");
        if(query_ids) {
            real_last=(last_fn)next_symbol("aclrtGetThreadLastTaskId");
            real_stream=(stream_fn)next_symbol("aclrtStreamGetId");
        }
    }
    initialized=1;
}
static launch_fn forwarder(void) {
    if(getpid()==owner) { pthread_once(&once,init);return real_launch; }
    /* Fork children never use an inherited logger/once/mutex. No new library load. */
    launch_fn f=(launch_fn)dlsym(RTLD_NEXT,"aclrtLaunchKernelWithHostArgs");
    if(!f) {
        const char *path=getenv("WIO_HOST_TRACE_REAL_LIBRARY");
        void *h=path?dlopen(path,RTLD_LAZY|RTLD_LOCAL|RTLD_NOLOAD):NULL;
        if(h)f=(launch_fn)dlsym(h,"aclrtLaunchKernelWithHostArgs");
    }
    return f==aclrtLaunchKernelWithHostArgs?NULL:f;
}
static int open_output(void) {
    if(fd>=0)return 1;
    if(io_failed)return 0;
    char path[1200];int n=snprintf(path,sizeof(path),"%s/hostargs_%s_%ld.jsonl",directory,run_id,(long)owner);
    if(n<0||(size_t)n>=sizeof(path)) { io_failed=1;return 0; }
    fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd<0) { io_failed=1;return 0; }
    return 1;
}
static int write_bytes(const char *data,size_t n) {
    size_t done=0;
    while(done<n) {
        ssize_t count=write(fd,data+done,n-done);
        if(count<0&&errno==EINTR)continue;
        if(count<=0) { io_failed=1;return 0; }
        done+=(size_t)count;bytes_written+=(size_t)count;
    }
    return 1;
}
static int append(char *out,size_t cap,size_t *used,const char *fmt,...) {
    va_list ap;va_start(ap,fmt);int n=vsnprintf(out+*used,cap-*used,fmt,ap);va_end(ap);
    if(n<0||(size_t)n>=cap-*used)return 0;
    *used+=(size_t)n;return 1;
}
static int hex(char *out,size_t cap,size_t *used,const unsigned char *bytes,size_t n) {
    static const char digits[]="0123456789abcdef";
    if(n>(cap-*used-1)/2)return 0;
    for(size_t i=0;i<n;i++) { out[(*used)++]=digits[bytes[i]>>4];out[(*used)++]=digits[bytes[i]&15]; }
    out[*used]=0;return 1;
}
__attribute__((constructor))static void load_owner(void) { owner=getpid(); }
/* Resolver/configuration handshake only:does not invoke a runtime getter. */
unsigned wio_host_trace_probe(void) {
    launch_fn real=forwarder();
    return (real?1u:0u)|(enabled?2u:0u)|(real_name?4u:0u);
}

int aclrtLaunchKernelWithHostArgs(void *func,uint32_t blocks,void *stream,void *cfg,void *args,size_t size,wio_placeholder *ph,size_t phn) {
    int entry_errno=errno;launch_fn real=forwarder();
    if(!real) { const char msg[]="wio host observer: real launch symbol unavailable;owned diagnostic must stop\n";ssize_t diagnostic_written=write(2,msg,sizeof(msg)-1);(void)diagnostic_written;_exit(125); }
    if(getpid()!=owner) { atomic_fetch_add(&fork_skips,1);errno=entry_errno;return real(func,blocks,stream,cfg,args,size,ph,phn); }
    atomic_fetch_add(&seen,1);
    if(inside) { reentered++;atomic_fetch_add(&reentrant_skips,1);errno=entry_errno;return real(func,blocks,stream,cfg,args,size,ph,phn); }
    if(!enabled) { errno=entry_errno;return real(func,blocks,stream,cfg,args,size,ph,phn); }
    if(arm_file[0]&&access(arm_file,R_OK)) { errno=entry_errno;return real(func,blocks,stream,cfg,args,size,ph,phn); }
    uint64_t mono=now_ns(CLOCK_MONOTONIC);
    pthread_mutex_lock(&lock);
    if(!armed_ns)armed_ns=mono;
    int expired=!mono||mono-armed_ns>=duration_ns;
    pthread_mutex_unlock(&lock);
    if(expired) { atomic_fetch_add(&time_skips,1);errno=entry_errno;return real(func,blocks,stream,cfg,args,size,ph,phn); }
    inside=1;
    char name[256]={0};int name_rc=real_name?real_name(func,sizeof(name),name):-1;
    int take=0;unsigned reservation=0;wio_host_snapshot snapshot;
    uint64_t begin=now_ns(CLOCK_REALTIME);
    unsigned reentry_before=reentered;
    if(name_rc!=0||!memchr(name,0,sizeof(name)-1))atomic_fetch_add(&name_failed,1);
    else if(!strstr(name,"MemSet"))atomic_fetch_add(&non_target,1);
    else {
        pthread_mutex_lock(&lock);
        if(reserved>=limit)atomic_fetch_add(&quota_skips,1);
        else if(open_output()) { reservation=++reserved;take=1; }
        pthread_mutex_unlock(&lock);
        if(take&&wio_snapshot_host_args(1,WIO_FROM_LIVE_HOST_ARGS_API,args,size,ph,phn,&snapshot)!=WIO_COPY_OK) {
            take=0;atomic_fetch_add(&bad_packet,1);
        }
    }
    errno=entry_errno;int rc=real(func,blocks,stream,cfg,args,size,ph,phn);int result_errno=errno;
    uint64_t end=now_ns(CLOCK_REALTIME);uint32_t task_id=0;int32_t stream_id=-1;int task_rc=-1,stream_rc=-1;
    if(take&&query_ids&&rc==0) {
        if(real_last)task_rc=real_last(&task_id);
        if(real_stream)stream_rc=real_stream(stream,&stream_id);
    }
    if(take) {
        char record[24576];size_t used=0;
        int ok=append(record,sizeof(record),&used,"{\"type\":\"host_args\",\"run_id\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"reservation\":%u,\"function_handle\":%llu,\"stream_handle\":%llu,\"num_blocks\":%u,\"api_begin_realtime_ns\":%llu,\"api_end_realtime_ns\":%llu,\"launch_rc\":%d,\"task_id_rc\":%d,\"task_id\":%u,\"stream_id_rc\":%d,\"stream_id\":%d,\"reentrant_skips_during_call\":%u,\"capture_scope\":\"NOT_QUERIED\",\"kernel_name_hex\":\"",run_id,(long)owner,(long)syscall(SYS_gettid),reservation,(unsigned long long)(uintptr_t)func,(unsigned long long)(uintptr_t)stream,blocks,(unsigned long long)begin,(unsigned long long)end,rc,task_rc,task_id,stream_rc,stream_id,reentered-reentry_before);
        ok=ok&&hex(record,sizeof(record),&used,(unsigned char*)name,strlen(name));
        ok=ok&&append(record,sizeof(record),&used,"\",\"args_size\":%zu,\"args_hex\":\"",snapshot.args_size);
        ok=ok&&hex(record,sizeof(record),&used,snapshot.args,snapshot.args_size);
        ok=ok&&append(record,sizeof(record),&used,"\",\"placeholders\":[");
        for(size_t i=0;i<snapshot.placeholder_count&&ok;i++)ok=append(record,sizeof(record),&used,"%s[%u,%u]",i?",":"",snapshot.placeholders[i].addrOffset,snapshot.placeholders[i].dataOffset);
        ok=ok&&append(record,sizeof(record),&used,"]}\n");
        pthread_mutex_lock(&lock);
        if(ok&&!io_failed&&used<=byte_limit-512&&bytes_written<=byte_limit-512-used) {
            if(write_bytes(record,used))written++;
        } else atomic_fetch_add(&byte_skips,1);
        pthread_mutex_unlock(&lock);
    }
    inside=0;errno=result_errno;return rc;
}

__attribute__((destructor))static void finish(void) {
    if(getpid()!=owner||!initialized||!enabled)return;
    pthread_mutex_lock(&lock);
    if(open_output()&&!io_failed) {
        char summary[512];
        int n=snprintf(summary,sizeof(summary),"{\"type\":\"summary\",\"pid\":%ld,\"seen\":%llu,\"reserved\":%u,\"written\":%u,\"non_target\":%llu,\"name_failed\":%llu,\"reentrant_skips\":%llu,\"quota_skips\":%llu,\"time_skips\":%llu,\"bad_packet\":%llu,\"byte_skips\":%llu,\"bytes_before_footer\":%zu,\"byte_limit\":%zu,\"armed\":%s}\n",(long)owner,(unsigned long long)atomic_load(&seen),reserved,written,(unsigned long long)atomic_load(&non_target),(unsigned long long)atomic_load(&name_failed),(unsigned long long)atomic_load(&reentrant_skips),(unsigned long long)atomic_load(&quota_skips),(unsigned long long)atomic_load(&time_skips),(unsigned long long)atomic_load(&bad_packet),(unsigned long long)atomic_load(&byte_skips),bytes_written,byte_limit,armed_ns?"true":"false");
        if(n>0&&(size_t)n<sizeof(summary)&&bytes_written+(size_t)n<=byte_limit)write_bytes(summary,(size_t)n);
    }
    if(fd>=0)close(fd);
    pthread_mutex_unlock(&lock);
    /* No dlclose: metadata provider may already be in process teardown. */
}
