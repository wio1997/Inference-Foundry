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
static pthread_mutex_t state_lock = PTHREAD_MUTEX_INITIALIZER;
static void reset_state_child(void);
static int log_fd = -1;
static _Atomic unsigned long long seq;
static _Atomic unsigned long long dropped;
static _Atomic int cap_seen;

static void fork_prepare(void) { pthread_mutex_lock(&state_lock); pthread_mutex_lock(&log_lock); }
static void fork_parent(void) { pthread_mutex_unlock(&log_lock); pthread_mutex_unlock(&state_lock); }
static void fork_child(void) { atomic_store(&seq, 0); atomic_store(&dropped, 0); atomic_store(&cap_seen, 0); in_hook=0; reset_state_child(); pthread_mutex_unlock(&log_lock); pthread_mutex_unlock(&state_lock); }
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


/* Fixed process-local tables. Unknown calls are retained as bounded metadata,
 * not written en masse. An entry match is only a binary-level candidate. */
#ifndef BIN_SLOTS
#define BIN_SLOTS 4096
#endif
#ifndef FUNC_SLOTS
#define FUNC_SLOTS 8192
#endif
#ifndef GRAPH_SLOTS
#define GRAPH_SLOTS 256
#endif
struct binrow { void *handle; unsigned long long gen; char sha[65]; int target_bin; };
struct funcrow { void *func,*bin; unsigned long long gen; uint64_t entry; int kind; };
#define PROMOTION_BATCH 64
#ifndef PENDING_SLOTS
#define PENDING_SLOTS 512
#endif
#ifndef PENDING_BATCH
#define PENDING_BATCH 64
#endif
struct promoted { void *func; uint64_t entry; };
struct pending_launch { void *func,*bin,*stream; unsigned long long gen,call; uint64_t entry,begin_ns; long tid; int rc; };
static struct binrow bins[BIN_SLOTS];
static struct funcrow funcs[FUNC_SLOTS];
static void *graphs[GRAPH_SLOTS];
static struct pending_launch pending[PENDING_SLOTS];
static size_t bin_cursor,func_cursor,graph_cursor,pending_cursor;
static unsigned long long generation;
static _Atomic unsigned long long observations,bin_evict,func_evict,graph_evict,state_miss,missing_bin,unmatched_entry,unknown_name,unknown_hash,promotion_drops,pending_drops,pending_seen,mutation_overlaps;
static _Atomic unsigned mutation_inflight, launch_inflight;
static _Atomic unsigned long long launch_mutation_overlaps,gate_unknown;
static _Atomic int gate_unknown_reported;
static _Atomic int state_incomplete;
static _Thread_local unsigned target_scope;
static _Thread_local int current_target_record;
static _Thread_local unsigned long long current_target_call;
static void reset_state_child(void) {
    memset(bins,0,sizeof bins);memset(funcs,0,sizeof funcs);memset(graphs,0,sizeof graphs);memset(pending,0,sizeof pending);
    bin_cursor=func_cursor=graph_cursor=pending_cursor=0;generation=0;target_scope=0;current_target_call=0;current_target_record=0;
    atomic_store(&observations,0);atomic_store(&bin_evict,0);atomic_store(&func_evict,0);
    atomic_store(&state_miss,0);atomic_store(&graph_evict,0);atomic_store(&missing_bin,0);atomic_store(&unmatched_entry,0);atomic_store(&unknown_name,0);atomic_store(&unknown_hash,0);atomic_store(&promotion_drops,0);atomic_store(&pending_drops,0);atomic_store(&pending_seen,0);atomic_store(&mutation_overlaps,0);atomic_store(&mutation_inflight,0);atomic_store(&launch_inflight,0);atomic_store(&launch_mutation_overlaps,0);atomic_store(&gate_unknown,0);atomic_store(&gate_unknown_reported,0);atomic_store(&state_incomplete,0);
}
/* stat only: this gate is an observation, not a device or graph barrier. */
static int active_gate(void) {
    const char *d=getenv("WIO_DIAG_DIR");char path[4096];struct stat st;
    if(!d||d[0]!='/'||snprintf(path,sizeof path,"%s/PROFILE_ACTIVE",d)>=(int)sizeof path)goto unknown;
    if(!stat(path,&st)) {if(S_ISREG(st.st_mode))return 1;goto unknown;}
    if(errno==ENOENT && !stat(d,&st) && S_ISDIR(st.st_mode))return 0;
unknown:
    atomic_fetch_add(&gate_unknown,1);atomic_store(&state_incomplete,1);
    if(!atomic_exchange(&gate_unknown_reported,1))emit("gate_unknown","\"status\":\"unknown\"");
    return 0;
}
static void launch_overlap(void){atomic_fetch_add(&launch_mutation_overlaps,1);atomic_store(&state_incomplete,1);}
static void mutation_begin(void) {
    if(atomic_load(&launch_inflight))launch_overlap();
    if(atomic_fetch_add(&mutation_inflight,1)){atomic_fetch_add(&mutation_overlaps,1);atomic_store(&state_incomplete,1);}
}
static void mutation_end(void) { atomic_fetch_sub(&mutation_inflight,1); }
static int state_take(void) {
    if(pthread_mutex_trylock(&state_lock)){atomic_fetch_add(&state_miss,1);atomic_store(&state_incomplete,1);return 0;}return 1;
}
static struct binrow *find_bin(void *h) {
    for(size_t i=0;i<BIN_SLOTS;i++){if(bins[i].handle==h&&h)return &bins[i];}
    return NULL;
}
static struct funcrow *find_func(void *h) {
    for(size_t i=0;i<FUNC_SLOTS;i++){if(funcs[i].func==h&&h)return &funcs[i];}
    return NULL;
}
static int known_sha(const char *s) {
    static const char *known[]={"4cf1474cb1009b96fc743f5a396a9c07a6980146c41818b60cdbf0d3453a43a9",
      "d88acd30e3b56893952a9e7094dc4c2c45ceed08942383f9e9d4c9dd3e2260b5",
      "a0aec99d7145b29bb226231ff24d751d0798ff5ead7a2be84cc5db0632797979"};
    for(size_t i=0;i<sizeof known/sizeof known[0];i++){if(!strcmp(s,known[i]))return 1;}
    return 0;
}
static void invalidate_bin(void *bin) {
    for(size_t i=0;i<FUNC_SLOTS;i++)if(funcs[i].bin==bin)memset(&funcs[i],0,sizeof funcs[i]);
}
static int cache_load(void *h,const char *sha,unsigned long long *gen) {
    int target=0;if(!h)return 0;if(!state_take())return 0;
    struct binrow *slot=find_bin(h);
    for(size_t i=0;i<FUNC_SLOTS;i++)if(funcs[i].bin==h&&funcs[i].func){atomic_store(&state_incomplete,1);break;}
    invalidate_bin(h);
    if(!slot) {slot=&bins[bin_cursor++%BIN_SLOTS];if(slot->handle){invalidate_bin(slot->handle);atomic_fetch_add(&bin_evict,1);atomic_store(&state_incomplete,1);}}
    memset(slot,0,sizeof *slot);slot->handle=h;slot->gen=++generation;
    if(sha&&*sha){snprintf(slot->sha,sizeof slot->sha,"%s",sha);slot->target_bin=known_sha(sha);}
    *gen=slot->gen;target=slot->target_bin;pthread_mutex_unlock(&state_lock);return target;
}
static int cache_function(void *bin,void *func,uint64_t entry,int named,int *kind,unsigned long long *gen,char digest[65],struct promoted *promoted,size_t *promoted_count,struct pending_launch *earlier,size_t *earlier_count) {
    if(!func||!state_take())return 0;
    struct binrow *b=find_bin(bin);struct funcrow *f=find_func(func);
    if(!f){f=&funcs[func_cursor++%FUNC_SLOTS];if(f->func){atomic_fetch_add(&func_evict,1);atomic_store(&state_incomplete,1);}}
    memset(f,0,sizeof *f);f->func=func;f->bin=bin;f->entry=entry;
    if(b){f->gen=b->gen;*gen=b->gen;snprintf(digest,65,"%s",b->sha);}else {atomic_fetch_add(&missing_bin,1);atomic_store(&state_incomplete,1);}
    f->kind=named?1:(b&&b->target_bin?2:0);*kind=f->kind;
    if(named&&b){b->target_bin=1;for(size_t i=0;i<FUNC_SLOTS;i++)if(funcs[i].bin==bin&&funcs[i].gen==b->gen&&funcs[i].kind==0){
        funcs[i].kind=2;
        if(promoted&&promoted_count&&*promoted_count<PROMOTION_BATCH){promoted[*promoted_count].func=funcs[i].func;promoted[*promoted_count].entry=funcs[i].entry;(*promoted_count)++;}
        else {atomic_fetch_add(&promotion_drops,1);atomic_store(&state_incomplete,1);}
    }
      for(size_t i=0;i<PENDING_SLOTS;i++)if(pending[i].func&&pending[i].bin==bin&&pending[i].gen==b->gen){
        if(earlier&&earlier_count&&*earlier_count<PENDING_BATCH){earlier[*earlier_count]=pending[i];(*earlier_count)++;}
        else {atomic_fetch_add(&pending_drops,1);atomic_store(&state_incomplete,1);}
        memset(&pending[i],0,sizeof pending[i]);
      }
    }
    if(!named&&f->kind==0)atomic_fetch_add(&unmatched_entry,1);
    pthread_mutex_unlock(&state_lock);return 1;
}
static int selected_function(void *func,void **bin,unsigned long long *gen,uint64_t *entry,char sha[65]) {
    if(!state_take()){return 0;}
    struct funcrow *f=find_func(func);int kind=0;
    if(f){struct binrow *b=find_bin(f->bin);
        if(f->gen==0 || (b&&b->gen==f->gen)){kind=f->kind?f->kind:3;*bin=f->bin;*gen=f->gen;*entry=f->entry;if(b)snprintf(sha,65,"%s",b->sha);}}
    pthread_mutex_unlock(&state_lock);return kind;
}
static void cache_pending_launch(void *func,void *bin,unsigned long long gen,uint64_t entry,void *stream,unsigned long long call,uint64_t begin_ns,long tid,int rc) {
    if(!state_take())return;
    struct funcrow *f=find_func(func);struct binrow *b=find_bin(bin);
    if(!f||f->kind!=0||f->bin!=bin||f->gen!=gen||!gen||!b||b->gen!=gen){atomic_fetch_add(&pending_drops,1);atomic_store(&state_incomplete,1);pthread_mutex_unlock(&state_lock);return;}
    struct pending_launch *r=&pending[pending_cursor++%PENDING_SLOTS];
    if(r->func){atomic_fetch_add(&pending_drops,1);atomic_store(&state_incomplete,1);}
    *r=(struct pending_launch){.func=func,.bin=bin,.stream=stream,.gen=gen,.entry=entry,.call=call,.begin_ns=begin_ns,.tid=tid,.rc=rc};
    atomic_fetch_add(&pending_seen,1);pthread_mutex_unlock(&state_lock);
}
static int cache_unload(void *bin,unsigned long long *gen) {
    if(!state_take()){return 0;}
    struct binrow *b=find_bin(bin);int selected=0;
    if(b){*gen=b->gen;selected=b->target_bin;}
    for(size_t i=0;i<FUNC_SLOTS;i++)if(funcs[i].bin==bin&&funcs[i].kind)selected=1;
    invalidate_bin(bin);if(b)memset(b,0,sizeof *b);
    pthread_mutex_unlock(&state_lock);return selected;
}
static int graph_add(void *model) {
    if(!model||!state_take()){return 0;}
    for(size_t i=0;i<GRAPH_SLOTS;i++)if(graphs[i]==model){pthread_mutex_unlock(&state_lock);return 1;}
    size_t slot=graph_cursor++%GRAPH_SLOTS;
    if(graphs[slot]){atomic_fetch_add(&graph_evict,1);atomic_store(&state_incomplete,1);}
    graphs[slot]=model;
    pthread_mutex_unlock(&state_lock);return 1;
}
static int graph_selected(void *model) {
    if(!model||!state_take()){return 0;}
    int yes=0;for(size_t i=0;i<GRAPH_SLOTS;i++)if(graphs[i]==model){yes=1;break;}
    pthread_mutex_unlock(&state_lock);return yes;
}
static int graph_remove(void *model) {
    if(!model||!state_take()){return 0;}
    int yes=0;for(size_t i=0;i<GRAPH_SLOTS;i++)if(graphs[i]==model){graphs[i]=NULL;yes=1;}
    pthread_mutex_unlock(&state_lock);return yes;
}
static void statistics(const char *why) {
    char f[800];snprintf(f,sizeof f,"\"reason\":\"%s\",\"observations\":%llu,\"bin_evicted\":%llu,\"func_evicted\":%llu,\"graph_evicted\":%llu,\"state_lock_drops\":%llu,\"missing_bin\":%llu,\"unknown_byentry\":%llu,\"unknown_name\":%llu,\"unknown_hash\":%llu,\"promotion_drops\":%llu,\"pending_drops\":%llu,\"pending_seen\":%llu,\"mutation_overlaps\":%llu,\"launch_mutation_overlaps\":%llu,\"gate_unknown\":%llu,\"state_incomplete\":%s",
      why,atomic_load(&observations),atomic_load(&bin_evict),atomic_load(&func_evict),atomic_load(&graph_evict),atomic_load(&state_miss),atomic_load(&missing_bin),atomic_load(&unmatched_entry),atomic_load(&unknown_name),atomic_load(&unknown_hash),atomic_load(&promotion_drops),atomic_load(&pending_drops),atomic_load(&pending_seen),atomic_load(&mutation_overlaps),atomic_load(&launch_mutation_overlaps),atomic_load(&gate_unknown),atomic_load(&state_incomplete)?"true":"false");emit("selective_stats",f);
}
static void tick(void) { unsigned long long n=atomic_fetch_add(&observations,1)+1;if((n%8192)==0&&active_gate())statistics("periodic"); }
__attribute__((destructor)) static void final_stats(void) { if(log_fd>=0)statistics("process_exit"); }
aclError aclrtBinaryLoadFromData(const void *data,size_t len,const aclrtBinaryLoadOptions *opt,aclrtBinHandle *out) {
    typedef aclError(*F)(const void*,size_t,const aclrtBinaryLoadOptions*,aclrtBinHandle*);REAL(F,aclrtBinaryLoadFromData);if(in_hook)return real(data,len,opt,out);in_hook=1;mutation_begin();
    int saved=errno,enabled=logging();char sha[65]="";int ok=enabled&&!hash_memory(data,len,sha);errno=saved;
    in_hook=0;aclError rc=real(data,len,opt,out);int e=errno;in_hook=1;aclrtBinHandle h=NULL;unsigned long long gen=0;
    if(enabled){tick();if(!ok)atomic_fetch_add(&unknown_hash,1);if(rc==0&&out&&safe_copy(&h,out,sizeof h)==sizeof h){int target=cache_load(h,ok?sha:"",&gen);
      if(target&&active_gate()){char f[300];snprintf(f,sizeof f,"\"kind\":\"sha_candidate\",\"binding_status\":\"candidate_unverified\",\"rc\":%d,\"bin\":\"%p\",\"generation\":%llu,\"pre_call_sha256\":\"%s\",\"state_incomplete\":%s",(int)rc,h,gen,sha,atomic_load(&state_incomplete)?"true":"false");emit("target_load",f);}}}
    mutation_end();in_hook=0;errno=e;return rc;
}
aclError aclrtBinaryLoadFromFile(const char *path,aclrtBinaryLoadOptions *opt,aclrtBinHandle *out) {
    typedef aclError(*F)(const char*,aclrtBinaryLoadOptions*,aclrtBinHandle*);REAL(F,aclrtBinaryLoadFromFile);if(in_hook)return real(path,opt,out);in_hook=1;mutation_begin();
    int saved=errno,enabled=logging();char sha[65]="";int ok=enabled&&!hash_file(path,sha);errno=saved;
    in_hook=0;aclError rc=real(path,opt,out);int e=errno;in_hook=1;aclrtBinHandle h=NULL;unsigned long long gen=0;
    if(enabled){tick();if(!ok)atomic_fetch_add(&unknown_hash,1);if(rc==0&&out&&safe_copy(&h,out,sizeof h)==sizeof h){int target=cache_load(h,ok?sha:"",&gen);
      if(target&&active_gate()){char f[300];snprintf(f,sizeof f,"\"kind\":\"file_sha_candidate\",\"binding_status\":\"candidate_unverified\",\"rc\":%d,\"bin\":\"%p\",\"generation\":%llu,\"pre_call_sha256\":\"%s\",\"state_incomplete\":%s",(int)rc,h,gen,sha,atomic_load(&state_incomplete)?"true":"false");emit("target_load",f);}}}
    mutation_end();in_hook=0;errno=e;return rc;
}
aclError aclrtBinaryGetFunction(const aclrtBinHandle bin,const char *name,aclrtFuncHandle *out) {
    typedef aclError(*F)(aclrtBinHandle,const char*,aclrtFuncHandle*);REAL(F,aclrtBinaryGetFunction);if(in_hook)return real(bin,name,out);in_hook=1;mutation_begin();
    int saved=errno,enabled=logging();char nm[MAX_NAME],encoded[MAX_NAME*2+1]="";size_t n=0;int ok=enabled&&!safe_string(name,nm,sizeof nm,&n);
    int target=ok&&strcasestr(nm,"DispatchFFNCombine")!=NULL;if(target)hex((unsigned char*)nm,n,encoded);errno=saved;
    in_hook=0;aclError rc=real(bin,name,out);int e=errno;in_hook=1;aclrtFuncHandle h=NULL;
    if(enabled){tick();if(!ok)atomic_fetch_add(&unknown_name,1);if(!target&&rc==0&&out&&safe_copy(&h,out,sizeof h)==sizeof h&&h&&state_take()){struct funcrow *old=find_func(h);if(old)memset(old,0,sizeof *old);pthread_mutex_unlock(&state_lock);}
      if(target){int kind=0;unsigned long long gen=0;char sha[65]="";struct promoted promoted[PROMOTION_BATCH];size_t promoted_count=0;struct pending_launch earlier[PENDING_BATCH];size_t earlier_count=0;
      if(rc==0&&out&&safe_copy(&h,out,sizeof h)==sizeof h)cache_function(bin,h,0,1,&kind,&gen,sha,promoted,&promoted_count,earlier,&earlier_count);
      int window=active_gate();if(!window&&earlier_count){atomic_fetch_add(&pending_drops,earlier_count);atomic_store(&state_incomplete,1);}
      if(window){char f[1200];snprintf(f,sizeof f,"\"kind\":\"name_match\",\"binding_status\":\"candidate_unverified\",\"rc\":%d,\"bin\":\"%p\",\"func\":\"%p\",\"generation\":%llu,\"pre_call_sha256\":\"%s\",\"name_hex\":\"%s\",\"state_incomplete\":%s",(int)rc,bin,h,gen,sha,encoded,atomic_load(&state_incomplete)?"true":"false");emit("target_function",f);
      for(size_t i=0;i<promoted_count;i++){char q[300];snprintf(q,sizeof q,"\"kind\":\"bin_candidate_byentry\",\"binding_status\":\"candidate_unverified\",\"bin\":\"%p\",\"func\":\"%p\",\"entry\":%llu,\"generation\":%llu,\"state_incomplete\":%s",bin,promoted[i].func,(unsigned long long)promoted[i].entry,gen,atomic_load(&state_incomplete)?"true":"false");emit("target_byentry_reclassified",q);}
      for(size_t i=0;i<earlier_count;i++){struct pending_launch *x=&earlier[i];char q[600];snprintf(q,sizeof q,"\"kind\":\"bin_candidate_byentry\",\"binding_status\":\"candidate_unverified\",\"bin\":\"%p\",\"func\":\"%p\",\"entry\":%llu,\"generation\":%llu,\"stream\":\"%p\",\"original_tid\":%ld,\"original_call\":%llu,\"original_begin_ns\":%llu,\"original_window_observed\":true,\"rc\":%d,\"state_incomplete\":%s",x->bin,x->func,(unsigned long long)x->entry,x->gen,x->stream,x->tid,x->call,(unsigned long long)x->begin_ns,x->rc,atomic_load(&state_incomplete)?"true":"false");emit("target_preclassification_launch",q);}}}}
    mutation_end();in_hook=0;errno=e;return rc;
}
aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle bin,uint64_t entry,aclrtFuncHandle *out) {
    typedef aclError(*F)(aclrtBinHandle,uint64_t,aclrtFuncHandle*);REAL(F,aclrtBinaryGetFunctionByEntry);if(in_hook)return real(bin,entry,out);in_hook=1;mutation_begin();
    int saved=errno,enabled=logging();errno=saved;in_hook=0;aclError rc=real(bin,entry,out);int e=errno;in_hook=1;aclrtFuncHandle h=NULL;
    if(enabled){tick();if(rc==0&&out&&safe_copy(&h,out,sizeof h)==sizeof h){int kind=0;unsigned long long gen=0;char sha[65]="";
      if(cache_function(bin,h,entry,0,&kind,&gen,sha,NULL,NULL,NULL,NULL)&&kind==2&&active_gate()){char f[400];snprintf(f,sizeof f,"\"kind\":\"bin_candidate_byentry\",\"binding_status\":\"candidate_unverified\",\"rc\":%d,\"bin\":\"%p\",\"func\":\"%p\",\"entry\":%llu,\"generation\":%llu,\"pre_call_sha256\":\"%s\",\"state_incomplete\":%s",(int)rc,bin,h,(unsigned long long)entry,gen,sha,atomic_load(&state_incomplete)?"true":"false");emit("target_function",f);}}}
    mutation_end();in_hook=0;errno=e;return rc;
}
aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle func,uint32_t blocks,aclrtStream stream,aclrtLaunchKernelCfg *cfg,void *args,size_t size,aclrtPlaceHolderInfo *ph,size_t phn) {
    typedef aclError(*F)(aclrtFuncHandle,uint32_t,aclrtStream,aclrtLaunchKernelCfg*,void*,size_t,aclrtPlaceHolderInfo*,size_t);REAL(F,aclrtLaunchKernelWithHostArgs);if(in_hook)return real(func,blocks,stream,cfg,args,size,ph,phn);in_hook=1;
    int saved=errno,enabled=logging(),kind=0,window=0,old_record=current_target_record;void *bin=NULL;unsigned long long gen=0,call=++call_seq,old_call=current_target_call;uint64_t entry=0,begin_ns=0;long tid=0;char sha[65]="";
    if(enabled){atomic_fetch_add(&launch_inflight,1);if(atomic_load(&mutation_inflight))launch_overlap();tick();window=active_gate();kind=selected_function(func,&bin,&gen,&entry,sha);if(kind==3&&window){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t)==0)begin_ns=(uint64_t)t.tv_sec*1000000000ULL+(uint64_t)t.tv_nsec;tid=(long)syscall(SYS_gettid);}else if(kind==1||kind==2){if(window){char f[450];snprintf(f,sizeof f,"\"call\":%llu,\"binding_status\":\"candidate_unverified\",\"kind\":\"%s\",\"func\":\"%p\",\"bin\":\"%p\",\"generation\":%llu,\"entry\":%llu,\"pre_call_sha256\":\"%s\",\"stream\":\"%p\",\"state_incomplete\":%s",call,kind==1?"name_match":"bin_candidate_byentry",func,bin,gen,(unsigned long long)entry,sha,stream,atomic_load(&state_incomplete)?"true":"false");emit("target_launch_begin",f);}target_scope++;current_target_call=call;current_target_record=window;}}
    errno=saved;in_hook=0;aclError rc=real(func,blocks,stream,cfg,args,size,ph,phn);int e=errno;in_hook=1;
    if(enabled&&kind==3&&window){cache_pending_launch(func,bin,gen,entry,stream,call,begin_ns,tid,(int)rc);if(atomic_load(&pending_seen)==1)statistics("unclassified_launch_observed");}
    if(enabled&&(kind==1||kind==2)){target_scope--;current_target_call=old_call;current_target_record=old_record;if(window){char f[300];snprintf(f,sizeof f,"\"call\":%llu,\"binding_status\":\"candidate_unverified\",\"rc\":%d,\"func\":\"%p\",\"stream\":\"%p\",\"blocks\":%u",call,(int)rc,func,stream,blocks);emit("target_launch_end",f);}}
    if(enabled)atomic_fetch_sub(&launch_inflight,1);
    in_hook=0;errno=e;return rc;
}
aclError aclrtBinaryUnLoad(aclrtBinHandle bin) {
    typedef aclError(*F)(aclrtBinHandle);REAL(F,aclrtBinaryUnLoad);if(in_hook)return real(bin);in_hook=1;mutation_begin();
    int saved=errno,enabled=logging();errno=saved;in_hook=0;aclError rc=real(bin);int e=errno;in_hook=1;
    if(enabled){tick();unsigned long long gen=0;if(rc==0&&cache_unload(bin,&gen)&&active_gate()){char f[200];snprintf(f,sizeof f,"\"rc\":%d,\"bin\":\"%p\",\"generation\":%llu",(int)rc,bin,gen);emit("target_unload",f);}}
    mutation_end();in_hook=0;errno=e;return rc;
}
aclError aclrtStreamGetId(aclrtStream stream,int32_t *id) {
    typedef aclError(*F)(aclrtStream,int32_t*);REAL(F,aclrtStreamGetId);if(in_hook)return real(stream,id);in_hook=1;
    int saved=errno,enabled=logging(),capture=enabled&&target_scope&&current_target_record;errno=saved;in_hook=0;aclError rc=real(stream,id);int e=errno;in_hook=1;int32_t value=0;
    int ok=capture&&rc==0&&id&&safe_copy(&value,id,sizeof value)==sizeof value;
    if(capture){char f[240];snprintf(f,sizeof f,"\"call\":%llu,\"rc\":%d,\"stream\":\"%p\",\"id_valid\":%s,\"stream_id\":%d",current_target_call,(int)rc,stream,ok?"true":"false",value);emit("target_stream_id",f);}
    in_hook=0;errno=e;return rc;
}
aclError aclrtGetThreadLastTaskId(uint32_t *id) {
    typedef aclError(*F)(uint32_t*);REAL(F,aclrtGetThreadLastTaskId);if(in_hook)return real(id);in_hook=1;
    int saved=errno,enabled=logging(),capture=enabled&&target_scope&&current_target_record;errno=saved;in_hook=0;aclError rc=real(id);int e=errno;in_hook=1;uint32_t value=0;
    int ok=capture&&rc==0&&id&&safe_copy(&value,id,sizeof value)==sizeof value;
    if(capture){char f[200];snprintf(f,sizeof f,"\"call\":%llu,\"rc\":%d,\"id_valid\":%s,\"task_id\":%u",current_target_call,(int)rc,ok?"true":"false",value);emit("target_last_task_id",f);}
    in_hook=0;errno=e;return rc;
}
aclError aclmdlRICaptureGetInfo(aclrtStream stream,aclmdlRICaptureStatus *status,aclmdlRI *model) {
    typedef aclError(*F)(aclrtStream,aclmdlRICaptureStatus*,aclmdlRI*);REAL(F,aclmdlRICaptureGetInfo);if(in_hook)return real(stream,status,model);in_hook=1;
    int saved=errno,enabled=logging(),observed=enabled&&target_scope,capture=observed&&current_target_record;errno=saved;in_hook=0;aclError rc=real(stream,status,model);int e=errno;in_hook=1;aclmdlRICaptureStatus st=0;aclmdlRI m=NULL;
    int ok=observed&&rc==0&&status&&model&&safe_copy(&st,status,sizeof st)==sizeof st&&safe_copy(&m,model,sizeof m)==sizeof m;
    if(ok&&m)graph_add(m);
    if(capture){char f[300];snprintf(f,sizeof f,"\"call\":%llu,\"rc\":%d,\"stream\":\"%p\",\"result_valid\":%s,\"capture_status\":%d,\"model_ri\":\"%p\",\"relation\":\"same_thread_nested_candidate\"",current_target_call,(int)rc,stream,ok?"true":"false",(int)st,m);emit("target_capture_info",f);}
    in_hook=0;errno=e;return rc;
}
aclError aclmdlRIExecuteAsync(aclmdlRI model,aclrtStream stream) {
    typedef aclError(*F)(aclmdlRI,aclrtStream);REAL(F,aclmdlRIExecuteAsync);if(in_hook)return real(model,stream);in_hook=1;
    int saved=errno,enabled=logging(),selected=enabled&&active_gate()&&graph_selected(model);unsigned long long call=++call_seq;
    if(selected){char f[250];snprintf(f,sizeof f,"\"call\":%llu,\"model_ri\":\"%p\",\"stream\":\"%p\",\"relation\":\"candidate_model_only\",\"binding_status\":\"candidate_unverified\",\"state_incomplete\":%s",call,model,stream,atomic_load(&state_incomplete)?"true":"false");emit("target_graph_begin",f);}
    errno=saved;in_hook=0;aclError rc=real(model,stream);int e=errno;in_hook=1;
    if(selected){char f[220];snprintf(f,sizeof f,"\"call\":%llu,\"rc\":%d,\"model_ri\":\"%p\",\"stream\":\"%p\"",call,(int)rc,model,stream);emit("target_graph_end",f);}
    in_hook=0;errno=e;return rc;
}

aclError aclmdlRIDestroy(aclmdlRI model) {
    typedef aclError(*F)(aclmdlRI);REAL(F,aclmdlRIDestroy);if(in_hook)return real(model);in_hook=1;
    int saved=errno,enabled=logging();errno=saved;in_hook=0;aclError rc=real(model);int e=errno;in_hook=1;
    if(enabled&&rc==0&&graph_remove(model)&&active_gate()){char f[120];snprintf(f,sizeof f,"\"model_ri\":\"%p\",\"rc\":%d",model,(int)rc);emit("target_graph_destroy",f);}
    in_hook=0;errno=e;return rc;
}
