#define _GNU_SOURCE
#include "host_packet_snapshot.h"
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <unistd.h>
int aclrtLaunchKernelWithHostArgs(void*,uint32_t,void*,void*,void*,size_t,wio_placeholder*,size_t);
unsigned fake_calls(void),fake_names(void),fake_last_queries(void),fake_stream_queries(void),fake_bad_inputs(void);
static void call(uintptr_t func,size_t size) {
    unsigned char packet[4097]={0};packet[0]=0x42;packet[1]=(unsigned char)syscall(SYS_gettid);wio_placeholder ph={8,24};
    errno=E2BIG;int rc=aclrtLaunchKernelWithHostArgs((void*)func,48,(void*)0x44,(void*)0x33,packet,size,&ph,1);
    assert(rc==(func==5?77:0));assert(errno==(func==5?EDOM:EAGAIN));
    if(size==152) { uint64_t x;memcpy(&x,packet+8,8);assert(x==0x120000002018ULL);assert(packet[0]==0x42&&ph.addrOffset==8&&ph.dataOffset==24); }
}
static void *many(void *p) { (void)p;for(int i=0;i<20;i++)call(1,152);return NULL; }
int main(int argc,char **argv) {
    assert(argc==2);
    if(!strcmp(argv[1],"many")) { pthread_t ts[8];for(int i=0;i<8;i++)assert(!pthread_create(&ts[i],NULL,many,NULL));for(int i=0;i<8;i++)pthread_join(ts[i],NULL); }
    else if(!strcmp(argv[1],"expired")) { call(1,152);struct timespec t={1,100000000};nanosleep(&t,NULL);call(1,152); }
    else if(!strcmp(argv[1],"failure"))call(5,152);
    else if(!strcmp(argv[1],"non_target"))call(2,152);
    else if(!strcmp(argv[1],"name_failure"))call(3,152);
    else if(!strcmp(argv[1],"oversize"))call(1,4097);
    else if(!strcmp(argv[1],"recursive"))call(6,152);
    else if(!strcmp(argv[1],"fork")) { pid_t child=fork();assert(child>=0);if(!child) { call(1,152);_exit(0); }int status=0;assert(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0);call(1,152); }
    else if(!strcmp(argv[1],"early_exit")) { call(1,152);_exit(0); }
    else call(1,152);
    assert(fake_bad_inputs()==0);
    printf("{\"status\":\"PASS\",\"calls\":%u,\"name_queries\":%u,\"last_task_queries\":%u,\"stream_queries\":%u,\"bad_inputs\":%u}\n",fake_calls(),fake_names(),fake_last_queries(),fake_stream_queries(),fake_bad_inputs());
    return 0;
}
