#define _GNU_SOURCE
#include "host_packet_snapshot.h"
#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

static atomic_uint calls,names,last_queries,stream_queries,bad_inputs;
static _Thread_local uint32_t last_task;
static _Thread_local int recursive;
int aclrtLaunchKernelWithHostArgs(void*,uint32_t,void*,void*,void*,size_t,wio_placeholder*,size_t);
int aclrtGetFunctionName(const void *func,uint32_t cap,char *name) {
    atomic_fetch_add(&names,1);errno=EILSEQ;
    if((uintptr_t)func==3)return 99;
    const char *s=(uintptr_t)func==2?"Cast_fixture":"MemSet_fixture__kernel0";
    if(strlen(s)+1>cap)return 98;
    strcpy(name,s);return 0;
}
int aclrtGetThreadLastTaskId(uint32_t *id) { atomic_fetch_add(&last_queries,1);*id=last_task;errno=EILSEQ;return 0; }
int aclrtStreamGetId(void *stream,int32_t *id) { atomic_fetch_add(&stream_queries,1);*id=stream==(void*)0x44?13:-2;errno=EILSEQ;return 0; }
int aclrtLaunchKernelWithHostArgs(void *func,uint32_t blocks,void *stream,void *cfg,void *args,size_t size,wio_placeholder *ph,size_t phn) {
    last_task=atomic_fetch_add(&calls,1)+1;
    if(errno!=E2BIG||blocks!=48||stream!=(void*)0x44||cfg!=(void*)0x33)atomic_fetch_add(&bad_inputs,1);
    if(size==152) {
        if(!args||phn!=1||!ph||ph[0].addrOffset!=8||ph[0].dataOffset!=24||((unsigned char*)args)[0]!=0x42)atomic_fetch_add(&bad_inputs,1);
        uint64_t relocated=0x120000002018ULL;memcpy((unsigned char*)args+8,&relocated,8);
    }
    if((uintptr_t)func==6&&!recursive) {
        recursive=1;errno=E2BIG;
        int nested=aclrtLaunchKernelWithHostArgs((void*)1,blocks,stream,cfg,args,size,ph,phn);
        recursive=0;if(nested)atomic_fetch_add(&bad_inputs,1);
    }
    errno=(uintptr_t)func==5?EDOM:EAGAIN;
    return (uintptr_t)func==5?77:0;
}
unsigned fake_calls(void){return atomic_load(&calls);}
unsigned fake_names(void){return atomic_load(&names);}
unsigned fake_last_queries(void){return atomic_load(&last_queries);}
unsigned fake_stream_queries(void){return atomic_load(&stream_queries);}
unsigned fake_bad_inputs(void){return atomic_load(&bad_inputs);}
