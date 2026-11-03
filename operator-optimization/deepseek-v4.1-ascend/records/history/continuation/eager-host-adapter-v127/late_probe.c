#define _GNU_SOURCE
#include "host_packet_snapshot.h"
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int (*launch_fn)(void*,uint32_t,void*,void*,void*,size_t,wio_placeholder*,size_t);
int main(int argc,char **argv) {
    assert(argc==3);
    if(!strcmp(argv[1],"plugin")) {
        void *h=dlopen(argv[2],RTLD_LAZY|RTLD_LOCAL);assert(h);
        int(*entry)(int,char**)=(int(*)(int,char**))dlsym(h,"main");assert(entry);
        char *v[]={"plugin","one",NULL};return entry(2,v);
    }
    if(!strcmp(argv[1],"missing")) {
        unsigned(*probe)(void)=(unsigned(*)(void))dlsym(RTLD_DEFAULT,"wio_host_trace_probe");assert(probe);
        printf("{\"status\":\"PASS\",\"probe_bits\":%u,\"no_runtime_call\":true}\n",probe());return 0;
    }
    void *h=dlopen(argv[2],RTLD_LAZY|RTLD_LOCAL);assert(h);
    launch_fn real=(launch_fn)dlsym(h,"aclrtLaunchKernelWithHostArgs");assert(real);
    unsigned(*calls)(void)=(unsigned(*)(void))dlsym(h,"fake_calls");assert(calls);
    unsigned char packet[152]={0};packet[0]=0x42;wio_placeholder ph={8,24};errno=E2BIG;
    assert(real((void*)1,48,(void*)0x44,(void*)0x33,packet,152,&ph,1)==0&&errno==EAGAIN);
    printf("{\"status\":\"PASS\",\"direct_handle_lookup_bypasses_adapter\":true,\"calls\":%u}\n",calls());
    return 0;
}
