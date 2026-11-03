#include <acl/acl_rt.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#define C(x) do{if(!(x))abort();}while(0)
static pthread_barrier_t barrier;static aclrtBinHandle bin;static aclrtFuncHandle fn;
static void *launch(void*x){(void)x;pthread_barrier_wait(&barrier);C(!aclrtLaunchKernelWithHostArgs(fn,24,(void*)0x3333,NULL,NULL,0,NULL,0));return NULL;}
static void *unload(void*x){(void)x;pthread_barrier_wait(&barrier);usleep(10000);C(!aclrtBinaryUnLoad(bin));return NULL;}
int main(int argc,char**argv){C(argc==3);C(!aclrtBinaryLoadFromData("abc",3,NULL,&bin));
 if(!strcmp(argv[1],"concurrent")){C(!aclrtBinaryGetFunctionByEntry(bin,42,&fn));pthread_t a,b;pthread_barrier_init(&barrier,NULL,2);pthread_create(&a,NULL,launch,NULL);pthread_create(&b,NULL,unload,NULL);pthread_join(a,NULL);pthread_join(b,NULL);return 0;}
 C(!aclrtBinaryGetFunction(bin,"DispatchFFNCombine_actual",&fn));
 if(!strcmp(argv[1],"graph")||!strcmp(argv[1],"overflow")){
  int count=!strcmp(argv[1],"graph")?12000:3;
  for(int i=0;i<count;i++)C(!aclrtLaunchKernelWithHostArgs(fn,24,(void*)0x3333,NULL,NULL,0,NULL,0));
  FILE*f=fopen(argv[2],"w");C(f);fclose(f);
  C(!aclmdlRIExecuteAsync((void*)(uintptr_t)(!strcmp(argv[1],"graph")?0x4444:0x4446),(void*)0x3333));unlink(argv[2]);return 0;
 }
 if(!strcmp(argv[1],"normal")){for(int i=0;i<12000;i++)C(!aclrtLaunchKernelWithHostArgs(fn,24,(void*)0x3333,NULL,NULL,0,NULL,0));FILE*f=fopen(argv[2],"w");C(f);fclose(f);for(int i=0;i<3;i++)C(!aclrtLaunchKernelWithHostArgs(fn,24,(void*)0x3333,NULL,NULL,0,NULL,0));unlink(argv[2]);}
 else if(!strcmp(argv[1],"unknown")){for(int i=0;i<100;i++)C(!aclrtLaunchKernelWithHostArgs(fn,24,(void*)0x3333,NULL,NULL,0,NULL,0));}else {abort();}
 return 0;}
