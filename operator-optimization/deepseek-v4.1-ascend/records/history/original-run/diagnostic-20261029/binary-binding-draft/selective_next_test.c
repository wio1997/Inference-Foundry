#include <acl/acl_rt.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"failed at %d\n",__LINE__);abort();}}while(0)
int main(int argc,char **argv){CHECK(argc==2);aclrtBinHandle bin;aclrtFuncHandle named,entry,unused;
 errno=0;CHECK(aclrtBinaryLoadFromData("abc",3,NULL,&bin)==0&&errno==EBUSY);
 for(int i=0;i<12000;i++){CHECK(!aclrtBinaryGetFunction(bin,"unrelated_kernel",&unused));CHECK(!aclrtLaunchKernelWithHostArgs(unused,24,(void*)0x3333,NULL,NULL,0,NULL,0));}
 CHECK(!aclrtBinaryGetFunctionByEntry(bin,42,&entry));CHECK(!aclrtLaunchKernelWithHostArgs(entry,24,(void*)0x3333,NULL,NULL,0,NULL,0));CHECK(!aclrtBinaryGetFunction(bin,"DispatchFFNCombine_actual",&named));
 errno=0;CHECK(!aclrtLaunchKernelWithHostArgs(named,24,(void*)0x3333,NULL,NULL,0,NULL,0)&&errno==EBUSY);
 CHECK(!aclrtLaunchKernelWithHostArgs(entry,24,(void*)0x3333,NULL,NULL,0,NULL,0));
 CHECK(!aclmdlRIExecuteAsync((void*)0x4444,(void*)0x3333));CHECK(!aclmdlRIDestroy((void*)0x4444));
 CHECK(!aclmdlRIExecuteAsync((void*)0x4444,(void*)0x3333));
 CHECK(!aclrtBinaryGetFunction(bin,"unrelated_reuses_target",&unused));CHECK(!aclrtLaunchKernelWithHostArgs(unused,24,(void*)0x3333,NULL,NULL,0,NULL,0));
 CHECK(!aclrtBinaryUnLoad(bin));CHECK(!aclrtLaunchKernelWithHostArgs(named,24,(void*)0x3333,NULL,NULL,0,NULL,0));
 CHECK(!aclrtBinaryLoadFromData("abc",3,NULL,&bin));CHECK(!aclrtLaunchKernelWithHostArgs(named,24,(void*)0x3333,NULL,NULL,0,NULL,0));
 CHECK(!aclrtBinaryLoadFromFile(argv[1],NULL,&bin));CHECK(!aclrtBinaryGetFunctionByEntry(bin,43,&entry));
 CHECK(!aclrtLaunchKernelWithHostArgs(entry,24,(void*)0x3333,NULL,NULL,0,NULL,0));
 puts("selective mock forwarding PASS");return 0;}
