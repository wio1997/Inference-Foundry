#include <acl/acl_rt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x))abort();}while(0)
int main(int argc,char **argv){CHECK(argc==2);aclrtBinHandle b;aclrtFuncHandle f;
 if(!strcmp(argv[1],"missing")){
  b=(void*)0x12345;CHECK(!aclrtBinaryGetFunction(b,"DispatchFFNCombine_actual",&f));CHECK(!aclrtBinaryUnLoad(b));
  CHECK(!aclrtLaunchKernelWithHostArgs(f,24,(void*)0x3333,NULL,NULL,0,NULL,0));puts("selective stress forwarding PASS");return 0;
 }
 CHECK(!aclrtBinaryLoadFromData("abcdefghijk",3,NULL,&b));
 if(!strcmp(argv[1],"pending")){
  CHECK(!aclrtBinaryGetFunctionByEntry(b,42,&f));
  for(int i=0;i<6;i++)CHECK(!aclrtLaunchKernelWithHostArgs(f,24,(void*)0x3333,NULL,NULL,0,NULL,0));
  CHECK(!aclrtBinaryGetFunction(b,"DispatchFFNCombine_actual",&f));return 0;
 }
 CHECK(!aclrtBinaryGetFunction(b,"DispatchFFNCombine_actual",&f));
 if(!strcmp(argv[1],"evict")){
  for(size_t n=4;n<=10;n++)CHECK(!aclrtBinaryLoadFromData("abcdefghijk",n,NULL,&b));
  CHECK(!aclrtLaunchKernelWithHostArgs(f,24,(void*)0x3333,NULL,NULL,0,NULL,0));
  CHECK(!aclrtBinaryLoadFromData("abc",3,NULL,&b));CHECK(!aclrtBinaryGetFunction(b,"DispatchFFNCombine_actual",&f));
  CHECK(!aclrtLaunchKernelWithHostArgs(f,24,(void*)0x3333,NULL,NULL,0,NULL,0));
 } else if(!strcmp(argv[1],"cap")){
  for(int i=0;i<1000;i++)CHECK(!aclrtLaunchKernelWithHostArgs(f,24,(void*)0x3333,NULL,NULL,0,NULL,0));
 } else abort();
 puts("selective stress forwarding PASS");return 0;}
