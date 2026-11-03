#include <acl/acl_rt.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"failed %d\n",__LINE__);abort();}}while(0)
static void *work(void *arg){(void)arg;
  for(int i=0;i<100;i++) {aclrtBinHandle b;aclrtFuncHandle f;
   errno=0;CHECK(aclrtBinaryLoadFromData("abc",3,NULL,&b)==0&&errno==EBUSY&&b==(void*)0x1111);
   errno=0;CHECK(aclrtBinaryGetFunction(b,"kernel\"test",&f)==0&&errno==EBUSY&&f==(void*)0x2222);
   errno=EDOM;CHECK(aclrtBinaryGetFunctionByEntry(b,42,&f)==0&&errno==EDOM);
   errno=0;CHECK(aclrtLaunchKernelWithHostArgs(f,24,(void*)0x3333,NULL,NULL,0,NULL,0)==0&&errno==EBUSY);
   errno=0;CHECK(aclrtBinaryUnLoad(b)==0&&errno==EBUSY);
  }return NULL;}
int main(int argc,char**argv){CHECK(argc==2); pthread_t t[4];
 for(int i=0;i<4;i++) CHECK(!pthread_create(&t[i],NULL,work,NULL));
 for(int i=0;i<4;i++) CHECK(!pthread_join(t[i],NULL));
 aclrtBinHandle h=NULL; errno=0;CHECK(aclrtBinaryLoadFromData((void*)1,999,NULL,&h)==-7&&errno==EINVAL);
 errno=0;CHECK(aclrtBinaryLoadFromData((void*)1,3,NULL,&h)==0&&errno==EBUSY);
 errno=0;CHECK(aclrtBinaryLoadFromFile(argv[1],NULL,&h)==0&&errno==EBUSY);
 errno=0;CHECK(aclrtBinaryLoadFromData((void*)1,77,NULL,&h)==0&&errno==EBUSY);
 int32_t sid=0;uint32_t task=0;aclmdlRICaptureStatus st=(aclmdlRICaptureStatus)0;aclmdlRI model=NULL;
 errno=0;CHECK(aclrtStreamGetId((void*)0x3333,&sid)==0&&sid==31&&errno==EBUSY);
 errno=0;CHECK(aclrtGetThreadLastTaskId(&task)==0&&task==55&&errno==EBUSY);
 errno=0;CHECK(aclmdlRICaptureGetInfo((void*)0x3333,&st,&model)==0&&model==(void*)0x4444&&errno==EBUSY);
 errno=0;CHECK(aclmdlRIExecuteAsync(model,(void*)0x3333)==0&&errno==EBUSY);
 pid_t child=fork(); CHECK(child>=0);if(!child){errno=0; if(aclrtBinaryUnLoad((void*)0x1111)!=0||errno!=EBUSY)_exit(1);_exit(0);} int status=0;CHECK(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
 puts("mock forwarding PASS");return 0;}
