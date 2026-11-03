#include <acl/acl_rt.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
aclError aclrtBinaryLoadFromData(const void*d,size_t n,const aclrtBinaryLoadOptions*o,aclrtBinHandle*h){(void)d;(void)o;if(n==10)usleep(40000);if(h)*h=(void*)(uintptr_t)(0x10000+n);errno=EBUSY;return 0;}
aclError aclrtBinaryLoadFromFile(const char*p,aclrtBinaryLoadOptions*o,aclrtBinHandle*h){(void)p;(void)o;if(h)*h=(void*)0x14000;errno=EBUSY;return 0;}
aclError aclrtBinaryGetFunction(aclrtBinHandle b,const char*n,aclrtFuncHandle*h){(void)b;if(h)*h=(strstr(n,"DispatchFFNCombine")||!strcmp(n,"unrelated_reuses_target"))?(void*)0x2222:(void*)0x7777;errno=EBUSY;return 0;}
aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle b,uint64_t e,aclrtFuncHandle*h){(void)b;if(h)*h=(void*)(uintptr_t)(0x30000+e);errno=EBUSY;return 0;}
aclError aclrtStreamGetId(aclrtStream s,int32_t*i){(void)s;if(i)*i=31;errno=EBUSY;return 0;}
aclError aclrtGetThreadLastTaskId(uint32_t*i){if(i)*i=55;errno=EBUSY;return 0;}
aclError aclmdlRICaptureGetInfo(aclrtStream s,aclmdlRICaptureStatus*st,aclmdlRI*m){(void)s;if(st)*st=(aclmdlRICaptureStatus)1;if(m)*m=(void*)0x4444;errno=EBUSY;return 0;}
aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle f,uint32_t b,aclrtStream s,aclrtLaunchKernelCfg*c,void*a,size_t z,aclrtPlaceHolderInfo*p,size_t n){(void)b;(void)c;(void)a;(void)z;(void)p;(void)n;if(f==(void*)0x3002a)usleep(50000);if(f==(void*)0x2222){int32_t id;uint32_t tid;aclmdlRICaptureStatus st;aclmdlRI m;aclrtStreamGetId(s,&id);aclrtGetThreadLastTaskId(&tid);aclmdlRICaptureGetInfo(s,&st,&m);}errno=EBUSY;return 0;}
aclError aclrtBinaryUnLoad(aclrtBinHandle b){(void)b;errno=EBUSY;return 0;}
aclError aclmdlRIExecuteAsync(aclmdlRI m,aclrtStream s){(void)m;(void)s;errno=EBUSY;return 0;}
aclError aclmdlRIDestroy(aclmdlRI m){(void)m;errno=EBUSY;return 0;}
