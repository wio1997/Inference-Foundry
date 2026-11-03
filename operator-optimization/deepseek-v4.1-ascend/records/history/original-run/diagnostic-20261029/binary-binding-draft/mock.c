#include <acl/acl_rt.h>
#include <errno.h>
#include <stdint.h>
aclError aclrtBinaryLoadFromData(const void*d,size_t n,const aclrtBinaryLoadOptions*o,aclrtBinHandle*h) { (void)d;(void)o;if(n==77){aclrtFuncHandle x; aclrtBinaryGetFunction((void*)0x1111,"nested",&x);} if(n==999){errno=EINVAL;return -7;} if(h)*h=(void*)0x1111;errno=EBUSY;return 0;}
aclError aclrtBinaryLoadFromFile(const char*p,aclrtBinaryLoadOptions*o,aclrtBinHandle*h){(void)p;(void)o;if(h)*h=(void*)0x1111;errno=EBUSY;return 0;}
aclError aclrtBinaryGetFunction(aclrtBinHandle b,const char*n,aclrtFuncHandle*h){(void)b;(void)n;if(h)*h=(void*)0x2222;errno=EBUSY;return 0;}
aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle b,uint64_t e,aclrtFuncHandle*h){(void)b;(void)e;if(h)*h=(void*)0x2222;return 0;}
aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle f,uint32_t b,aclrtStream s,aclrtLaunchKernelCfg*c,void*a,size_t z,aclrtPlaceHolderInfo*p,size_t n){(void)f;(void)b;(void)s;(void)c;(void)a;(void)z;(void)p;(void)n;errno=EBUSY;return 0;}
aclError aclrtBinaryUnLoad(aclrtBinHandle b){(void)b;errno=EBUSY;return 0;}
aclError aclrtStreamGetId(aclrtStream s,int32_t *id){(void)s;if(id)*id=31;errno=EBUSY;return 0;}
aclError aclrtGetThreadLastTaskId(uint32_t *id){if(id)*id=55;errno=EBUSY;return 0;}
aclError aclmdlRICaptureGetInfo(aclrtStream s,aclmdlRICaptureStatus *st,aclmdlRI *m){(void)s;if(st)*st=(aclmdlRICaptureStatus)1;if(m)*m=(void*)0x4444;errno=EBUSY;return 0;}
aclError aclmdlRIExecuteAsync(aclmdlRI m,aclrtStream s){(void)m;int32_t sid;uint32_t task;aclrtStreamGetId(s,&sid);aclrtGetThreadLastTaskId(&task);errno=EBUSY;return 0;}
