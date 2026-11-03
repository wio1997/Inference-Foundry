#include <acl/acl_rt.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
typedef aclError (*sdk_launch)(aclrtFuncHandle,uint32_t,aclrtStream,aclrtLaunchKernelCfg*,void*,size_t,aclrtPlaceHolderInfo*,size_t);
typedef int (*name_fn)(const void*,uint32_t,char*);
typedef int (*last_fn)(uint32_t*);
typedef int (*stream_fn)(void*,int32_t*);
_Static_assert(__builtin_types_compatible_p(__typeof__(&aclrtLaunchKernelWithHostArgs),sdk_launch),"SDK launch prototype");
_Static_assert(__builtin_types_compatible_p(__typeof__(&aclrtGetFunctionName),name_fn),"SDK name getter prototype");
_Static_assert(__builtin_types_compatible_p(__typeof__(&aclrtGetThreadLastTaskId),last_fn),"SDK task getter prototype");
_Static_assert(__builtin_types_compatible_p(__typeof__(&aclrtStreamGetId),stream_fn),"SDK stream getter prototype");
_Static_assert(sizeof(aclError)==sizeof(int) && sizeof(int)==4,"return ABI");
_Static_assert(sizeof(void*)==8 && sizeof(size_t)==8 && sizeof(uint32_t)==4,"AArch64 scalar ABI");
_Static_assert(sizeof(aclrtPlaceHolderInfo)==8 && _Alignof(aclrtPlaceHolderInfo)==4,"placeholder layout");
_Static_assert(offsetof(aclrtPlaceHolderInfo,addrOffset)==0 && offsetof(aclrtPlaceHolderInfo,dataOffset)==4,"placeholder offsets");
int main(void){printf("{\"pointer_bytes\":%zu,\"size_t_bytes\":%zu,\"return_bytes\":%zu,\"placeholder_bytes\":%zu,\"placeholder_align\":%zu,\"addr_offset\":%zu,\"data_offset\":%zu}\n",sizeof(void*),sizeof(size_t),sizeof(aclError),sizeof(aclrtPlaceHolderInfo),_Alignof(aclrtPlaceHolderInfo),offsetof(aclrtPlaceHolderInfo,addrOffset),offsetof(aclrtPlaceHolderInfo,dataOffset));return 0;}
