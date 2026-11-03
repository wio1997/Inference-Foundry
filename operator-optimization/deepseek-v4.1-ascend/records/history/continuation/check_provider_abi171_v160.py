"""CPU-only SDK ABI layout check; no runtime linking or device API call."""
import json,sys
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,write
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
SOURCE=r'''#include <acl/acl_rt.h>
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
'''
code='C_SOURCE = '+repr(SOURCE)+'\n'+r'''import json,subprocess,hashlib,socket
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171'
cid='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb'
base=Path('/data/tiankuan/wio/migration-171-20261102/runtime-tmp/wio-vllm-v41-repro171-20261102/abi-layout-v160');base.mkdir(mode=0o700,exist_ok=False)
(base/'abi.c').write_text(C_SOURCE)
inside='/tmp/abi-layout-v160'
p=subprocess.run(['docker','exec',cid,'/usr/bin/cc','-std=c11','-Wall','-Wextra','-Werror','-O2','-I/usr/local/Ascend/cann-9.2.0/aarch64-linux/include',inside+'/abi.c','-o',inside+'/abi'],capture_output=True,text=True,timeout=30)
result={'scope':'CPU_ABI_LAYOUT_ONLY','build_rc':p.returncode,'build_stdout':p.stdout,'build_stderr':p.stderr,'source_sha256':hashlib.sha256(C_SOURCE.encode()).hexdigest(),'CANN_runtime_linked_or_called':False,'new_NPU_context':False,'actual_interposition_or_error_state_semantics_tested':False,'overall_complete':False}
if p.returncode==0:
 q=subprocess.run(['docker','exec',cid,'readelf','-d',inside+'/abi'],capture_output=True,text=True,check=True,timeout=10)
 needed=[line.strip() for line in q.stdout.splitlines() if '(NEEDED)' in line]
 assert all(not any(k in line.lower() for k in ['ascend','torch','runtime','acl']) for line in needed),needed
 x=subprocess.run(['docker','exec',cid,inside+'/abi'],capture_output=True,text=True,timeout=10)
 result.update(execution_rc=x.returncode,values=json.loads(x.stdout) if x.returncode==0 else None,stderr=x.stderr,dynamic_dependencies=needed,executable_sha256=hashlib.sha256((base/'abi').read_bytes()).hexdigest())
result['server_directory']=str(base)
out=Path('/data/tiankuan/wio/migration-171-20261102/operator-evidence-v160/sdk-abi-layout.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
d=invoke('target',code,60);write(O/'provider-sdk-abi-layout-v160.json',d);(N/'provider-abi-cache-v160/abi-layout.c').write_text(SOURCE)
print(json.dumps(d,ensure_ascii=False))
assert d['build_rc']==0 and d['execution_rc']==0
