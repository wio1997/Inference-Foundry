"""Read-only ACL device metadata; no setDevice/context/allocation/model/kernel API."""
import ctypes as C
import hashlib
import json
import os
import time
from pathlib import Path
root=Path('/usr/local/Ascend/ascend-toolkit/latest')
p=root/'lib64/libascendcl.so'
lib=C.CDLL(str(p))
lib.aclInit.argtypes=[C.c_char_p];lib.aclInit.restype=C.c_int
lib.aclFinalize.argtypes=[];lib.aclFinalize.restype=C.c_int
lib.aclrtGetDeviceCount.argtypes=[C.POINTER(C.c_uint32)];lib.aclrtGetDeviceCount.restype=C.c_int
lib.aclrtGetDeviceInfo.argtypes=[C.c_uint32,C.c_int,C.POINTER(C.c_int64)];lib.aclrtGetDeviceInfo.restype=C.c_int
names=['aclrtGetLogicDevIdByUserDevId','aclrtGetPhyDevIdByLogicDevId','aclrtGetPhyDevIdByUserDevId','aclrtGetUserDevIdByLogicDevId','aclrtGetLogicDevIdByPhyDevId','aclrtGetUserDevIdByPhyDevId']
for name in names:
    f=getattr(lib,name);f.argtypes=[C.c_int32,C.POINTER(C.c_int32)];f.restype=C.c_int
attributes={'AICPU_CORE_NUM':1,'AICORE_CORE_NUM':101,'CUBE_CORE_NUM':102,'VECTOR_CORE_NUM':201,'UBUF_PER_VECTOR_CORE':204,'TOTAL_GLOBAL_MEM_SIZE':301,'L2_CACHE_SIZE':302,'SMP_ID':401,'PHY_CHIP_ID':402,'MAINBOARD_ID':407,'IS_VIRTUAL':501,'NPU_ARCH':601}
init=lib.aclInit(None)
result={'pid':os.getpid(),'generated_unix':time.time(),'aclInit_rc':init,'library':{'path':str(p),'resolved':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'visibility_env':{k:os.environ[k] for k in ['ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','NPU_VISIBLE_DEVICES'] if k in os.environ},'rows':[],'explicit_set_device_context_allocation_model_kernel_API_calls':0}
if init==0:
    count=C.c_uint32();rc=lib.aclrtGetDeviceCount(C.byref(count));result['device_count']={'rc':rc,'value':count.value if rc==0 else None}
    if rc==0:
        assert count.value<=64
        for user in range(count.value):
            row={'user_device_id':user,'mapping':{},'attributes':{}}
            logic=C.c_int32(-1);lr=lib.aclrtGetLogicDevIdByUserDevId(user,C.byref(logic));row['mapping']['user_to_logic']={'rc':lr,'value':logic.value if lr==0 else None}
            phy=C.c_int32(-1)
            if lr==0:
                pr=lib.aclrtGetPhyDevIdByLogicDevId(logic.value,C.byref(phy));row['mapping']['logic_to_physical']={'rc':pr,'value':phy.value if pr==0 else None}
                back=C.c_int32(-1);br=lib.aclrtGetUserDevIdByLogicDevId(logic.value,C.byref(back));row['mapping']['logic_to_user']={'rc':br,'value':back.value if br==0 else None}
                if pr==0:
                    for name in ['aclrtGetLogicDevIdByPhyDevId','aclrtGetUserDevIdByPhyDevId']:
                        out=C.c_int32(-1);rr=getattr(lib,name)(phy.value,C.byref(out));row['mapping'][name]={'rc':rr,'value':out.value if rr==0 else None}
            direct=C.c_int32(-1);rr=lib.aclrtGetPhyDevIdByUserDevId(user,C.byref(direct));row['mapping']['direct_user_to_physical']={'rc':rr,'value':direct.value if rr==0 else None}
            for name,attr in attributes.items():
                out=C.c_int64(-9223372036854775808);rr=lib.aclrtGetDeviceInfo(user,attr,C.byref(out));row['attributes'][name]={'attribute':attr,'rc':rr,'value':out.value if rr==0 else None}
            result['rows'].append(row)
    result['aclFinalize_rc']=lib.aclFinalize()
print(json.dumps(result))
