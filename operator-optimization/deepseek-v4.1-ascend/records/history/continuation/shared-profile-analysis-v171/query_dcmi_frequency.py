"""Read only current/max clock fields from the installed management API."""
import ctypes as C
import ctypes.util
import hashlib
import json
import re
import time
from pathlib import Path
O=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint171')
header=Path('/usr/local/Ascend/driver/include/dcmi_interface_api.h')
b=header.read_bytes();text=b.decode(errors='replace')
assert re.search(r'dcmi_init\s*\(\s*(void)?\s*\)',text)
assert 'DCMI_FREQ_AICORE_MAX = 9' in text
candidates=[ctypes.util.find_library('dcmi'),'/usr/local/dcmi/lib64/libdcmi.so','/usr/local/dcmi/libdcmi.so','/usr/local/Ascend/driver/lib64/libdcmi.so','/usr/local/Ascend/driver/lib64/driver/libdcmi.so']
lib=None;attempts=[]
for name in candidates:
    if not name:continue
    try:lib=C.CDLL(name);attempts.append({'library':name,'loaded':True});break
    except OSError as e:attempts.append({'library':name,'error':str(e)})
result={'generated_unix':time.time(),'header':{'path':str(header),'sha256':hashlib.sha256(b).hexdigest()},'library_attempts':attempts,'rows':[],'setter_or_workload_API_calls':0}
if lib is not None:
    lib.dcmi_init.argtypes=[];lib.dcmi_init.restype=C.c_int
    lib.dcmi_get_device_frequency.argtypes=[C.c_int,C.c_int,C.c_int,C.POINTER(C.c_uint32)];lib.dcmi_get_device_frequency.restype=C.c_int
    rc=lib.dcmi_init();result['dcmi_init_rc']=rc
    result['loaded_library_name']=lib._name
    if Path(lib._name).is_file():result['loaded_library_sha256']=hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()
    if rc==0:
        for card in range(8):
            for chip in range(2):
                row={'NPU_card_id':card,'chip_id':chip,'frequencies':{}}
                for name,kind in [('AICORE_MAX',9),('AICORE_CURRENT',7),('VECTOR_CURRENT',12),('HBM',6)]:
                    out=C.c_uint32(4294967295);rr=lib.dcmi_get_device_frequency(card,chip,kind,C.byref(out));row['frequencies'][name]={'enum':kind,'rc':rr,'value':out.value if rr==0 else None}
                result['rows'].append(row)
(O/'DCMI-FREQUENCY-QUERY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
