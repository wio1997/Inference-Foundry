from pathlib import Path
from datetime import datetime, timezone
import json, socket, subprocess, hashlib

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030')
P=ROOT/'hostprof-20261030T212812Z-360e0c72'
assert json.loads((P/'controller-finished.json').read_text())['state']=='CAPTURED_AND_RESTORED'
assert json.loads((P/'restore-result.json').read_text())['status']=='PASS_ORIGINAL_BASELINE_RESTORED'
base=json.loads(subprocess.check_output(['docker','inspect','wio-vllm-v41-repro166-20261030']))[0]
assert base['Id']=='8757dcc7a2f8db672bbf6ddfc1ed29e6cff74d5dd89199770b7b6f3b27fd48e3' and base['State']['Running']
code='''from pathlib import Path
import ctypes,json,hashlib
root=Path('/usr/local/Ascend/ascend-toolkit/latest')
header=root/'include/acl/acl_rt.h'
library=root/'lib64/libascendcl.so'
lib=ctypes.CDLL(str(library))
get=lib.aclrtGetDeviceInfo
get.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.POINTER(ctypes.c_int64)]
get.restype=ctypes.c_int
attrs={'AICORE_CORE_NUM':101,'CUBE_CORE_NUM':102,'VECTOR_CORE_NUM':201,'UBUF_PER_VECTOR_CORE_BYTES':204,'IS_VIRTUAL':501,'NPU_ARCH':601}
rows=[]
for device in range(16):
 record={'device':device,'attributes':{}}
 for name,attr in attrs.items():
  value=ctypes.c_int64(-1)
  rc=get(device,attr,ctypes.byref(value))
  record['attributes'][name]={'attr_id':attr,'returncode':rc,'value':value.value if rc==0 else None}
 rows.append(record)
print(json.dumps(dict(toolkit=str(root.resolve()),header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),library=str(library.resolve()),rows=rows,
                    explicit_calls=['ctypes.CDLL(libascendcl)','aclrtGetDeviceInfo'],aclInit_called=False,set_device_or_context_called=False,inference_or_kernel_launch=False)))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=45)
assert r.returncode==0,(r.returncode,r.stderr)
result=json.loads(r.stdout)
out=P/'analysis/readonly-device-core-attributes-v12.json';assert not out.exists();out.parent.mkdir(exist_ok=True)
result.update(checked_utc=datetime.now(timezone.utc).isoformat(),scope='16logical device IDs in166same image/toolkit. Reports hardware attributes;not MAC throughput/latency,not old169runtime attestation.',
              limits=['No ACL init,set-device,context creation,inference or kernel API is explicitly called;runtime internals are not independently attested. Read-only attribute query finishes before posthealth check.',
                      'Reported core count is one denominator fact;does not establish counter averaging,pipe ops/cycle,clock integration or full-module NPU utilization.'])
payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
