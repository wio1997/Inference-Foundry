from pathlib import Path
from datetime import datetime,timezone
import socket,subprocess,json,hashlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
code="""from pathlib import Path
import re,json,hashlib
root=Path('/usr/local/Ascend/ascend-toolkit/latest')
paths=[root/'include/acl/acl_rt.h',root/'include/acl/acl_base.h',root/'include/runtime/dev.h',root/'include/runtime/rt.h']
result=[]
for p in paths:
 if not p.is_file():continue
 b=p.read_bytes();lines=b.decode(errors='replace').splitlines();hits=[i for i,s in enumerate(lines) if re.search(r'aclrtGetDeviceInfo|aclrtGetDeviceCapability|rtGetDeviceInfo|rtGetDevInfo|CORE_NUM|AICORE.*NUM|AIC.*CORE_NUM',s,re.I)]
 selected=sorted({j for i in hits for j in range(max(0,i-6),min(len(lines),i+8))})
 result.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),total_lines=len(lines),hits=len(hits),excerpts=[str(i+1)+': '+lines[i] for i in selected]))
print(json.dumps(dict(toolkit_resolved=str(root.resolve()),headers=result,no_runtime_library_loaded=True)))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30)
assert r.returncode==0,r.stderr
x=json.loads(r.stdout);x.update(created_utc=datetime.now(timezone.utc).isoformat(),question='PMUmodule activity denominator:does the installed SDK expose AIC/AIVcore counts as a read-only device attribute? Header contract only, no API invoked or NPU context.')
out=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/sdk-device-attribute-header-contract-v12.json');assert not out.exists();payload=json.dumps(x,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(result=x,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
