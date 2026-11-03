from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'EP-mapping-loader-and-quant-contract-excerpts-v19.json';assert not OUT.exists()
loc=P/'EP-expert-mapping-symbol-locations-v19.json';b=loc.read_bytes();assert hashlib.sha256(b).hexdigest()=='320a0e1ded6537a9fd8b8a150bfd743484766489c1e57f8edcae75fe201a543e'
pins={f['path']:f['sha256']for f in json.loads(b)['files']}
ar='/vllm-workspace/vllm-ascend/vllm_ascend/';vr='/vllm-workspace/vllm/vllm/'
specs=[(ar+'models/deepseek_v41/model.py',[(175,260),(262,303),(1785,1796)]),(ar+'ops/fused_moe/moe_comm_method.py',[(563,615)]),(ar+'quantization/methods/w8a8/w8a8_dynamic.py',[(203,235),(238,277),(279,333),(351,395)]),(vr+'model_executor/layers/fused_moe/expert_map_manager.py',[(22,113),(336,352),(417,455)]),(vr+'model_executor/layers/fused_moe/routed_experts.py',[(575,626),(660,708)])]
code='from pathlib import Path\nimport json,hashlib\npins='+repr(pins)+'\nspecs='+repr(specs)+'''\nrows=[]
for name,ranges in specs:
 p=Path(name);s=p.read_text();assert hashlib.sha256(s.encode()).hexdigest()==pins[name];ls=s.splitlines()
 rows.append(dict(path=name,sha256=pins[name],excerpts=[dict(start=a,end=b,lines=[str(i+1)+': '+ls[i]for i in range(a-1,b)])for a,b in ranges]))
print(json.dumps(dict(files=rows,no_module_import_or_device_work=True)))
'''
q=subprocess.run(['docker','exec','bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d','python3','-c',code],capture_output=True,text=True,timeout=80);assert q.returncode==0,q.stderr
v=json.loads(q.stdout);v.update(created_utc=datetime.now(timezone.utc).isoformat(),source_locations_sha256=hashlib.sha256(b).hexdigest(),scope='Profile-raised exact ranges in five files, bound to current166 stage source hashes; no model/device/service work.',overall_complete=False)
payload=(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(result=v,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
