from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
out=M/'F04-norm-cast-kernel-and-router-consumer-contract-v29.json';assert not out.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
root=Path('/vllm-workspace/vllm-ascend');A=root/'vllm_ascend';rows=[]
for rel,expected,cap in [('csrc/moe/rms_norm_cast/op_kernel/rms_norm_cast.h','5a05b11cce4a727b8c48aec4fe764d854c71c5a8ba072dad315a2acec342d472',180),('csrc/moe/rms_norm_cast/op_kernel/rms_norm_cast_base.h','7e2d92f63d3c47c845d915c7c1a54b76eccf075bfabf88326d9c7984b1c02358',140),('csrc/moe/rms_norm_cast/op_kernel/rms_norm_cast.cpp','84fbfd73ba944cc5e382c89f7d7114cf651521c4956df4253a44b4cda18b643f',40),('csrc/moe/rms_norm_cast/rms_norm_cast_torch_adpt.h',None,50)]:
 p=root/rel;b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();ls=b.decode().splitlines()
 if expected:assert sha==expected
 assert len(ls)<=cap,(rel,'sourcecap')
 rows.append(dict(path=str(p),sha256=sha,source_scope='Exactopfileonly',source=[str(i+1)+': '+l for i,l in enumerate(ls)]))
for p,expected,symbols in [(A/'models/deepseek_v41/model.py','101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',[('forward',285)]),(A/'ops/fused_moe/router/fused_topk_router.py',None,[('_compute_routing',204)])]:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
 if expected:assert sha==expected
 ls=b.decode().splitlines();tree=ast.parse(b.decode());fs=[]
 for name,line in symbols:
  matches=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name==name and n.lineno<=line<=n.end_lineno]
  assert len(matches)==1,(str(p),name);n=matches[0];assert n.end_lineno-n.lineno<220
  owners=[c.name for c in ast.walk(tree)if isinstance(c,ast.ClassDef)and c.lineno<=n.lineno<=c.end_lineno]
  fs.append(dict(name=name,owners=owners,start=n.lineno,end=n.end_lineno,profile_frame_line=line,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 rows.append(dict(path=str(p),sha256=sha,functions=fs))
assert sum(len(r.get('source',[]))+sum(len(f['source'])for f in r.get('functions',[]))for r in rows)<700
print(json.dumps(dict(files=rows,no_model_or_torch_import=True,scope='FourexactRmsNormCastsourcefiles andtwoactualnativecaller methods;noSDKorbroadmodelaudit.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
