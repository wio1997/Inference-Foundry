#!/usr/bin/env python3
"""L00: exact CPU-only DeepEP/CANN source/capability inspection, no NPU use."""
import hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1260-deepep-native-contract-v1';IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
CODE=r'''import pathlib,json,hashlib,importlib.metadata as md,re
site=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages');paths=[];listing={};versions={}
for pkg in ['deep_ep','zbal']:
 root=site/pkg;listing[pkg]=[str(p) for p in root.rglob('*.py')]
 paths.extend(root.rglob('*.py'))
for pkg in ['deep_ep','torch','torch_npu']:
 try:versions[pkg]=md.version(pkg)
 except md.PackageNotFoundError:versions[pkg]='UNKNOWN'
root=pathlib.Path('/sgl-workspace/sglang/python/sglang/srt');paths += [root/'layers/moe/token_dispatcher/ascend.py',root/'layers/moe/token_dispatcher/standard.py',root/'layers/communicator.py',root/'layers/moe/fused_moe_triton/layer.py',root/'layers/moe/utils.py']
files=[];budget=2500000
for p in paths:
 if not p.is_file():continue
 raw=p.read_bytes()
 if len(raw)>800000 or len(raw)>budget:continue
 budget-=len(raw);files.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()})
ops=[];bud=600000;vendor=pathlib.Path('/usr/local/Ascend/ascend-toolkit/latest/opp/vendors')
for p in vendor.rglob('*'):
 if not p.is_file() or p.stat().st_size>5000000:continue
 norm=re.sub('[^a-z0-9]','',p.name.lower());related=any(x in norm for x in ['cammoedispatchnormal','moelowlatencydispatchv2'])
 if (related and p.suffix in ['.json','.ini','.h','.cpp']) or p.name.endswith('ops-info.json'):
  raw=p.read_bytes();text=raw.decode(errors='replace');lines=text.splitlines();hits=[i for i,l in enumerate(lines) if any(x in l.lower() for x in ['cammoedispatchnormal','moelowlatencydispatchv2','cam_moe_dispatch_normal','moe_low_latency_dispatch_v2'])]
  if not related and not hits:continue
  ids=sorted({j for i in hits for j in range(max(0,i-4),min(len(lines),i+20))});excerpt=text if related and len(raw)<200000 else '\n'.join(lines[i] for i in ids)
  if len(excerpt)>bud:continue
  bud-=len(excerpt);ops.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'excerpt':excerpt})
model=pathlib.Path('/data/wio/models/GLM-5.3-w8a8/config.json');raw=model.read_bytes();assert hashlib.sha256(raw).hexdigest()=='ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca';cfg=json.loads(raw)
fields=['hidden_size','moe_intermediate_size','intermediate_size','n_routed_experts','num_experts_per_tok','num_attention_heads','num_hidden_layers','num_nextn_predict_layers','index_topk_freq','index_skip_topk_offset']
print('GLM53_DEEPEP_NATIVE_CONTRACT='+json.dumps({'files':files,'listing':listing,'versions':versions,'operator_capability':ops,'model_fields':{k:cfg.get(k) for k in fields},'NPU_devices':False,'framework_imports':False}),flush=True)
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,host=b.NODES[34];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.34'
 args=['docker','run','--rm','--pull=never','--network=none','--cpus=2','--memory=2g','--pids-limit=128','--label','project_id=glm53','--name','glm53-deepep-contract-20261017-34','-v','/data/wio/models/GLM-5.3-w8a8:/data/wio/models/GLM-5.3-w8a8:ro','--entrypoint','python3',IMAGE,'-c',CODE]
 code=r'''import socket,subprocess,json,sys
assert socket.gethostname()==HOST
r=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15);assert r.returncode==0 and r.stdout.strip()==IMAGE
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=110);print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)
'''
 for k,v in [('HOST',host),('IMAGE',IMAGE),('ARGS',args)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 r=b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=170);(RUN/'source-34.json').write_text(json.dumps(r,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0,'npu_allocations':0,'loop':'diagnostics/deepep-dispatch/loops/L00-native-contract-source.json'},indent=2)+'\n');print('nativecontract',r['rc'],flush=True)
if __name__=='__main__':main()
