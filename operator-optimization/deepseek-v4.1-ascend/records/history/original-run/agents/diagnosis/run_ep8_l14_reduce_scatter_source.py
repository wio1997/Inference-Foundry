#!/usr/bin/env python3
"""Execute only the read-only frozen EP8 DB query within the verified stage."""
from pathlib import Path,PurePosixPath
import subprocess,shlex,json,hashlib
here=Path(__file__).resolve().parent
src=here/'query_ep8_l14_reduce_scatter_source.py';out=here/'ep8-l14-reduce-scatter-source.json'
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
expected='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602';image='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
root=PurePosixPath('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime')
def call(args,timeout):
 r=subprocess.run([ssh,shlex.join(args)],capture_output=True,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError(r.stderr[-1500:])
 return r.stdout
def inspect():
 cid,img,raw=call(['docker','inspect','--format','{{.Id}}|{{.Image}}|{{json .Mounts}}',stage],30).strip().split('|',2);mounts=json.loads(raw)
 assert cid==expected and img==image
 assert len([m for m in mounts if m.get('Destination')=='/work' and m.get('Source')=='/data/tiankuan/wio/main' and m.get('Type')=='bind'])==1
 for m in mounts:
  d=PurePosixPath(m['Destination'])
  assert d==PurePosixPath('/work') or not(d==root or d in root.parents or root in d.parents)
 return (cid,img,sorted(mounts,key=lambda x:json.dumps(x,sort_keys=True)))
before=inspect();code=src.read_text();compile(code,str(src),'exec')
raw=call(['docker','exec',stage,'nice','-n','15','ionice','-c','3','python3','-c',code],300)
assert inspect()==before
data=json.loads(raw);data.update(script_sha256=hashlib.sha256(code.encode()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),stage_id=expected,no_new_device_work=True)
out.write_text(json.dumps(data,indent=2)+'\n')
print('sidecar layer',data['sidecar_call'][0]['layer_index'],'relative_ns',data['relative'])
