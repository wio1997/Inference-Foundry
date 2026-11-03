#!/usr/bin/env python3
"""Compare one vLLM metric file between two existing isolated containers, no device work."""
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2]; SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C={'baseline':'wio-vllm-v41-repro169-20261028','stage':'wio-vllm-v41-repro169-stage-20261028'}
P='/vllm-workspace/vllm/vllm/v1/spec_decode/metrics.py'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
def remote(args):
 r=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError('read-only remote command failed '+str(r.returncode))
 return r.stdout
out={}
for label,c in C.items():
 info=json.loads(remote(['docker','inspect','--format','{{json .}}',c]))
 if info['Image']!=IMAGE:raise RuntimeError('image mismatch '+label)
 if not info['State']['Running']:raise RuntimeError('container not running '+label)
 for m in info['Mounts']:
  d=m['Destination'].rstrip('/') or '/'
  if d=='/' or d==P or P.startswith(d+'/') or d.startswith(P+'/'):raise RuntimeError('target mounted '+label)
 code="import hashlib,json;from pathlib import Path;p=Path('/vllm-workspace/vllm/vllm/v1/spec_decode/metrics.py');b=p.read_bytes();print(json.dumps({'sha256':hashlib.sha256(b).hexdigest(),'size':len(b),'resolved':str(p.resolve())}))"
 res=json.loads(remote(['docker','exec',c,'python3','-c',code]))
 if res['resolved']!=P or res['size']>1000000:raise RuntimeError('unexpected file')
 out[label]={'container':c,**res}
x={'files':out,'match':out['baseline']['sha256']==out['stage']['sha256'],'image':IMAGE,'limits':['Current container file contents, not historical live objects or proof no runtime patch.','No NPU, no restart.']}
p=T/'agents/diagnosis/spec-metrics-container-compare.json'
if p.exists():raise RuntimeError('output already exists')
p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
