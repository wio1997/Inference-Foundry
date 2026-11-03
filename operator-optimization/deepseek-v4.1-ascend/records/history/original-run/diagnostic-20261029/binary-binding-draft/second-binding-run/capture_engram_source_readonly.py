from pathlib import Path
import shlex,subprocess,json
here=Path(__file__).resolve().parent
inner='''from pathlib import Path
import hashlib,json
p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41');out={}
for name,start,end in [('engram/common.py',55,72),('engram/embedding.py',333,348),('engram/layer.py',41,53)]:
 f=p/name;b=f.read_bytes();rows=b.decode().splitlines();out[name]={'sha256':hashlib.sha256(b).hexdigest(),'lines':[[i,rows[i-1]] for i in range(start,min(end,len(rows))+1)]}
print(json.dumps(out))'''
outer='''import subprocess,json
out={}
for n in ('wio-vllm-v41-repro169-stage-20261028','wio-vllm-v41-binding-diag2-169-20261029'):
 x=json.loads(subprocess.check_output(['docker','inspect',n],text=True))[0]
 out[n]={'image':x['Image'],'readonlyrootfs':x['HostConfig']['ReadonlyRootfs'],'mount_destinations':[m['Destination'] for m in x['Mounts']]}
r=subprocess.run(ARG,capture_output=True,text=True,check=True);out['source']=json.loads(r.stdout);print(json.dumps(out,indent=2))
'''
args=['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',inner];src='ARG='+repr(args)+'\n'+outer
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(['python3','-c',src])],capture_output=True,text=True,timeout=30)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
(here/'engram-source-readonly-evidence.json').write_text(r.stdout);print('saved source+image evidence')
