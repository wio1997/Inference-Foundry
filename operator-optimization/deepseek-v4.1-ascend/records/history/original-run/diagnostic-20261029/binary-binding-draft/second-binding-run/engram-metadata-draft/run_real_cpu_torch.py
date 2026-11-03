"""Stage an independent CPU test and execute only in the owned 169 stage."""
from pathlib import Path
import base64,zlib,shlex,subprocess
p=Path(__file__).resolve().parent;names=['engram_metadata.py','test_real_cpu_torch.py'];files={n:base64.b64encode((p/n).read_bytes()).decode() for n in names}
d='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/engram-metadata-real-cpu-test'
code='''from pathlib import Path
import base64,os,subprocess
p=Path(DIR);p.mkdir(exist_ok=True)
for n,b in FILES.items():
 q=p/n;v=base64.b64decode(b)
 if q.exists() and q.read_bytes()!=v:raise RuntimeError('refuse overwrite '+str(q))
 if not q.exists():q.write_bytes(v)
tmp=p/'tmp';tmp.mkdir(exist_ok=True)
env=dict(os.environ);env.update(TMPDIR=str(tmp),PYTHONNOUSERSITE='1',PYTHONPATH='',ASCEND_RT_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
r=subprocess.run(['python3',str(p/'test_real_cpu_torch.py')],capture_output=True,text=True,env=env,timeout=90)
(p/'test.stdout').write_text(r.stdout);(p/'test.stderr').write_text(r.stderr);print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)
'''
raw='DIR='+repr(d)+'\nFILES='+repr(files)+'\n'+code;z=base64.b64encode(zlib.compress(raw.encode())).decode();inner='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(z)+')))' 
outer='import subprocess;raise SystemExit(subprocess.call('+repr(['docker','exec','wio-vllm-v41-repro169-stage-20261028','nice','-n','15','python3','-c',inner])+'))'
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(['python3','-c',outer])],capture_output=True,text=True,timeout=120)
(p/'real-cpu-stage.stdout').write_text(r.stdout);(p/'real-cpu-stage.stderr').write_text(r.stderr)
print(r.stdout,r.stderr);raise SystemExit(r.returncode)
