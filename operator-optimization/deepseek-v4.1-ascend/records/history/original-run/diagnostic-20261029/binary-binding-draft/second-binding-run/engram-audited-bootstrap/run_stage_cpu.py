"""Execute isolated CPU mocks in existing owned stage; never load NPU."""
from pathlib import Path
import base64,zlib,shlex,subprocess
p=Path(__file__).resolve().parent;names=['audited_observer.py','engram_bootstrap.py','sitecustomize.py','audit_window.py','test_audited.py','test_audit_window.py','test_import_hook.py','test_cpu_torch_audited.py','test_install_failure.py'];files={n:base64.b64encode((p/n).read_bytes()).decode() for n in names}
d='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/engram-audited-bootstrap-cpu-v4'
code='''from pathlib import Path
import base64,os,subprocess,json
p=Path(DIR);p.mkdir(exist_ok=True)
for n,b in FILES.items():
 q=p/n;v=base64.b64decode(b)
 if q.exists() and q.read_bytes()!=v:raise RuntimeError('refuse overwrite '+str(q))
 if not q.exists():q.write_bytes(v)
tmp=p/'tmp';tmp.mkdir(exist_ok=True)
env=dict(os.environ);env.pop('WIO_ENGRAM_META_ENABLE',None);env.pop('PYTHONOPTIMIZE',None);env.update(TMPDIR=str(tmp),PYTHONPATH=str(p),PYTHONNOUSERSITE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0',ASCEND_RT_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
out=[]
for name in ('test_audited.py','test_audit_window.py','test_import_hook.py','test_cpu_torch_audited.py','test_install_failure.py'):
 r=subprocess.run(['python3',str(p/name)],capture_output=True,text=True,env=env,timeout=30);out.append({'test':name,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 if r.returncode:print(json.dumps(out));raise SystemExit(r.returncode)
(p/'cpu-results.json').write_text(json.dumps(out,indent=2)+'\\n');print(json.dumps(out))
'''
raw='DIR='+repr(d)+'\nFILES='+repr(files)+'\n'+code;packed=base64.b64encode(zlib.compress(raw.encode())).decode();inner='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))' 
outer='import subprocess;raise SystemExit(subprocess.call('+repr(['docker','exec','wio-vllm-v41-repro169-stage-20261028','nice','-n','15','python3','-c',inner])+'))';r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(['python3','-c',outer])],capture_output=True,text=True,timeout=120)
(p/'stage-cpu-v4-result.json').write_text(r.stdout);(p/'stage-cpu-v4-stderr.txt').write_text(r.stderr);print(r.stdout,r.stderr);raise SystemExit(r.returncode)
