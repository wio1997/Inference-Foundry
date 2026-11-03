import base64,hashlib,json,sys,zlib
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102');from control import invoke
H=Path(__file__).resolve().parent;s=(H/'recover_public_evidence.py').read_text();compile(s,'cpu-recovery.py','exec');payload=base64.b64encode(zlib.compress(s.encode())).decode()
code='PAYLOAD='+repr(payload)+'\n'+r'''from pathlib import Path
import base64,json,os,subprocess,zlib
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint176-derived');p.mkdir(exist_ok=True)
assert not (p/'SUBMITTED.json').exists(),'Observe existing CPU parser rather than relaunch'
f=p/'cpu-recovery.py';f.write_bytes(zlib.decompress(base64.b64decode(PAYLOAD)));compile(f.read_text(),str(f),'exec')
log=(p/'cpu-recovery.log').open('ab');c=subprocess.Popen(['timeout','300','python3','-I','-S',str(f)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
h={'pid':c.pid,'start_ticks':Path('/proc/'+str(c.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],'root':str(p),'CPU_only':True,'timeout_seconds':300}
(p/'SUBMITTED.json').write_text(json.dumps(h,indent=2)+'\n');print(json.dumps(h))'''
x=invoke('target',code,timeout=30);(H/'cpu-recovery-submission.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps(x))
