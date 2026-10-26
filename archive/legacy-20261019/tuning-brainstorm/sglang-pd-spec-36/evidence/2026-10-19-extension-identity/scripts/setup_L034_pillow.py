#!/usr/bin/env python3
import datetime,hashlib,json,pathlib,subprocess,sys,os
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L034-R003-pillow';O.mkdir(parents=True,exist_ok=False);D=R/'.deps/L034-pillow';assert not D.exists()
cmd=[sys.executable,'-m','pip','install','--no-deps','--only-binary=:all:','--index-url','https://pypi.org/simple','--cache-dir',str(R/'.deps/pip-cache'),'--target',str(D),'--report',str(O/'install-report.json'),'Pillow==12.0.0']
code=125
with (O/'stdout.log').open('xb') as so,(O/'stderr.log').open('xb') as se:
 try:code=subprocess.run(cmd,stdout=so,stderr=se,timeout=240).returncode
 except subprocess.TimeoutExpired:code=124
j={'status':'PASS' if code==0 else 'UNAVAILABLE','exit_code':code,'command':cmd,'target':str(D),'existing_environments_modified':False,'native_executed':False,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if code==0:
 sys.path.insert(0,str(D));import PIL;from PIL import Image
 j['Pillow']=PIL.__version__;j['Image_module_sha256']=hashlib.sha256(pathlib.Path(Image.__file__).read_bytes()).hexdigest();assert PIL.__version__=='12.0.0'
(O/'result.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({'status':j['status'],'exit_code':code}));sys.exit(code)
