import json,sys,base64
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
H=Path('/Users/wio/work/166-vllm-analysis-continuation-20261030/shared-profile-analysis-v171');b=base64.b64encode((H/'query_acl_device_metadata.py').read_bytes()).decode()
code='import json,subprocess,base64,time\nfrom pathlib import Path\nb='+repr(b)+'''\nO=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint171');O.mkdir(exist_ok=True);f=O/'query_acl_device_metadata.py';f.write_bytes(base64.b64decode(b));c='wio-vllm-v41-repro171-20261102';inner=base64.b64decode(b).decode();p=subprocess.run(['docker','exec',c,'python3','-c',inner],capture_output=True,text=True,timeout=35);r={'generated_unix':time.time(),'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-4000:]};
(O/'ACL-DEVICE-METADATA-RAW.json').write_text(json.dumps(r,indent=2)+'\\n')
if p.returncode==0:
 for line in p.stdout.splitlines():
  try:
   z=json.loads(line)
   if isinstance(z,dict) and 'aclInit_rc' in z:r['review']=z;break
  except ValueError:pass
 if 'review' not in r:r['decode_error']='No metadata JSONline found;rawstdout preserved'
 else:r['stdout']='Decodedresultinreview;rawruntimefooter preservedremote'
(O/'ACL-DEVICE-METADATA-QUERY.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))'''
r=invoke('target',code,timeout=45);(H/'ACL-device-metadata-query.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='review'},ensure_ascii=False))
if 'review' in r:
 print('init/final',r['review']['aclInit_rc'],r['review'].get('aclFinalize_rc'),'count',r['review'].get('device_count'));print(json.dumps(r['review']['rows'][:2],ensure_ascii=False))
