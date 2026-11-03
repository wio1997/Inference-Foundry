import json,sys
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
H=Path('/Users/wio/work/166-vllm-analysis-continuation-20261030/shared-profile-analysis-v171')
inner=r'''import json,hashlib
from pathlib import Path
p=Path('/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h');lines=p.read_text().splitlines();segments=[]
for lo,hi in [(3986,4014),(5158,5230)]:segments.append({'path':str(p),'first_line':lo,'last_line':hi,'text':'\n'.join(lines[lo-1:hi])})
p=Path('/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl.h')
if p.exists():
 lines=p.read_text().splitlines()
 for i,line in enumerate(lines):
  if 'aclrtGetSocName' in line or 'aclInit(' in line or 'aclFinalize(' in line:segments.append({'path':str(p),'line':i+1,'text':'\n'.join(lines[max(0,i-5):i+4])})
print(json.dumps(segments))'''
code='import json,subprocess,hashlib\nfrom pathlib import Path\ninner='+repr(inner)+'''\nc='wio-vllm-v41-repro171-20261102';p=subprocess.run(['docker','exec',c,'python3','-c',inner],capture_output=True,text=True,timeout=15);r={'SDK_doc':{'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-1000:]},'commands':[]}
for args in [['npu-smi','info','-m'],['npu-smi','info','-l'],['npu-smi','info','-t','product','-i','0','-c','0'],['npu-smi','info','-t','board','-i','0','-c','0']]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);r['commands'].append({'argv':args,'rc':p.returncode,'text':p.stdout+p.stderr})
D=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint171');D.mkdir(exist_ok=True);(D/'MAPPING-SDK-COMMANDS.json').write_text(json.dumps(r,indent=2)+'\\n')
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168');r['captured_device_info']=[]
for p in sorted((P/'runtime').glob('*_ascend_pt/PROF_*/device_*/info.json.*')):
 b=p.read_bytes()
 if not p.suffix[1:].isdigit():continue
 try:data=json.loads(b)
 except Exception as e:data={'parse_error':str(e),'raw_text':b.decode(errors='replace')[:1500]}
 r['captured_device_info'].append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'data':data})
(D/'MAPPING-AND-CAPTURE-METADATA.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))'''
r=invoke('target',code,timeout=65)
if r['SDK_doc']['rc']==0:r['SDK_doc']['parsed']=json.loads(r['SDK_doc']['stdout']);r['SDK_doc']['stdout']='Parsedsegmentsstored'
(H/'physical-mapping-and-capture-metadata.json').write_text(json.dumps(r,indent=2)+'\n');out={k:v for k,v in r.items() if k!='captured_device_info'};out['captured_info_count']=len(r['captured_device_info']);out['captured_info_example']=r['captured_device_info'][:1];print(json.dumps(out,ensure_ascii=False))
