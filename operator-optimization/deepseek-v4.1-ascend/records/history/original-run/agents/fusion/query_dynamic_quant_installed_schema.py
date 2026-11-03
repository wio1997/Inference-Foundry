#!/usr/bin/env python3
"""Read bounded installed torch_npu metadata and static declarations; no package import."""
import json, shlex, subprocess
from pathlib import Path
T = Path(__file__).resolve().parents[2]
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C = 'wio-vllm-v41-repro169-stage-20261028'
IMAGE = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
SITE = '/usr/local/python3.12.13/lib/python3.12/site-packages'
code = r'''import csv,hashlib,json,mmap,re,sys
from pathlib import Path
site=Path('/usr/local/python3.12.13/lib/python3.12/site-packages')
records=[]; result={'python_version':sys.version,'site':str(site),'distributions':[],'files':[],'binary_hits':[]}
for dist in sorted(site.glob('torch_npu-*.dist-info')):
 if not dist.is_dir() or dist.is_symlink(): continue
 info={'path':str(dist)}
 for n in ('METADATA','RECORD'):
  p=dist/n
  if not p.is_file() or p.is_symlink() or p.stat().st_size>20000000: continue
  b=p.read_bytes(); info[n+'_sha256']=hashlib.sha256(b).hexdigest()
  if n=='METADATA':
   lines=b.decode('utf8',errors='replace').splitlines()
   info['metadata_header']=[x for x in lines if x.startswith(('Name:','Version:'))]
  else:
   for row in csv.reader(b.decode('utf8',errors='replace').splitlines()):
    if row: records.append(row[0])
 result['distributions'].append(info)
selected=[]
for name in records:
 p=Path(name)
 if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!='torch_npu': continue
 if p.suffix not in ('.py','.pyi','.yaml','.yml','.json'): continue
 if len(p.parts)<=3 or p.suffix=='.pyi' or 'dynamic_quant' in name.lower() or 'op_plugin' in name.lower(): selected.append(name)
selected=sorted(set(selected))
if len(selected)>1200: raise SystemExit('file-count bound exceeded')
total=0
for name in selected:
 p=site/name
 if p.is_symlink() or not p.is_file() or p.stat().st_size>4000000: continue
 b=p.read_bytes(); total+=len(b)
 if total>40000000: raise SystemExit('byte bound exceeded')
 try: lines=b.decode('utf8').splitlines()
 except UnicodeDecodeError: continue
 indexes=[i for i,line in enumerate(lines) if 'npu_dynamic_quant' in line]
 if indexes:
  hits=[]
  for i in indexes[:30]:
   lo=max(0,i-2); hi=min(len(lines),i+9)
   hits.append({'line':i+1,'context':[{'line':j+1,'text':lines[j][:1000]} for j in range(lo,hi)]})
  result['files'].append({'path':str(p),'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'hits':hits,'hit_count':len(indexes)})
result['selected_count']=len(selected); result['read_bytes']=total
for path in [site/'torch_npu'/'_C.so',site/'torch_npu'/'lib'/'libtorch_npu.so',site/'torch_npu'/'lib'/'libop_plugin.so']:
 if path.is_symlink() or not path.is_file() or path.stat().st_size>500000000: continue
 with path.open('rb') as f:
  h=hashlib.sha256()
  for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
  f.seek(0)
  with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
   hits=[]
   for m in re.finditer(rb'npu_dynamic_quant',mm):
    start=max(0,m.start()-100); end=min(len(mm),m.end()+400)
    # nearest NUL-bounded string only, prevents binary or unrelated data in output
    a=mm.rfind(b'\x00',start,m.start()); z=mm.find(b'\x00',m.end(),end)
    if a<0 or z<0: continue
    text=mm[a+1:z].decode('ascii','replace')
    if len(text)<=500 and text not in hits: hits.append(text)
    if len(hits)>=30: break
 result['binary_hits'].append({'path':str(path),'size':path.stat().st_size,'sha256':h.hexdigest(),'strings':hits})
print(json.dumps(result,ensure_ascii=True))
'''
def remote(args):
 p=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=120)
 if p.returncode: raise RuntimeError('read-only remote query failed: '+str(p.returncode)+' '+p.stderr[-1000:])
 return p.stdout
if remote(['docker','inspect','--format','{{.Image}}',C]).strip()!=IMAGE: raise RuntimeError('image differs')
mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',C])) or []
for m in mounts:
 dest=m.get('Destination','').rstrip('/') or '/'; root=SITE.rstrip('/')
 if dest=='/' or root==dest or root.startswith(dest+'/') or dest.startswith(root+'/'): raise RuntimeError('site-packages mount detected')
data=json.loads(remote(['docker','exec',C,'python3','-c',code]))
data.update({'container':C,'image_id':IMAGE,'site_mount_checked':True,'no_device_work':True,'no_torch_npu_import':True,'record_scope':'static text paths indexed by installed torch_npu RECORD, plus three exact binary names'})
output=T/'agents/fusion/dynamic-quant-installed-schema.json'
output.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'OK','output':str(output),'distributions':data['distributions'],'selected_count':data['selected_count'],'read_bytes':data['read_bytes'],'matched_text_files':len(data['files']),'binary_files':[b['path'] for b in data['binary_hits']]}))
