#!/usr/bin/env python3
"""Find data_ptr handling in static installed Triton Python launcher source."""
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,os,pathlib
roots=[pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/triton'),pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/triton_ascend')]
out=[]
for root in roots:
 if not root.exists():continue
 for base,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=sorted(d for d in dirs if not pathlib.Path(base,d).is_symlink())
  for name in sorted(names):
   if not name.endswith(('.py','.h','.cpp','.c')) or not ('driver' in name or 'launch' in name):continue
   p=pathlib.Path(base,name)
   try:
    if p.is_symlink() or p.stat().st_size>1000000:continue
    b=p.read_bytes();s=b.decode('utf8');lines=s.splitlines()
    indices=[i for i,l in enumerate(lines) if 'data_ptr' in l or 'storage_offset' in l]
    if indices:
     snippets=[]
     for i in indices[:12]:snippets.append({'line':i+1,'excerpt':lines[max(0,i-3):min(len(lines),i+5)]})
     out.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'snippets':snippets})
   except (OSError,UnicodeDecodeError):pass
print(json.dumps({'roots':[str(r) for r in roots if r.exists()],'hits':out}))'''
e=base64.b64encode(remote.encode()).decode()
cmd='docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=90,check=True)
data=json.loads(p.stdout);o=pathlib.Path(__file__).resolve().parent/'engram-launch-pointer-sources.json';o.write_text(json.dumps(data,indent=2)+'\n');print('roots',len(data['roots']),'sources',len(data['hits']),'saved',o)
