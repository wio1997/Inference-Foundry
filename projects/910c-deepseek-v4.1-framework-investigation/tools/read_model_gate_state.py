#!/usr/bin/env python3
"""Read one initial-exec flag and descriptor rotation metadata only, on host CPU."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_embedding_quant_description import REMOTE as DESCRIPTION
ROOT=Path('audit/execution-search')
FLAGS=r'''
from pathlib import Path
import json,subprocess
name='wio-vllm-v41-repro171-20261102';c,=json.loads(subprocess.check_output(['docker','inspect',name]));assert c['Id']=='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb' and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
rows=[]
for line in subprocess.check_output(['docker','top',name,'-eo','pid,comm'],text=True).splitlines()[1:]:
 fields=line.split();pid=int(fields[0]);comm=fields[1];value=None;status='READ'
 try:
  for piece in Path('/proc',str(pid),'environ').read_bytes().split(b'\0'):
   k,sep,v=piece.partition(b'=')
   if k==b'VLLM_MOE_SKIP_PADDING':value=v.decode() if v in [b'0',b'1'] else 'NONBOOLEAN_NOT_EMITTED'
 except OSError as e:status=type(e).__name__
 rows.append(dict(host_pid=pid,comm=comm,status=status,VLLM_MOE_SKIP_PADDING=value))
print(json.dumps(dict(container_id=c['Id'],image=c['Image'],processes=rows,scope='Only whitelisted initial-exec environment bytes, not runtime Python env/cache. No process arguments, other env, container process, framework import, writes or device work.')))
'''
def main():
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',FLAGS],capture_output=True,text=True,timeout=70)
 if p.returncode:raise RuntimeError(p.stderr)
 flags=json.loads(p.stdout)
 remote=DESCRIPTION[:DESCRIPTION.index('d=json.loads(b);')]+r'''
d=json.loads(b);optional=d.get('optional',{});rotation=optional.get('quarot',{}).get('rotation_map',{}).get('global_rotation')
assert rotation is None or isinstance(rotation,str)
print(json.dumps(dict(container_id=c['Id'],image=c['Image'],path=path,sha256=sha256(b).hexdigest(),bytes=len(b),global_rotation_present=rotation is not None,global_rotation_relative_path=rotation,qualification='Descriptor metadata only; no rotation/weight tensor payload or framework import.')))
'''
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote],capture_output=True,text=True,timeout=70)
 if p.returncode:raise RuntimeError(p.stderr)
 desc=json.loads(p.stdout);assert desc['sha256']==json.loads((ROOT/'embedding-quant-description.json').read_text())['sha256']
 out=dict(flags=flags,descriptor=desc,reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),base_descriptor_reader_sha256=sha256(Path('tools/read_embedding_quant_description.py').read_bytes()).hexdigest(),new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],whole_chain_complete=False)
 (ROOT/'model-gate-state.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(processes=len(flags['processes']),readable=sum(r['status']=='READ' for r in flags['processes']),flag_values=sorted({str(r['VLLM_MOE_SKIP_PADDING']) for r in flags['processes'] if r['status']=='READ'}),descriptor=desc,new_NPU_work=0,remote_writes=[]),ensure_ascii=False))
if __name__=='__main__':main()
