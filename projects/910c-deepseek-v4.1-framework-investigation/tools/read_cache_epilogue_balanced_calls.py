#!/usr/bin/env python3
"""Read two saved balanced steady CSVs for exact typed epilogue grammars; no new profile."""
from pathlib import Path
from hashlib import sha256
import json,subprocess,zlib
INNER=r'''
from pathlib import Path
from hashlib import sha256
import csv,json,sys
names=['RmsNorm','RmsNormCast','InplacePartialRotaryMul','DynamicQuant','ScatterNdUpdateSk','HcPre','HcPost','MatMulV2','MatMulV3','MatMul','Cast','ZerosLike','_pool_kernel','_cache_update_kernel','VllmAscendQuantLightningIndexerV2','SparseFlashMla']
out=[]
for spec in json.loads(sys.argv[1]):
 p=Path(spec['path']);assert p.is_relative_to('/data/tiankuan/wio') and p.name=='kernel_details.csv';raw=p.read_bytes();rows=list(csv.DictReader(raw.decode().splitlines()));assert rows
 fields=list(rows[0]);selected=[]
 for i,r in enumerate(rows):
  if r.get('Type') in names or r.get('Name') in names:
   selected.append(dict(csv_row=i+2,data={k:v for k,v in r.items() if k in ['Name','Type','Stream ID','Task ID','Task Type','Start Time(us)','Duration(us)','Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Block Num','Mix Block Num','aicore_time(us)','aiv_time(us)']}))
 out.append(dict(**spec,sha256=sha256(raw).hexdigest(),total_csv_rows=len(rows),headers=fields,selected=selected))
print(json.dumps(out))
'''
REMOTE=r'''
import json,subprocess,sys
name='wio-vllm-v41-repro171-20261102';c,=json.loads(subprocess.check_output(['docker','inspect',name]));assert c['State']['Running'] and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
# CSVs live in the owned host tree; do not invoke a runtime/framework backend.
r=subprocess.run(['python3','-B','-I','-S','-c',sys.argv[1],sys.argv[2]],capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr;print(r.stdout)
'''
if __name__=='__main__':
 root=Path('audit/execution-search');inv=json.loads(Path('audit/balanced-profile-171/kernel-full-inventory.json').read_text());spec=[{k:r[k] for k in ['phase','ep','dp','path']} for r in inv['files'] if r['phase']=='steady' and r['ep'] in [0,8]];assert len(spec)==2
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER,json.dumps(spec)],capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr)
 data=json.loads(r.stdout);raw=json.dumps(data,separators=(',',':')).encode();dest=root/'cache-epilogue-balanced-calls.json.zlib';dest.write_bytes(zlib.compress(raw,9));meta=dict(profiles=[{k:v for k,v in d.items() if k!='selected'}|dict(selected_rows=len(d['selected'])) for d in data],archive=dict(path=str(dest),sha256=sha256(dest.read_bytes()).hexdigest(),uncompressed_sha256=sha256(raw).hexdigest()),reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='OnlyEP0/EP8steady savedkernel CSVs; representative native geometry/call grammar notallranks/fullrequests. Selection excludesunrelatedoptypes; rawtotal counts/headers retained.',new_NPU_work=0,model_loads=0,service_requests=0,new_profile_collection=0,remote_writes=[]);(root/'cache-epilogue-balanced-call-pins.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(profiles=len(data),selected_rows=[len(d['selected']) for d in data],headers=data[0]['headers'][:16],new_profile_collection=0)))
