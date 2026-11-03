"""Read exact attention/indexer operator contracts without starting the container."""
from pathlib import Path
import hashlib,base64,json
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import N,O,read,write
import mirror_saved_profile_core_v53 as transport
from read_original_source_archive_v53 import CONTAINER
transport.CONTROL=N/'ssh166-v56-xinyufeng.sock'

def batch(paths,label):
    receipt=N/('attention-operator-source166-v60-'+label+'.json')
    assert not receipt.exists()
    code='''import socket,os,subprocess,tarfile,io,json,hashlib,base64
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
rows=[]
for path in %r:
 p=subprocess.run(['docker','cp','-L',%r+':'+path,'-'],capture_output=True,timeout=20)
 if p.returncode:
  rows.append(dict(path=path,status='UNAVAILABLE_AT_EXACT_PATH',stderr=p.stderr.decode(errors='replace')[-400:]));continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*') as tar:
  fs=[x for x in tar.getmembers() if x.isfile()];assert len(fs)==1 and fs[0].size<2*1024*1024
  b=tar.extractfile(fs[0]).read()
 rows.append(dict(path=path,status='READ',sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=base64.b64encode(b).decode()))
print(json.dumps(dict(files=rows,container_process_started=False)))
'''%(paths,CONTAINER)
    p=transport.run_readonly_python(code,timeout=30+len(paths)*20)
    rec=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,stdout=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace'),no_model_device_work=True)
    write(receipt,rec);assert p.returncode==0,rec['stderr'][-500:]
    index=O/'attention-operator-inputs-v60.json'
    d=read(index) if index.exists() else dict(files=[],source_scope='Current original166stage exactoperatorfiles; not per-executionloadedbinaryidentity.',overall_complete=False)
    rows=[]
    frozen=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot/vllm-ascend')
    for r in json.loads(rec['stdout'])['files']:
      if r['status']!='READ':rows.append(r);continue
      b=base64.b64decode(r['data']);assert hashlib.sha256(b).hexdigest()==r['sha256']
      dest=N/'attention-operator-source-v60'/Path(r['path']).relative_to('/')
      dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
      row=dict({k:v for k,v in r.items() if k!='data'},local_path=str(dest))
      if r['path'].startswith('/vllm-workspace/vllm-ascend/'):
        prior=frozen/Path(r['path']).relative_to('/vllm-workspace/vllm-ascend')
        if prior.is_file():row['historical_snapshot_hash_matches']=hashlib.sha256(prior.read_bytes()).hexdigest()==r['sha256'];assert row['historical_snapshot_hash_matches']
      rows.append(row)
    d['files'].extend(rows);write(index,d)
    print(json.dumps([dict(path=r['path'],status=r['status']) for r in rows]))

if __name__=='__main__':
    root='/vllm-workspace/vllm-ascend/'
    batch([root+p for p in ['vllm_ascend/attention/dsa_v41.py','vllm_ascend/models/deepseek_v41/indexer.py',
      'csrc/attention/sparse_flash_mla/sparse_flash_mla_torch_adpt.h','csrc/attention/sparse_flash_mla/sparse_flash_mla_torch_adpt.cpp',
      'csrc/attention/sparse_flash_mla/op_kernel/sparse_flash_mla.cpp',
      'csrc/attention/quant_lightning_indexer_v2/quant_lightning_indexer_v2_torch_adpt.h',
      'csrc/attention/quant_lightning_indexer_v2/quant_lightning_indexer_v2_torch_adpt.cpp',
      'csrc/attention/quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2.cpp']], 'entry')
