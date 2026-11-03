"""Point-read only active projection/quantization numerical callers."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,base64,json
from finalize_origin_coverage_v9 import N,O,write
import mirror_saved_profile_core_v53 as transport
from read_original_source_archive_v53 import CONTAINER
transport.CONTROL=N/'ssh166-v56-xinyufeng.sock'

def main():
 assert not (O/'projection-operator-inputs-v61.json').exists()
 root='/vllm-workspace/vllm-ascend/'
 rel=['vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py','vllm_ascend/attention/context_parallel/dsa_cp.py']
 paths=[root+p for p in rel]
 code='''import socket,os,subprocess,tarfile,io,json,hashlib,base64
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
rows=[]
for path in %r:
 p=subprocess.run(['docker','cp','-L',%r+':'+path,'-'],capture_output=True,timeout=20);assert p.returncode==0,p.stderr.decode()[-300:]
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*') as tar:
  fs=[x for x in tar.getmembers() if x.isfile()];assert len(fs)==1 and fs[0].size<2*1024*1024
  b=tar.extractfile(fs[0]).read()
 rows.append(dict(path=path,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=base64.b64encode(b).decode()))
print(json.dumps(dict(files=rows,container_process_started=False)))
'''%(paths,CONTAINER)
 p=transport.run_readonly_python(code,timeout=80)
 receipt=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,stdout=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace'),no_model_device_work=True)
 write(N/'projection-operator-source166-v61.json',receipt);assert p.returncode==0,receipt['stderr'][-500:]
 frozen=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot/vllm-ascend')
 rows=[]
 for r in json.loads(receipt['stdout'])['files']:
  b=base64.b64decode(r['data']);assert hashlib.sha256(b).hexdigest()==r['sha256']
  relative=Path(r['path']).relative_to(root);assert hashlib.sha256((frozen/relative).read_bytes()).hexdigest()==r['sha256']
  dest=N/'projection-operator-source-v61'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
  rows.append(dict({k:v for k,v in r.items() if k!='data'},local_path=str(dest),historical_snapshot_hash_matches=True))
 write(O/'projection-operator-inputs-v61.json',dict(files=rows,other_source='Reusev60pinneddsa3505/indexerd5d6 callers andpriorF32/F33 exactoperatorcontracts;no otherframeworkreading.',
  current_loaded_native_binary_identity_proved=False,no_model_Torch_NPU_newcapture_health_process_or_implementation=True,overall_complete=False))
 print(json.dumps(dict(read=2,historical_hashes_match=True,overall_complete=False)))

if __name__=='__main__':main()
