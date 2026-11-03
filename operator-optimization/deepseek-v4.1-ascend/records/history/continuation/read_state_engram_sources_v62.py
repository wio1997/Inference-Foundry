"""Point-read three operator kernels with already known historical SHA controls."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,base64,json
from finalize_origin_coverage_v9 import N,O,read,write
import mirror_saved_profile_core_v53 as transport
from read_original_source_archive_v53 import CONTAINER
transport.CONTROL=N/'ssh166-v56-xinyufeng.sock'

def main():
 assert not (O/'state-engram-operator-inputs-v62.json').exists()
 expected={r['path']:r['sha256'] for r in read(O/'C2-state-opportunity-contract-v24.json')['source_files'] if r['path'].endswith('/compressor_triton.py')}
 expected.update({k:v for k,v in read(O/'Engram-hash-lookup-publication-and-gate-contract-v30.json')['source_files'].items() if k.endswith(('/engram_hash.py','/engram_lookup.py'))})
 assert len(expected)==3
 code='''import socket,os,subprocess,tarfile,io,json,hashlib,base64
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
rows=[]
for path,sha in %r.items():
 p=subprocess.run(['docker','cp','-L',%r+':'+path,'-'],capture_output=True,timeout=20);assert p.returncode==0,p.stderr.decode()[-300:]
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*') as tar:
  fs=[x for x in tar.getmembers() if x.isfile()];assert len(fs)==1 and fs[0].size<2*1024*1024
  b=tar.extractfile(fs[0]).read()
 assert hashlib.sha256(b).hexdigest()==sha
 rows.append(dict(path=path,sha256=sha,bytes=len(b),data=base64.b64encode(b).decode()))
print(json.dumps(dict(files=rows,container_process_started=False)))
'''%(expected,CONTAINER)
 p=transport.run_readonly_python(code,timeout=100)
 rec=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,stdout=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace'),no_model_device_work=True)
 write(N/'state-engram-source166-v62.json',rec);assert p.returncode==0,rec['stderr'][-500:]
 rows=[]
 for r in json.loads(rec['stdout'])['files']:
  b=base64.b64decode(r['data']);assert hashlib.sha256(b).hexdigest()==r['sha256']
  dest=N/'state-engram-operator-source-v62'/Path(r['path']).relative_to('/vllm-workspace/vllm-ascend')
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
  rows.append(dict({k:v for k,v in r.items() if k!='data'},local_path=str(dest),historical_source_hash_matches=True))
 write(O/'state-engram-operator-inputs-v62.json',dict(files=rows,source_scope='3operatornumeric/statekernels;reuseoldv24/25/30consumercontracts andpinnedEngramcommon/layer. No newframework/allocation/loaderaudit.',
  current_loaded_native_binary_identity_proved=False,no_new_model_Torch_NPU_capture_health_process_or_implementation=True,overall_complete=False))
 print(json.dumps(dict(read=3,all_historical_hashes_match=True,overall_complete=False)))

if __name__=='__main__':main()
