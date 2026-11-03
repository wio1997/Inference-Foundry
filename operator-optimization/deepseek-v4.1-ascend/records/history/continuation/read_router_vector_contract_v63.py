"""Read only exact operator numeric/tiling files to answer row-selection/fusion questions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,base64,json
from finalize_origin_coverage_v9 import N,O,read,write
import mirror_saved_profile_core_v53 as transport
from read_original_source_archive_v53 import CONTAINER
transport.CONTROL=N/'ssh166-v56-xinyufeng.sock'

def main():
 assert not (O/'router-vector-operator-inputs-v63.json').exists()
 rels=['csrc/moe/moe_gating_top_k_hash/op_kernel/'+x for x in ['moe_gating_top_k_hash.cpp','moe_gating_top_k_hash_without_group.h','moe_gating_top_k_hash_e_k_fullload.h','moe_gating_top_k_hash_generalized.h']]
 rels+=['csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_tiling.cpp','csrc/attention/inplace_partial_rotary_mul/op_kernel/inplace_partial_rotary_mul.h','csrc/attention/inplace_partial_rotary_mul/op_kernel/inplace_partial_rotary_mul.cpp']
 code='''import socket,os,subprocess,tarfile,io,json,hashlib,base64
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
rows=[]
for rel in %r:
 path='/vllm-workspace/vllm-ascend/'+rel
 p=subprocess.run(['docker','cp','-L',%r+':'+path,'-'],capture_output=True,timeout=20);assert p.returncode==0,p.stderr.decode()[-300:]
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*') as tar:
  fs=[x for x in tar.getmembers() if x.isfile()];assert len(fs)==1 and fs[0].size<2*1024*1024
  b=tar.extractfile(fs[0]).read()
 rows.append(dict(path=path,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=base64.b64encode(b).decode()))
print(json.dumps(dict(files=rows,container_process_started=False)))
'''%(rels,CONTAINER)
 p=transport.run_readonly_python(code,timeout=160);rec=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,stdout=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace'),no_model_device_work=True)
 write(N/'router-vector-source166-v63.json',rec);assert p.returncode==0,rec['stderr'][-500:]
 rows=[]
 for r in json.loads(rec['stdout'])['files']:
  b=base64.b64decode(r['data']);assert hashlib.sha256(b).hexdigest()==r['sha256'];rel=Path(r['path']).relative_to('/vllm-workspace/vllm-ascend');dest=N/'router-vector-operator-source-v63'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
  old=Path('/Users/wio/work/vllm-ascend-perf')/rel
  rows.append(dict({k:v for k,v in r.items() if k!='data'},local_path=str(dest),other_local_repository_hash_matches=old.exists() and hashlib.sha256(old.read_bytes()).hexdigest()==r['sha256']))
 write(O/'router-vector-operator-inputs-v63.json',dict(files=rows,source_scope='7exactoperatorfiles:currenthashTopKscore/normalization/tie/tiling andpartialRoPEtail/broadcast/castcost. Reuseexistingpinnednorm/quant/pad/outputcallers; no independentframeworkresearch.',current_loaded_native_binary_identity_proved=False,no_model_Torch_NPU_capture_health_process_or_implementation=True,overall_complete=False))
 print(json.dumps(dict(read=len(rows),bytes=sum(r['bytes'] for r in rows),other_local_hashes_match=sum(r['other_local_repository_hash_matches'] for r in rows),overall_complete=False)))

if __name__=='__main__':main()
