"""Read only HC operator sources needed for fusion/roofline decisions."""
from pathlib import Path
import hashlib,base64,json
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import N,O,write
import mirror_saved_profile_core_v53 as transport
from read_original_source_archive_v53 import CONTAINER
transport.CONTROL=N/'ssh166-v56-xinyufeng.sock'

def main():
    assert not (O/'HC-operator-inputs-v58.json').exists()
    root='/vllm-workspace/vllm-ascend/'
    paths=[root+p for p in ['csrc/torch_binding.cpp',
      'csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h',
      'csrc/moe/hc_pre/op_kernel/hc_pre.cpp',
      'csrc/moe/hc_pre/op_host/hc_pre.cpp',
      'csrc/moe/hc_post/op_kernel/hc_post.cpp',
      'csrc/moe/hc_post/op_kernel/hc_post.h',
      'csrc/moe/hc_post/op_host/hc_post.cpp']]
    paths += ['/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/'+n for n in ['aclnn_hc_pre.h','aclnn_hc_post.h']]
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
    p=transport.run_readonly_python(code,timeout=220)
    rec=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,stdout=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace'),no_model_device_work=True)
    write(N/'HC-operator-source166-v58.json',rec)
    assert p.returncode==0, rec['stderr'][-500:]
    rows=[]
    for r in json.loads(rec['stdout'])['files']:
      if r['status']!='READ':rows.append(r);continue
      b=base64.b64decode(r['data']);assert hashlib.sha256(b).hexdigest()==r['sha256']
      dest=N/'HC-operator-source-v58'/Path(r['path']).relative_to('/')
      dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
      rows.append(dict({k:v for k,v in r.items() if k!='data'},local_path=str(dest)))
    frozen=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot/vllm-ascend/csrc/torch_binding.cpp')
    binding=next(r for r in rows if r['path'].endswith('torch_binding.cpp'))
    assert binding['sha256']==hashlib.sha256(frozen.read_bytes()).hexdigest()
    write(O/'HC-operator-inputs-v58.json',dict(files=rows,wrapper_matches_historical_snapshot=True,
      source_scope='Exact current166 stage filesystem. HC source/SDK reads only; native loaded object identity remains unproved.',
      no_new_model_Torch_NPU_capture_or_service_process_work=True,overall_complete=False))
    print(json.dumps(dict(read=[r['path'] for r in rows if r['status']=='READ'],unavailable=[r['path'] for r in rows if r['status']!='READ'],wrapper_hash_matches=True,overall_complete=False)))

if __name__=='__main__':main()
