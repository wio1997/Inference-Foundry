"""Read only the exact MC2 kernel/tiling dependencies and saved EP count analyses."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,read,write
import mirror_saved_profile_core_v53 as transport
from read_original_source_archive_v53 import CONTAINER
transport.CONTROL=N/'ssh166-v56-xinyufeng.sock'


def main():
    assert not(O/'MC2-operator-inputs-v56.json').exists()
    pins={
      'EP-count-distribution.json':('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/EP-row-distribution-and-placement-counterfactual-v17.json','4dd1ffc5aaf63d7ad54bcdd23f1a9435962a319976d7c9df8ecd7b5bde1b1c69'),
      'EP-routes.json':('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/EP-routes-ownership-and-placement-tradeoffs-v17.json','b9ded553f181b708eff60eb308e5813d70110a08fba200d94cc45fbd813d9db3')}
    saved=[]
    for name,(remote,sha)in pins.items():
        dest=N/'MC2-operator-source-v56/saved-observer'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        code='''import os,socket,sys
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
p=Path(%r);a=p.stat();assert a.st_size<100*1024*1024
with p.open('rb')as f:
 while True:
  b=f.read(1024*1024)
  if not b:break
  sys.stdout.buffer.write(b)
z=p.stat();assert(a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
'''%remote
        with dest.open('wb')as output:
            p=subprocess.run(transport.ssh_args('python3 -I -S -c '+shlex.quote(code)),stdout=output,stderr=subprocess.PIPE,timeout=120)
        assert p.returncode==0,p.stderr.decode(errors='replace')[-500:]
        b=dest.read_bytes();assert hashlib.sha256(b).hexdigest()==sha
        saved.append(dict(path=remote,local_path=str(dest),sha256=sha,bytes=len(b),historical_hash_match=True))
    prefix='/vllm-workspace/vllm-ascend/'
    rels=['vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py',
        'csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp',
        'csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine.h',
        'csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_tiling.h',
        'csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp',
        'csrc/mc2/dispatch_ffn_combine/op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp',
        'csrc/mc2/dispatch_ffn_combine/op_kernel/utils/block_epilogue_pertoken_swiglu.hpp',
        'csrc/mc2/dispatch_ffn_combine/op_kernel/utils/const_args.hpp']
    code='''import os,socket,subprocess,tarfile,io,json,base64,hashlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
rows=[]
for path in %r:
 p=subprocess.run(['docker','cp',%r+':'+path,'-'],capture_output=True,timeout=20)
 if p.returncode:
  rows.append(dict(path=path,status='READ_ERROR',stderr=p.stderr.decode(errors='replace')[-500:]));continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as tar:
  fs=[m for m in tar.getmembers()if m.isfile()];assert len(fs)==1 and fs[0].size<2*1024*1024
  b=tar.extractfile(fs[0]).read()
 sha=hashlib.sha256(b).hexdigest()
 if path.endswith('/w8a8_dynamic.py'):assert sha=='a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910'
 rows.append(dict(path=path,status='READ',sha256=sha,bytes=len(b),source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(files=rows,container_process_started=False)))
'''%([prefix+r for r in rels],CONTAINER)
    p=transport.run_readonly_python(code,timeout=190)
    rec=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,stdout=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace'),no_model_device_work=True)
    write(N/'MC2-operator-source166-v56.json',rec);assert p.returncode==0
    d=json.loads(rec['stdout']);sources=[]
    for row in d['files']:
        if row['status']!='READ':sources.append(row);continue
        b=base64.b64decode(row['source_b64']);assert hashlib.sha256(b).hexdigest()==row['sha256']
        dest=N/'MC2-operator-source-v56'/Path(row['path']).relative_to('/vllm-workspace')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
        sources.append(dict({k:v for k,v in row.items()if k!='source_b64'},local_path=str(dest)))
    write(O/'MC2-operator-inputs-v56.json',dict(saved_evidence=saved,operator_source=sources,
        source_scope='Knownoriginal166stagefilesystem;W8controlhashmatched. Currentoperator source isnot execution-loadedbinary attestation.',
        no_model_import_device_newcapture_service_health_process_query=True,overall_complete=False))
    print(json.dumps(dict(saved=len(saved),source_read=sum(r['status']=='READ'for r in sources),errors=[r['path']for r in sources if r['status']!='READ'],overall_complete=False)))


if __name__=='__main__':main()
