"""Read exact metadata-operator files from original container, without starting it."""
from datetime import datetime,timezone
from pathlib import Path
import base64,hashlib,json
from finalize_origin_coverage_v9 import N,O,W,read,write
import mirror_saved_profile_core_v53 as transport
CONTAINER='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'

def main():
    receipt=N/'planner-source166-v71.json';assert not receipt.exists()
    transport.CONTROL=N/'ssh166-v71-xinyufeng.sock'
    controls={'/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py':'3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd'}
    rels=['csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp','csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.h','csrc/attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/quant_lightning_indexer_v2_metadata_aicpu.cpp','csrc/attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/quant_lightning_indexer_v2_metadata_aicpu.h','csrc/attention/sparse_flash_mla/op_kernel/sparse_flash_mla_kernel_metadata.h','csrc/attention/quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2_metadata.h','csrc/attention/common/op_kernel/aicpu_common.h','vllm_ascend/ascend_forward_context.py']
    paths=list(controls)+['/vllm-workspace/vllm-ascend/'+x for x in rels]+['/vllm-workspace/vllm/vllm/config/speculative.py','/vllm-workspace/vllm/vllm/v1/spec_decode/llm_base_proposer.py']
    code='''import os,socket,json,hashlib,subprocess,tarfile,io,base64,time
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
caps=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/profile-capabilities-index-v12-v2.json')
assert hashlib.sha256(caps.read_bytes()).hexdigest()=='79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095'
paths=%s;controls=%s;container=%s;rows=[];t0=time.monotonic()
for path in paths:
 assert time.monotonic()-t0<100
 p=subprocess.run(['docker','cp',container+':'+path,'-'],capture_output=True,timeout=15)
 if p.returncode:
  rows.append(dict(path=path,status='READ_ERROR',returncode=p.returncode,stderr=p.stderr.decode(errors='replace')[-1000:]))
  if path in controls:break
  continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as tar:
  fs=[m for m in tar.getmembers()if m.isfile()];assert len(fs)==1
  f=fs[0];assert f.size<2*1024*1024;b=tar.extractfile(f).read()
 sha=hashlib.sha256(b).hexdigest()
 if path in controls and sha!=controls[path]:
  rows.append(dict(path=path,status='CONTROL_HASH_MISMATCH',sha256=sha,expected=controls[path]));break
 rows.append(dict(path=path,status='READ',bytes=len(b),sha256=sha,archive_mtime=f.mtime,historical_control_match=path in controls,source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(container_id=container,execution_host=socket.gethostname(),files=rows,container_started=False,container_inspect_health_or_process_query=False,model_Torch_NPU_capture_implementation=False,scope='Exactfiledockerarchivecopiesonly;nooperatororcontainerexecuted. Currentfilebytesnotproofhistoricallyloadedbinary.')))
'''%(repr(paths),repr(controls),repr(CONTAINER))
    result=transport.run_readonly_python(code,timeout=125);stdout=result.stdout.decode(errors='replace');stderr=result.stderr.decode(errors='replace')
    write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,stdout=stdout,stderr=stderr,scope='Authenticatedoriginal166knowncontaineroperatorfilesonly;nohealth/process/devicework.'))
    assert result.returncode==0,stderr[-1200:]
    payload=json.loads(stdout);files=[]
    public={r['repo_path']:r for r in read(O/'planner-public-source-locators-v71.json')['files']}
    for r in payload['files']:
        if r['status']!='READ':files.append(r);continue
        b=base64.b64decode(r['source_b64']);assert hashlib.sha256(b).hexdigest()==r['sha256']
        local=N/'planner-operator-source-v71'/Path(r['path']).relative_to('/')
        local.parent.mkdir(parents=True,exist_ok=True);assert not local.exists();local.write_bytes(b)
        old={k:v for k,v in r.items()if k!='source_b64'};old['local_path']=str(local)
        rel=r['path'].removeprefix('/vllm-workspace/vllm-ascend/')
        if rel in public:
            p=public[rel];old.update(public_blob_SHA=p['blob_sha'],public_immutable_blob_url=p['git_url'],public_source_bytes_match=b==Path(p['path']).read_bytes())
        if rel=='vllm_ascend/ascend_forward_context.py':prior=W/'frozen-source-snapshot/vllm-ascend'/rel
        elif r['path'].startswith('/vllm-workspace/vllm/'):prior=W/'frozen-source-snapshot/vllm'/r['path'].removeprefix('/vllm-workspace/vllm/')
        else:prior=None
        if prior is not None:old.update(prior_frozen_path=str(prior),prior_frozen_SHA_match=hashlib.sha256(prior.read_bytes()).hexdigest()==r['sha256'])
        files.append(old)
    write(O/'planner-operator-source-index-v71.json',dict(files=files,container_id=CONTAINER,execution_host=payload['execution_host'],all_requested_read=len(files)==len(paths)and all(x['status']=='READ'for x in files),caller_historical_control_match=files[0].get('historical_control_match',False),new_readonly_remote_source=True,loaded_native_binary_attested=False,no_container_started_model_Torch_NPU_capture_health_process_kill_implementation=True,overall_complete=False))
    print(json.dumps(dict(files_read=sum(x['status']=='READ'for x in files),requested=len(paths),errors=[{'path':x['path'],'status':x['status']}for x in files if x['status']!='READ'],public_source_matches=sum(x.get('public_source_bytes_match',False)for x in files),prior_frozen_support_matches=sum(x.get('prior_frozen_SHA_match',False)for x in files),no_container_started=True,overall_complete=False)))

if __name__=='__main__':main()
