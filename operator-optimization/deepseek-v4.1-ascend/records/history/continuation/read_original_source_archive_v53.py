"""Read exact source files from the known original container filesystem without starting it."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
import subprocess

from finalize_origin_coverage_v9 import N, O, read, write
from mirror_saved_profile_core_v53 import run_readonly_python

CONTAINER = 'bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'


def main():
    assert not (N / 'original-source-files166-v53.json').exists()
    controls = {
        '/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py':'68c48e5df0ea4a7cc1f04436b7b9153eb53802ecd9bb55d38db8738508daac6d',
        '/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py':'308c00f63fcb6f44518ec3a56d8f902c7bba75928c2e3aea30fbe1f3f473973b',
        '/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py':'101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',
        '/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py':'568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c',
        '/vllm-workspace/vllm-ascend/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py':'a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910',
        '/vllm-workspace/vllm/vllm/v1/engine/detokenizer.py':'213d71cf6eefcea061b28b656cf8a08af60c9f3513403e724b16876995f3de93',
    }
    extra = [
        '/vllm-workspace/vllm/vllm/distributed/eplb/eplb_state.py',
        '/vllm-workspace/vllm/vllm/model_executor/utils.py',
        '/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/__init__.py',
        '/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/layer.py',
        '/vllm-workspace/vllm/vllm/outputs.py',
        '/vllm-workspace/vllm/vllm/v1/engine/output_processor.py',
        '/vllm-workspace/vllm/vllm/v1/engine/async_llm.py',
        '/vllm-workspace/vllm/vllm/v1/engine/logprobs.py',
        '/vllm-workspace/vllm/vllm/entrypoints/openai/completion/serving.py',
        '/vllm-workspace/vllm-ascend/vllm_ascend/quantization/method_adapters.py',
        '/vllm-workspace/vllm-ascend/vllm_ascend/quantization/configs/modelslim_config.py',
        '/vllm-workspace/vllm-ascend/vllm_ascend/eplb/adaptor/vllm_adaptor.py',
        '/vllm-workspace/vllm-ascend/vllm_ascend/eplb/eplb_updator.py',
        '/vllm-workspace/vllm-ascend/vllm_ascend/eplb/core/eplb_device_transfer_loader.py',
    ]
    code = '''import os,socket,json,hashlib,subprocess,tarfile,io,base64,time
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
caps=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/profile-capabilities-index-v12-v2.json')
assert hashlib.sha256(caps.read_bytes()).hexdigest()=='79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095'
container=%s;controls=%s;paths=list(controls)+%s;rows=[];t0=time.monotonic()
for path in paths:
 assert time.monotonic()-t0<130
 p=subprocess.run(['docker','cp',container+':'+path,'-'],capture_output=True,timeout=20)
 if p.returncode:
  rows.append(dict(path=path,status='READ_ERROR',returncode=p.returncode,stderr=p.stderr.decode(errors='replace')[-1000:]))
  if path in controls:break
  continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as tar:
  fs=[m for m in tar.getmembers()if m.isfile()];assert len(fs)==1
  f=fs[0];assert f.size<2*1024*1024
  b=tar.extractfile(f).read();sha=hashlib.sha256(b).hexdigest()
 if path in controls and sha!=controls[path]:
  rows.append(dict(path=path,status='CONTROL_HASH_MISMATCH',sha256=sha,expected_sha256=controls[path]));break
 rows.append(dict(path=path,status='READ',bytes=len(b),sha256=sha,archive_mtime=f.mtime,
  historical_control_match=path in controls,source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(container_id=container,execution_host=socket.gethostname(),files=rows,
 container_process_started=False,container_inspect_or_health_or_process_query=False,
 scope='Read-onlydockerarchivecopyfromtheknownoriginalcontainerID;no model/code imported or container started. Currentfilesystem bytes do not attest historical loaded code objects.'),ensure_ascii=False))
''' % (repr(CONTAINER), repr(controls), repr(extra))
    result = run_readonly_python(code, timeout=160)
    stdout=result.stdout.decode(errors='replace');stderr=result.stderr.decode(errors='replace')
    payload=json.loads(stdout) if result.returncode==0 else None
    receipt=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,
        stdout=stdout,stderr=stderr,application_json_valid=payload is not None,
        no_model_torch_device_service_health_process_work=True)
    write(N / 'original-source-files166-v53.json', receipt)
    assert payload is not None, stderr[-1500:]
    local_rows=[]
    for row in payload['files']:
        if row['status']!='READ':
            local_rows.append(row);continue
        b=base64.b64decode(row['source_b64']);assert hashlib.sha256(b).hexdigest()==row['sha256']
        p=N/'original-source-cache-v53'/Path(row['path']).relative_to('/vllm-workspace')
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        else:p.write_bytes(b)
        local_rows.append(dict({k:v for k,v in row.items()if k!='source_b64'},local_path=str(p)))
    controls_ok=sum(r['status']=='READ'and r.get('historical_control_match',False)for r in local_rows)==len(controls)
    write(O/'original-source-cache-index-v53.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        container_id=CONTAINER,files=local_rows,all_six_historical_controls_match=controls_ok,
        all_requested_files_read=len(local_rows)==len(controls)+len(extra)and all(r['status']=='READ'for r in local_rows),
        historical_loaded_code_identity_for_previously_unpinned_files_proved=False,
        no_container_started_model_import_device_service_health_process_work=True,overall_complete=False))
    print(json.dumps(dict(source_files_read=sum(r['status']=='READ'for r in local_rows),
        all_six_controls_match=controls_ok,errors=[{k:r[k]for k in ['path','status']}for r in local_rows if r['status']!='READ'],
        container_process_started=False,overall_complete=False)))


if __name__=='__main__':main()
