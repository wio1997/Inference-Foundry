from pathlib import Path
import json,socket,subprocess,hashlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
code="from pathlib import Path;import hashlib,json;p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py');print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())))"
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
source=json.loads(r.stdout)
b=(P/'analysis/prepare-field-origins-v12.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f';x=json.loads(b)
groups=[g for g in x['source_groups'] if 'dsa_v41_cp.py(131)' in g['first_project_location'] or 'dsa_v41_cp.py(140)' in g['first_project_location']]
result=dict(installed_source=source,matches_cached_source=source['sha256']=='039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e',CP_source_groups=groups,no_target_source_body_read=True,no_inference_or_service_operation=True,overall_complete=False)
out=P/'analysis/CP-metadata-shapes-and-source-pin-v12.json';assert not out.exists();payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
