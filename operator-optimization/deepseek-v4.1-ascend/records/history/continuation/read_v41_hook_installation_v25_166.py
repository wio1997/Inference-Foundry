from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,re,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
OUT=P/'analysis/native-PMU-complete-v18/V41-cache-hook-installation-and-startup-params-v25.json';assert not OUT.exists()
code=r'''from pathlib import Path
import json,ast,hashlib
p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_utils.py');b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='5fc49ebb9b64a2f14c4eb38b8c66d3daf1b7612961cf76392cf113d76f555aab';s=b.decode();ls=s.splitlines();tree=ast.parse(s)
terms=['_ascend_get_packed_kv_cache_groups','_ascend_pool_bytes_per_block','_ascend_get_kv_cache_config_from_groups'];rows=[]
for n in ast.walk(tree):
 if isinstance(n,(ast.Assign,ast.Expr))and any(t in ast.get_source_segment(s,n)for t in terms):
  scopes=[dict(name=f.name,start=f.lineno,end=f.end_lineno)for f in ast.walk(tree)if isinstance(f,ast.FunctionDef)and f.lineno<=n.lineno<=f.end_lineno]
  rows.append(dict(start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(max(0,n.lineno-3),min(len(ls),n.end_lineno+2))],enclosing_functions=scopes))
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),hook_assignment_or_call_contexts=rows,no_model_or_torch_import=True)))'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=25);assert r.returncode==0,r.stderr
p=P/'runtime/logs/service.log';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='a02e582afaf94f0911b0bfb62ddac9028f2ac25936e4776373fc0ebfd2e1c61e';fields={}
patterns={'max_num_seqs':r'''['"]?max_num_seqs['"]?\s*[=:]\s*(\d+)''','kv_transfer_none':r'''['"]?kv_transfer_config['"]?\s*[=:]\s*(None)''','disable_hybrid':r'''['"]?disable_hybrid_kv_cache_manager['"]?\s*[=:]\s*(True|False)''','num_gpu_blocks_override':r'''['"]?num_gpu_blocks_override['"]?\s*[=:]\s*(None|\d+)''','kv_cache_layout':r'''['"]?kv_cache_layout['"]?\s*[=:]\s*['"]?([A-Z]+)''','cpu_offload_gb':r'''['"]?cpu_offload_gb['"]?\s*[=:]\s*([\d.]+)'''}
for k,pattern in patterns.items():fields[k]=sorted(set(re.findall(pattern,b.decode(errors='replace'))))
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),installed_source=json.loads(r.stdout),saved_startup_whitelisted_fields=fields,log_path=str(p),log_sha256=hashlib.sha256(b).hexdigest(),absent_field_means='Not found bythiswhitelist;not false/default or a runtimeobject observation.',no_model_import_or_device_or_service_work=True,overall_complete=False)
bb=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(bb)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(bb).hexdigest()),ensure_ascii=False))
