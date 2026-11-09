#!/usr/bin/env python3
"""Extract scalar guard fields, never raw launch/log lines or credentials."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from read_inactive_dp_profile_probe import ROOT

REMOTE = r'''
from pathlib import Path
from hashlib import sha256
import json,re,shlex,sys
root=Path(sys.argv[1]);assert root.is_relative_to('/data/tiankuan/wio')
p=root/'service.log';st=p.stat();raw=p.read_bytes()
keys=['enable_prefix_caching','prefix_caching','kv_transfer_config','ec_transfer_config','cpu_offload_gb','offload_group_size','kv_offloading_size','async_scheduling','max_concurrent_batches','enable_chunked_prefill','use_v2_model_runner','block_size']
rows=[]
for n,line in enumerate(raw.decode(errors='replace').splitlines(),1):
 if not any(t in line for t in ['Initializing a V1','non-default args','cache config','CacheConfig(','VllmConfig(','offload_config']):continue
 selected={}
 for k in keys:
  vals=re.findall(r'\b'+k+r"(?:['\"])?(?:=|:\s*)(None|True|False|[0-9.]+)",line)
  if vals:selected[k]=vals
 if selected:rows.append(dict(line=n,selected_scalars=selected))
assert (st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
s=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/scripts/serve.sh');sr=s.read_bytes();tokens=shlex.split(sr.decode().replace('\\\n',' '),comments=True)
allow={'--offload-backend','--offload-group-size','--offload-num-in-group','--offload-prefetch-step','--cpu-offload-gb','--kv-offloading-size','--kv-offloading-backend','--kv-transfer-config','--ec-transfer-config','--enable-prefix-caching','--no-enable-prefix-caching','--prefix-caching-hash-algo','--kv-sharing-fast-prefill'}
args={}
for i,t in enumerate(tokens):
 k=t.split('=',1)[0]
 if k in allow:args[k]=t.split('=',1)[1] if '=' in t else (tokens[i+1] if i+1<len(tokens) and not tokens[i+1].startswith('--') else True)
print(json.dumps(dict(completed_profile_startup=dict(path=str(p),sha256=sha256(raw).hexdigest(),bytes=len(raw),mtime_ns=st.st_mtime_ns,scalar_records=rows),saved_serve=dict(path=str(s),sha256=sha256(sr).hexdigest(),selected_options=args,absent_whitelisted_options=sorted(allow-set(args))),scope='Completed independent profile startup + unchanged saved serving script/default source, not live object introspection',new_profile_collection=0,service_requests=0,new_NPU_work=0,remote_writes=[])))
'''


def main():
    r = subprocess.run(['/Users/wio/.local/bin/baai-ssh', '171', '--',
                        'python3', '-I', '-S', '-c', REMOTE, ROOT],
                       capture_output=True, text=True, timeout=100)
    if r.returncode:
        raise RuntimeError(r.stderr)
    x = json.loads(r.stdout)
    x['reader_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    Path('audit/execution-search/prefix-retirement-config-record.json').write_text(
        json.dumps(x, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(x))


if __name__ == '__main__':
    main()
