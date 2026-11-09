#!/usr/bin/env python3
"""Read-only source pins and whitelisted saved launch options for loop021."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from read_metadata_reader_identity import BASE, FROZEN

OUT = Path('audit/execution-search')
NAMES = [
    'vllm/vllm/v1/core/sched/scheduler.py',
    'vllm/vllm/v1/core/sched/async_scheduler.py',
    'vllm/vllm/v1/core/kv_cache_manager.py',
    'vllm/vllm/v1/core/single_type_kv_cache_manager.py',
    'vllm/vllm/v1/core/kv_cache_coordinator.py',
    'vllm/vllm/v1/core/block_pool.py',
    'vllm/vllm/v1/core/kv_cache_utils.py',
    'vllm/vllm/v1/core/sched/output.py',
    'vllm/vllm/v1/engine/core.py',
    'vllm/vllm/config/cache.py',
    'vllm/vllm/config/scheduler.py',
    'vllm/vllm/config/offload.py',
    'vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
    'vllm-ascend/vllm_ascend/worker/worker.py',
    'vllm-ascend/vllm_ascend/ascend_config.py',
    'vllm-ascend/vllm_ascend/model_executor/offloader/__init__.py',
    'vllm-ascend/vllm_ascend/worker/block_table.py',
    'vllm-ascend/vllm_ascend/models/deepseek_v41/cache_config.py',
]
REMOTE = r'''
from pathlib import Path
from hashlib import sha256
import json,subprocess,sys,shlex
spec=json.loads(sys.argv[1]);name='wio-vllm-v41-repro171-20261102'
item,=json.loads(subprocess.check_output(['docker','inspect',name]))
assert item['State']['Running'] and item['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner="""from pathlib import Path
from hashlib import sha256
import json,sys
out=[]
for r in json.loads(sys.argv[1]):
 p=Path(r['runtime_path'])
 if not p.is_file():out.append(dict(**r,missing=True));continue
 raw=p.read_bytes();h=sha256(raw).hexdigest()
 out.append(dict(**r,runtime_sha256=h,matches_local=h==r.get('local_sha256'),content=None if h==r.get('local_sha256') else raw.decode()))
print(json.dumps(out))
"""
sources=json.loads(subprocess.check_output(['docker','exec',name,'python3','-I','-S','-c',inner,json.dumps(spec)]))
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/scripts/serve.sh')
assert p.is_relative_to('/data/tiankuan/wio');raw=p.read_bytes()
tokens=shlex.split(raw.decode().replace('\\\n',' '),comments=True)
allow={'--kv-transfer-config','--ec-transfer-config','--additional-config','--engram-config','--speculative-config','--compilation-config','--enable-prefix-caching','--no-enable-prefix-caching','--prefix-caching-hash-algo','--cpu-offload-gb','--kv-offloading-size','--kv-offloading-backend','--block-size','--async-scheduling','--no-async-scheduling','--max-num-seqs','--max-num-batched-tokens','--max-model-len','--max-logprobs','--kv-cache-dtype','--num-lookahead-slots','--enable-chunked-prefill','--no-enable-chunked-prefill','--disable-hybrid-kv-cache-manager','--gpu-memory-utilization','--quantization','--tensor-parallel-size','--data-parallel-size'}
selected={}
for i,t in enumerate(tokens):
 key=t.split('=',1)[0]
 if key not in allow:continue
 selected[key]=t.split('=',1)[1] if '=' in t else (tokens[i+1] if i+1<len(tokens) and not tokens[i+1].startswith('--') else True)
print(json.dumps(dict(container_id=item['Id'],image=item['Image'],sources=sources,saved_launch=dict(path=str(p),sha256=sha256(raw).hexdigest(),selected_options=selected,scope='Saved launch script only; defaults/live config require separate source/record binding'),new_NPU_work=0,service_requests=0,new_profile_collection=0,remote_writes=[])))
'''


def main():
    old = []
    for p in OUT.glob('*identity.json'):
        try:
            x = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        old.extend(x.get('sources', []))
    spec = []
    for name in NAMES:
        runtime = '/vllm-workspace/' + name
        known = next((r.get('local_path') for r in old
                      if r.get('runtime_path') == runtime and r.get('local_path')
                      and Path(r['local_path']).is_file()), None)
        local = Path(known) if known else FROZEN / name
        data = local.read_bytes() if local.is_file() else None
        spec.append(dict(runtime_path=runtime, local_path=str(local),
                         local_sha256=sha256(data).hexdigest() if data else None))
    r = subprocess.run(['/Users/wio/.local/bin/baai-ssh', '171', '--',
                        'python3', '-I', '-S', '-c', REMOTE, json.dumps(spec)],
                       capture_output=True, text=True, timeout=100)
    if r.returncode:
        raise RuntimeError(r.stderr)
    x = json.loads(r.stdout)
    for row in x['sources']:
        content = row.pop('content', None)
        if content is not None:
            name = row['runtime_path'].removeprefix('/vllm-workspace/')
            p = OUT / 'prefix-retirement-runtime-source' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            assert sha256(p.read_bytes()).hexdigest() == row['runtime_sha256']
            row['local_path'] = str(p)
    x['reader_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / 'prefix-retirement-execution-identity.json').write_text(
        json.dumps(x, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(sources=len(x['sources']),
                          matched=sum(r.get('matches_local', False) for r in x['sources']),
                          missing=[r['runtime_path'] for r in x['sources'] if r.get('missing')],
                          saved_launch=x['saved_launch'], new_NPU_work=0, remote_writes=[])))


if __name__ == '__main__':
    main()
