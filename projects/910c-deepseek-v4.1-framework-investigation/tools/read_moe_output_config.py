#!/usr/bin/env python3
"""Read only whitelisted shared-expert launch scalars and saved-script SHA."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
REMOTE=r'''
from pathlib import Path
from hashlib import sha256
import json,shlex
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/scripts/serve.sh');b=p.read_bytes();tokens=shlex.split(b.decode().replace('\\\n',' '),comments=True)
for i,t in enumerate(tokens):
 if t=='--additional-config':d=json.loads(tokens[i+1]);break
else:raise AssertionError('additional-config missing')
keys=['enable_fused_mc2','enable_dsa_cp','enable_flashcomm1','enable_shared_expert_dp','multistream_overlap_shared_expert','mix_placement','finegrained_tp_config']
selected={k:d[k] for k in keys if k in d}
assert all(isinstance(v,(int,bool)) for k,v in selected.items() if k!='finegrained_tp_config')
if 'finegrained_tp_config' in selected:selected['finegrained_tp_config']={k:v for k,v in selected['finegrained_tp_config'].items() if k in ['mlp_tensor_parallel_size','oproj_tensor_parallel_size','embedding_tensor_parallel_size','lmhead_tensor_parallel_size']}
print(json.dumps(dict(path=str(p),sha256=sha256(b).hexdigest(),selected=selected,absent=sorted(set(keys)-set(selected)),scope='Saved script only; whitelisted scalar JSON, no execution, args/environment dump, device import, request, remote write or loaded-config proof.')))
'''
def main():
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=70)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout);assert d['sha256']=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba';d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
 Path('audit/execution-search/moe-output-config.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(d['selected']))
if __name__=='__main__':main()
