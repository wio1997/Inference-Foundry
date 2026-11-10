#!/usr/bin/env python3
"""Read saved draft-output policy scalars only; never execute the launch."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
REMOTE=r'''
from pathlib import Path
from hashlib import sha256
import json,shlex
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/scripts/serve.sh');raw=p.read_bytes();tokens=shlex.split(raw.decode().replace('\\\n',' '),comments=True)
def option(key):
 for i,t in enumerate(tokens):
  if t==key:return json.loads(tokens[i+1])
  if t.startswith(key+'='):return json.loads(t.split('=',1)[1])
 return {}
a=option('--additional-config');s=option('--speculative-config');allow=['method','num_speculative_tokens','enforce_eager','draft_sample_method','rejection_sample_method','parallel_drafting','use_heterogeneous_vocab','use_fp64_gumbel']
spec={k:s[k] for k in allow if k in s};assert all(isinstance(v,(str,int,float,bool,type(None))) for v in spec.values())
dyn=a.get('dynamic_spec_config',{});fine=a.get('finegrained_tp_config',{});assert isinstance(dyn,dict) and isinstance(fine,dict)
print(json.dumps(dict(path=str(p),sha256=sha256(raw).hexdigest(),speculative_selected=spec,speculative_absent=sorted(set(allow)-set(spec)),additional_selected={k:a[k] for k in ['enable_dsa_cp','enable_flashcomm1'] if k in a},dynamic_spec_config_present='dynamic_spec_config' in a,dynamic_method=dyn.get('method'),finegrained_lmhead_tp=fine.get('lmhead_tensor_parallel_size'),hf_override_present='--hf-overrides' in tokens,scope='Saved script whitelisted scalar policy only, not loaded config/request population; no args/env dump, execution/framework/device import, request or remote write.')))
'''
def main():
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=70)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout);assert d['sha256']=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba';d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();Path('audit/execution-search/draft-output-config.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(d))
if __name__=='__main__':main()
