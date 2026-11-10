#!/usr/bin/env python3
"""Pin selected default-policy source and allowlisted saved launch flags."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
FLAGS=r'''
from pathlib import Path
from hashlib import sha256
import json,shlex
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/scripts/serve.sh');b=p.read_bytes();ts=shlex.split(b.decode().replace('\\\n',' '),comments=True)
allow={'--additional-config','--enable-dbo','--enable-eplb','--kv-sharing-fast-prefill','--mamba-cache-mode','--kv-transfer-config','--ec-transfer-config'};out={}
for i,t in enumerate(ts):
 k=t.split('=',1)[0]
 if k not in allow:continue
 if k in {'--kv-transfer-config','--ec-transfer-config'}:out[k]='PRESENT_CONTENT_NOT_RETRIEVED';continue
 v=t.split('=',1)[1] if '=' in t else ts[i+1] if i+1<len(ts) and not ts[i+1].startswith('--') else True
 if k=='--additional-config':
  d=json.loads(v);keys=['multistream_engram_overlap','multistream_dsv4_dsa_overlap','enable_dsa_cp','eplb_config','sparse_kv_offload_config'];v={n:(d[n] if isinstance(d[n],(bool,int)) else 'NONBOOLEAN_NOT_EXPANDED') for n in keys if n in d};v['absent_selected_suboptions']=[n for n in keys if n not in d]
  for n,sub in [('eplb_config','dynamic_eplb'),('sparse_kv_offload_config','enabled')]:
   if isinstance(d.get(n),dict) and isinstance(d[n].get(sub),(bool,int)):v[n+'.'+sub]=d[n][sub]
 else:v=v if isinstance(v,bool) or v in {'all','align',True,False} else 'NONLITERAL_NOT_EXPANDED'
 out[k]=v
print(json.dumps(dict(path=str(p),sha256=sha256(b).hexdigest(),selected_options=out,absent_options=sorted(allow-set(out)),qualification='Saved literals only; no live object/env expansion or secret-bearing connector values.')))
'''
def main():
    old=json.loads((ROOT/'whole-forward-dispatch.json').read_text());prior={s['runtime_path']:s for s in old['sources']};prior.update({s['runtime_path']:s for s in json.loads((ROOT/'embedding-static-source-review.json').read_text())['sources']});spec=[]
    for runtime in ['/vllm-workspace/vllm-ascend/vllm_ascend/ascend_config.py','/vllm-workspace/vllm-ascend/vllm_ascend/device/hardware_profile.py','/vllm-workspace/vllm-ascend/vllm_ascend/attention/utils.py','/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py']:
        s=prior.get(runtime);p=Path(s['local_path']) if s else None;spec.append(dict(runtime_path=runtime,local_path=str(p) if p else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for s in d['sources']:
        body=s.pop('content',None)
        if body is not None:
            p=ROOT/'input-flag-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',FLAGS],capture_output=True,text=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr)
    launch=json.loads(r.stdout);assert launch['sha256']==json.loads((ROOT/'whole-forward-frozen-config.json').read_text())['saved_serve']['sha256']
    d.update(saved_launch=launch,reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Current source default policy plus frozen saved launch only; not loaded object dump. Original trace kernel population is independently pinned.')
    (ROOT/'input-frozen-flags.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),saved_launch=launch,new_NPU_work=0,remote_writes=[])))
if __name__=='__main__':main()
