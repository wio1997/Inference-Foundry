#!/usr/bin/env python3
"""Once-only exact-image preflight after owned v2 idle shutdown; no model engine."""
import ast,hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'stage_vllm_cache_batch_v3.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='b652e1773c8a68f8b80b3596d7714962d4d4822e65be8fecf6e12d639a5a520f'
s=importlib.util.spec_from_file_location('stage',prior);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
RUN=m.b.ROOT/'runs/20261017T0830-vllm-cache-batch-preflight-v3'
def main():
 stopped=json.loads((m.b.ROOT/'runs/20261017T0820-stop-vllm-correctness-v2/receipt.json').read_text())
 assert stopped['meta_stop']==0 and all(stopped['stops'].get(str(n))==0 for n in (30,31,43,45))
 staged=json.loads((m.b.ROOT/'runs/20261017T0810-stage-vllm-cache-batch-v3/receipt.json').read_text());assert all(v==0 for v in staged['nodes'].values())
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 r=m.preflight();(RUN/'preflight-30.json').write_text(json.dumps(r,indent=2)+'\n')
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'preflight_rc':r['rc'],'nodes':staged['nodes'],'model_requests':0},indent=2)+'\n');print('cachebatch native preflight',r['rc'],flush=True)
if __name__=='__main__':main()
