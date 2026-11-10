#!/usr/bin/env python3
"""Read the actual model registry and DSpark flag normalization dependencies."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
    old=json.loads((ROOT/'whole-forward-dependencies.json').read_text());model=next(s for s in old['sources'] if s['runtime_path'].endswith('/model.py'));frozen=Path(model['local_path']).parents[4]
    names=['vllm-ascend/vllm_ascend/models/__init__.py','vllm-ascend/vllm_ascend/patch/platform/patch_speculative_config.py','vllm/vllm/config/speculative.py','vllm/vllm/transformers_utils/configs/deepseek_v41.py'];spec=[]
    for n in names:
        p=frozen/n
        if not p.exists() and n.endswith('patch_speculative_config.py'):p=ROOT/'dspark-runtime-source/patch_speculative_config.py'
        if not p.exists() and n.endswith('configs/deepseek_v41.py'):p=ROOT/'startup-warmup-runtime-source/vllm/vllm/transformers_utils/configs/deepseek_v41.py'
        spec.append(dict(runtime_path='/vllm-workspace/'+n,local_path=str(p) if p.exists() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.exists() else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=90)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for s in d['sources']:
        body=s.pop('content',None)
        if body is not None:
            p=ROOT/'whole-forward-binding-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Additional current identity/body acquisition only; no import or runtime-object dump.')
    (ROOT/'whole-forward-binding-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_NPU_work=0)))
if __name__=='__main__':main()
