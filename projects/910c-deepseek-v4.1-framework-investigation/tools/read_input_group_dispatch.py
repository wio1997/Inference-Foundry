#!/usr/bin/env python3
"""Read the actual Ascend monkeypatch before interpreting upstream group code."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
    old=json.loads((ROOT/'input-resource-dependencies.json').read_text());prior={s['runtime_path']:s for s in json.loads((ROOT/'embedding-static-source-review.json').read_text())['sources']}
    inventory=json.loads((ROOT/'input-resource-callsite-inventory.json').read_text());allpaths={s['runtime_path']:s for s in inventory['inventory']}
    names=['/patch/platform/patch_kv_cache_utils.py','/patch/platform/__init__.py','/patch/__init__.py','/worker/worker.py','/models/deepseek_v41/cache_config.py','/models/deepseek_v41/compressor.py','/models/deepseek_v41/indexer.py','/models/layer/attention/layer.py']
    paths=['/vllm-workspace/vllm-ascend/vllm_ascend'+n for n in names];spec=[]
    for runtime in paths:
        assert runtime in allpaths
        p=Path(prior[runtime]['local_path']) if runtime in prior else None
        spec.append(dict(runtime_path=runtime,local_path=str(p) if p else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=120)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for s in d['sources']:
        assert s['runtime_sha256']==allpaths[s['runtime_path']]['sha256']
        body=s.pop('content',None)
        if body is not None:
            p=ROOT/'input-group-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),inventory_sha256=sha256((ROOT/'input-resource-callsite-inventory.json').read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Actual Ascend group patch/registration dependency source only; not initialized group population or native replacement bits.')
    (ROOT/'input-group-dispatch-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_NPU_work=0,remote_writes=[])))
if __name__=='__main__':main()
