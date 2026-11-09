#!/usr/bin/env python3
"""Reuse original ID replay with the actual selected DeepseekV41 wrapper."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from replay_input_renderer_ids import INNER as GENERIC_INNER,REMOTE

MARKER="cached=env['get_cached_tokenizer'](hf);pooled=env['maybe_make_thread_pool'](cached,2)"
assert GENERIC_INNER.count(MARKER)==1
INNER=GENERIC_INNER.replace(MARKER,"""exec(original('tokenizers/deepseek_v41.py',[(None,'get_deepseek_v41_tokenizer')]),env)
wrapped=env['get_deepseek_v41_tokenizer'](hf)
cached=env['get_cached_tokenizer'](wrapped);pooled=env['maybe_make_thread_pool'](cached,2)
assert 'DSV41' in type(pooled).__name__
""").replace('Original renderer methods plus installed generic HF backend constructed from fixed tokenizer.json/config, not AutoTokenizer model-class resolution;',
 'Original actual get_deepseek_v41_tokenizer factory, original cached/copy pool and renderer methods over installed HF backend constructor/config. Does not run TokenizersBackend.from_pretrained bootstrap or chat formatter;')


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
    x=json.loads(r.stdout);x['probe_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();x['generic_probe_sha256']=sha256(Path('tools/replay_input_renderer_ids.py').read_bytes()).hexdigest()
    reference=json.loads(Path('audit/execution-search/input-renderer-ID-replay.json').read_text())
    assert [r['IDs_sha256'] for r in x['rows']]==[r['IDs_sha256'] for r in reference['rows']]
    x['same_generic_replay_ID_hashes']=True
    Path('audit/execution-search/actual-tokenizer-scalar-replay.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=x['status'],pooled_type=x['pooled_type'],same_generic_ID_hashes=True,source_pins=x['source_pins'],torch_npu_imported=x['torch_npu_imported'],scope=x['scope']),ensure_ascii=False))


if __name__=='__main__':main()
