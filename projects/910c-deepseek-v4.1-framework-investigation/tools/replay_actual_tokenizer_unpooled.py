#!/usr/bin/env python3
"""Actual DeepseekV4Renderer uses BaseRenderer, without HfRenderer's pool."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from replay_actual_tokenizer_scalar import INNER as POOLED_INNER
from replay_input_renderer_ids import REMOTE

MARKER="cached=env['get_cached_tokenizer'](wrapped);pooled=env['maybe_make_thread_pool'](cached,2)"
assert POOLED_INNER.count(MARKER)==1
INNER=POOLED_INNER.replace(MARKER,"cached=env['get_cached_tokenizer'](wrapped);pooled=cached\nassert not isinstance(pooled,env['ThreadSafeHFTokenizerMixin'])")
INNER=INNER.replace('native_deep_copies_in_pool=2','native_deep_copies_in_pool=0')
INNER=INNER.replace('Original actual get_deepseek_v41_tokenizer factory, original cached/copy pool and renderer methods over installed HF backend constructor/config.',
 'Original actual get_deepseek_v41_tokenizer factory and cached backend WITHOUT HfRenderer copy pool, matching selected DeepseekV4Renderer->BaseRenderer scalar path over installed HF backend constructor/config.')


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
    x=json.loads(r.stdout);x['probe_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();x['parent_probe_sha256']=sha256(Path('tools/replay_actual_tokenizer_scalar.py').read_bytes()).hexdigest()
    reference=json.loads(Path('audit/execution-search/input-renderer-ID-replay.json').read_text())
    assert [r['IDs_sha256'] for r in x['rows']]==[r['IDs_sha256'] for r in reference['rows']]
    assert x['native_deep_copies_in_pool']==0 and x['pooled_type']=='CachedDSV41TokenizersBackend'
    x['same_generic_replay_ID_hashes']=True
    Path('audit/execution-search/actual-tokenizer-unpooled-replay.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=x['status'],actual_backend_type=x['pooled_type'],pool_copies=0,same_generic_ID_hashes=True,torch_npu_imported=x['torch_npu_imported']),ensure_ascii=False))


if __name__=='__main__':main()
