#!/usr/bin/env python3
"""Reversible exact-source input-token cache adapter; offline installation only."""
import argparse,hashlib,json
from pathlib import Path
SOURCE=Path('/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/core/kv_cache_utils.py')
EXPECTED='6add6f1d60634b0819833675d86be4adf00c13fe5d0a1605074353b55d3e2d39'
APPEND='\n# FOUNDRY_INITIAL_PREFIX_HASH_MEMO: only immutable initial text hashes.\nimport os as _foundry_hash_os\nif _foundry_hash_os.getenv("EXTREME_INITIAL_PREFIX_HASH_CACHE") == "1":\n    import sys as _foundry_hash_sys\n    if "/data/wio/Inference_Foundry" not in _foundry_hash_sys.path:\n        _foundry_hash_sys.path.insert(0, "/data/wio/Inference_Foundry")\n    from experiments.prefix_hash_cache.cache import InitialPrefixHashMemo as _FoundryHashMemo\n    _foundry_original_hasher_factory = get_request_block_hasher\n    def _foundry_hasher_factory(hash_block_size, caching_hash_fn):\n        original = _foundry_original_hasher_factory(hash_block_size, caching_hash_fn)\n        logger.info("Foundry initial prefix hash cache active block_size=%s", hash_block_size)\n        seed = NONE_HASH\n        mode_path = _foundry_hash_os.getenv("EXTREME_PREFIX_HASH_MODE_FILE")\n        if mode_path:\n            from experiments.prefix_hash_cache.router import InitialHashRouter\n            return InitialHashRouter(original, mode_path, valid=lambda: NONE_HASH == seed)\n        return _FoundryHashMemo(original, valid=lambda: NONE_HASH == seed)\n    get_request_block_hasher = _foundry_hasher_factory\n'

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['check','install','restore']);p.add_argument('--state-dir');p.add_argument('--record',required=True);p.add_argument('--offline-confirmed',action='store_true');a=p.parse_args()
 raw=SOURCE.read_bytes()
 if a.action in ('check','install'):
  assert sha(raw)==EXPECTED,'source drift'
  candidate=raw+APPEND.encode();compile(candidate,str(SOURCE),'exec')
  record={'source':str(SOURCE),'original_sha256':sha(raw),'candidate_sha256':sha(candidate),'action':a.action}
  if a.action=='install':
   assert a.offline_confirmed and a.state_dir
   state=Path(a.state_dir);state.mkdir(parents=True,exist_ok=False)
   (state/'base.py').write_bytes(raw);(state/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
   SOURCE.write_bytes(candidate)
 else:
  assert a.offline_confirmed and a.state_dir
  state=Path(a.state_dir);record=json.loads((state/'manifest.json').read_text());original=(state/'base.py').read_bytes()
  assert sha(raw)==record['candidate_sha256'],'patched source drift'
  assert sha(original)==record['original_sha256'];SOURCE.write_bytes(original)
  record.update(action='restore',restored_sha256=sha(SOURCE.read_bytes()))
 Path(a.record).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
