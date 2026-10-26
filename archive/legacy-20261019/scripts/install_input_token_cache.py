#!/usr/bin/env python3
"""Reversible exact-source input-token cache adapter; offline installation only."""
import argparse,hashlib,json
from pathlib import Path
SOURCE=Path('/data/wio/vllm_ascend_26/framework/vllm/vllm/renderers/base.py')
EXPECTED='d3f748aafa3729fa0a27f29ed7e845e60460148a18c11c893ba02c638e9fd644'
APPEND = '\n# FOUNDRY_INPUT_TOKEN_CACHE: fixed-tokenizer text-only memoization.\nimport os as _foundry_cache_os\nif _foundry_cache_os.getenv("EXTREME_TEXT_TOKEN_CACHE") == "1":\n    import sys as _foundry_cache_sys\n    if "/data/wio/Inference_Foundry" not in _foundry_cache_sys.path:\n        _foundry_cache_sys.path.insert(0, "/data/wio/Inference_Foundry")\n    from serving.input_token_cache import cached_tokenize_prompt as _foundry_cached_encode\n    _foundry_uncached_encode = BaseRenderer._tokenize_prompt\n    BaseRenderer._foundry_uncached_tokenize_prompt = _foundry_uncached_encode\n    def _foundry_tokenize_prompt(self, prompt, params):\n        if type(self).__name__ != "DeepseekV4Renderer":\n            return _foundry_uncached_encode(self, prompt, params)\n        result = _foundry_cached_encode(_foundry_uncached_encode, self, prompt, params)\n        if not getattr(self, "_foundry_cache_logged", False):\n            self._foundry_cache_logged = True\n            logger.info("Foundry bounded input-token cache active for DeepseekV4Renderer")\n        return result\n    BaseRenderer._tokenize_prompt = _foundry_tokenize_prompt\n'

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
