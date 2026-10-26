"""Run-owned same-service initial-hash crossover; no continuation switching."""
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from .cache import InitialPrefixHashMemo

class InitialHashRouter:
    def __init__(self, original, mode_path, *, valid=None, report_every=48):
        self.original=original
        self.memo=InitialPrefixHashMemo(original,valid=valid)
        self.mode_path=Path(mode_path)
        self.report_path=self.mode_path.with_name('hash_routing.jsonl')
        self.report_every=report_every
        self.rows={}
        self.lock=Lock()

    def __call__(self, request):
        # Read the external control only at fresh text request initialization.
        if (request.block_hashes or request.prompt_token_ids is None
                or request.num_tokens!=request.num_prompt_tokens
                or request.prompt_embeds is not None or request.mm_features
                or request.lora_request or request.cache_salt):
            return self.original(request)
        mode=json.loads(self.mode_path.read_text())
        if (set(mode)!={'phase','enabled','version'}
                or type(mode['enabled']) is not bool
                or type(mode['version']) is not int
                or mode['version']<1 or not isinstance(mode['phase'],str)):
            raise RuntimeError('invalid hash cache crossover mode')
        key=(mode['version'],mode['phase'],mode['enabled'])
        start=perf_counter()
        result=self.memo(request) if mode['enabled'] else self.original(request)
        elapsed=perf_counter()-start
        with self.lock:
            if any(v==mode['version'] and k!=key for k in self.rows for v in [k[0]]):
                raise RuntimeError('reused mode version with different route')
            row=self.rows.setdefault(key,dict(mode,initial_requests=0,hash_call_s=0.0))
            row['initial_requests']+=1;row['hash_call_s']+=elapsed
            if row['initial_requests']%self.report_every==0:
                with self.report_path.open('a') as f:
                    f.write(json.dumps(dict(row,cache_entries=len(self.memo.rows),cache_estimated_bytes=self.memo.bytes,memo_hits=self.memo.hits,memo_misses=self.memo.misses))+'\n')
        return result
