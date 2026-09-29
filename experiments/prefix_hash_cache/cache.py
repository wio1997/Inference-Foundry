"""Off-path candidate: memoize initial plain-text hashes, never continuation.

One memo belongs to one fixed process/hash configuration. Hash seed, algorithm
or block-size changes require a new instance. No serving installation here.
"""
from collections import OrderedDict
from threading import Lock
import sys

class InitialPrefixHashMemo:
    def __init__(self, original, *, max_entries=128, max_bytes=128*1024*1024, valid=None):
        if max_entries < 0 or max_bytes < 0:raise ValueError('negative capacity')
        self.original=original
        self.valid=valid
        self.max_entries=max_entries
        self.max_bytes=max_bytes
        self.rows=OrderedDict()
        self.bytes=0
        self.lock=Lock()

    def __call__(self, request):
        # Request construction invokes the hasher before later fields exist.
        # Require precisely a fresh, plain-text prompt; fall back otherwise.
        if self.valid is not None and not self.valid():
            return self.original(request)
        if (request.block_hashes or request.prompt_embeds is not None
            or request.mm_features or request.lora_request or request.cache_salt
            or request.prompt_token_ids is None
            or request.num_tokens != request.num_prompt_tokens):
            return self.original(request)
        key=tuple(request.all_token_ids)
        with self.lock:
            row=self.rows.get(key)
            if row is not None:
                self.rows.move_to_end(key)
                return list(row[0])
        result=self.original(request)
        hashes=tuple(result)
        cost=(sys.getsizeof(key)+sum(sys.getsizeof(x) for x in key)
              +sys.getsizeof(hashes)+sum(sys.getsizeof(x) for x in hashes)+512)
        if not self.max_entries or cost>self.max_bytes:return result
        with self.lock:
            if key not in self.rows:
                self.rows[key]=(hashes,cost);self.bytes+=cost
                while len(self.rows)>self.max_entries or self.bytes>self.max_bytes:
                    _,(_,n)=self.rows.popitem(last=False);self.bytes-=n
            else:self.rows.move_to_end(key)
        return result
