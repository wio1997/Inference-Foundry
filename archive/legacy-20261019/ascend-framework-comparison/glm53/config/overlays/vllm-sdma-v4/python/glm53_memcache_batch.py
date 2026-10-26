"""Bound each native MemCache get lease to a finite ordered key batch."""
from glm53_persistent_audit import persist
import hashlib
import json
import logging
import time

logger=logging.getLogger('vllm.glm53_memcache_batch')


def batched_get_into_layers(store, keys, addresses, sizes, direction, batch_blocks):
    if not isinstance(batch_blocks,int) or isinstance(batch_blocks,bool) or not 1<=batch_blocks<=128:
        raise ValueError('GLM53 MemCache get batch size must be an integer in1..128')
    if not keys or len(keys)!=len(addresses) or len(keys)!=len(sizes):
        raise ValueError('GLM53 MemCache get requires aligned nonempty key/address/size rows')
    for pointers,lengths in zip(addresses,sizes):
        if not pointers or len(pointers)!=len(lengths) or any(type(n) is not int or n<=0 for n in lengths):
            raise ValueError('GLM53 MemCache get requires aligned positive layer sizes')
    results=[]
    for start in range(0,len(keys),batch_blocks):
        stop=min(start+batch_blocks,len(keys));begin=time.monotonic_ns()
        chunk_keys=keys[start:stop];chunk_sizes=sizes[start:stop]
        native=store.batch_get_into_layers(chunk_keys,addresses[start:stop],chunk_sizes,direction)
        event={'event':'native_get_batch_complete','begin_monotonic_ns':begin,'terminal_monotonic_ns':time.monotonic_ns(),
               'first_key_index':start,'key_count':stop-start,'total_key_count':len(keys),
               'key_list_sha256':hashlib.sha256(json.dumps(chunk_keys,separators=(',',':')).encode()).hexdigest(),
               'payload_bytes_requested':sum(sum(row) for row in chunk_sizes),'direction':direction,'native_results':native,
               'batch_limit_blocks':batch_blocks,'retry_attempts':0,'fallback':False}
        persist('MC_GET', event)
        logger.info('GLM53_MC_GET='+json.dumps(event,sort_keys=True))
        if native is None or len(native)!=stop-start:
            raise RuntimeError('Native MemCache get returned a mismatched result count; no retry')
        if any(int(status)!=0 for status in native):
            raise RuntimeError('Native MemCache get batch failed; no retry or payload fallback')
        results.extend(native)
    return results
