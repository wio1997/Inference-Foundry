"""Observe one unchanged full-batch native SDK read, without retry or slicing."""
import hashlib,json,logging,time
from glm53_persistent_audit import persist
logger=logging.getLogger('vllm.glm53_memcache_native')
def native_get_into_layers(store,keys,addresses,sizes,direction):
    if not keys or len(keys)!=len(addresses) or len(keys)!=len(sizes):
        raise ValueError('Aligned nonempty native MemCache key/address/size rows required')
    for pointers,lengths in zip(addresses,sizes):
        if not pointers or len(pointers)!=len(lengths) or any(type(n) is not int or n<=0 for n in lengths):
            raise ValueError('Aligned positive native layer sizes required')
    begin=time.monotonic_ns()
    result=store.batch_get_into_layers(keys,addresses,sizes,direction)
    event={'event':'native_get_batch_complete','begin_monotonic_ns':begin,'terminal_monotonic_ns':time.monotonic_ns(),
           'first_key_index':0,'key_count':len(keys),'total_key_count':len(keys),
           'key_list_sha256':hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),
           'payload_bytes_requested':sum(sum(row) for row in sizes),'direction':direction,'native_results':result,
           'batch_limit_blocks':None,'native_call_count':1,'retry_attempts':0,'fallback':False}
    persist('MC_GET',event)
    logger.info('GLM53_MC_GET='+json.dumps(event,sort_keys=True))
    if result is None or len(result)!=len(keys):
        raise RuntimeError('Native full-batch MemCache result shape mismatch; no retry')
    if any(int(value)!=0 for value in result):
        raise RuntimeError('Native full-batch MemCache read failed; no retry/fallback')
    return result
