"""Reclaim inactive allocator cache on P under device pressure; preserve all live state."""
import json,time
_calls=0
def reserve_runtime_headroom(npu,*,role,mode,tokens,minimum_free_bytes=2*1024**3):
    global _calls
    if role!='prefill':raise RuntimeError('Runtime-headroom candidate is P-only')
    started=time.monotonic();_calls+=1
    free,total=npu.mem_get_info();allocated=npu.memory_allocated();reserved=npu.memory_reserved()
    before=dict(device=npu.current_device(),free_bytes=int(free),total_bytes=int(total),allocated_bytes=int(allocated),reserved_bytes=int(reserved))
    reclaimed=False
    if free<minimum_free_bytes and reserved>allocated:
        npu.empty_cache();reclaimed=True
    after_free,after_total=npu.mem_get_info() if reclaimed else (free,total)
    record=dict(event='GLM53_RUNTIME_HEADROOM_G29',call=_calls,role=role,mode=mode,tokens=tokens,threshold_bytes=minimum_free_bytes,before=before,reclaim_called=reclaimed,after_free_bytes=int(after_free),after_reserved_bytes=int(npu.memory_reserved()) if reclaimed else int(reserved),elapsed_ms=1000*(time.monotonic()-started))
    if reclaimed or _calls<=4:print(json.dumps(record),flush=True)
    return record
