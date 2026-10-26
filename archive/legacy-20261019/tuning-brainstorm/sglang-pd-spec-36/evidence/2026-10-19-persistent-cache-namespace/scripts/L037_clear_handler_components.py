"""Controller-only patch plus original scheduler clear handler, bounded CPU composition."""
from L037_candidate_components import *
import msgspec
ns.update(msgspec=msgspec,logging=logging)
load('python/sglang/srt/managers/io_struct.py',classes=['BaseReq','ClearHiCacheReqInput','ClearHiCacheReqOutput'])
load('python/sglang/srt/managers/scheduler.py',methods={'Scheduler':['clear_hicache_storage_wrapped']})
BOUNDARIES=BOUNDARIES+['OriginalScheduler clearhandler andrealmsgspec input/output/msgpack roundtrip; no HTTP/processIPC or fullScheduler. Controller-onlypatch alreadyactive to isolate upperreturn loss.']
