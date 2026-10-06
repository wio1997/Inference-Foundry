
# Run253 diagnostic only: globally-idle A/B selection; not the performance patch.
import mmap as _glm_event_mmap
import os as _glm_event_os
import json as _glm_event_json
from pathlib import Path as _glm_event_Path
_glm_event_root = _glm_event_Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0253')
_glm_event_fd = _glm_event_os.open(_glm_event_root/'event_mode.bin', _glm_event_os.O_RDONLY)
_glm_event_mode = _glm_event_mmap.mmap(_glm_event_fd, 1, access=_glm_event_mmap.ACCESS_READ)
_glm_event_os.close(_glm_event_fd)
_glm_event_candidate = maybe_record_moe_event
_glm_event_last_mode = None

def maybe_record_moe_event():
    global _glm_event_last_mode
    mode = _glm_event_mode[0]
    assert mode in (0,1,2,3)
    configured = get_ascend_config().multistream_overlap_shared_expert
    result = _glm_event_candidate() if mode%2 else torch.npu.current_stream().record_event()
    if mode != _glm_event_last_mode and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        scope = _glm_event_os.environ.get('GLM_EVENT_WITNESS_SCOPE','model')
        row = dict(rank=rank,pid=_glm_event_os.getpid(),mode=mode,configured_overlap=configured,returned_none=result is None,
                   candidate_utils_sha256='c76223cde8fac1593cd3dc2874b55c1cd2dad5a5f40d24389b391ca8983c3b30')
        (_glm_event_root/'witnesses').mkdir(exist_ok=True)
        (_glm_event_root/'witnesses'/('%s_mode%d_rank%d.json'%(scope,mode,rank))).write_text(_glm_event_json.dumps(row)+'\n')
        _glm_event_last_mode = mode
    return result
