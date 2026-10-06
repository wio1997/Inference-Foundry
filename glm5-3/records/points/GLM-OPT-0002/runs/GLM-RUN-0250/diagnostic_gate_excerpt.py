# Excerpt from actual frozen shim; SHA256 310c47b255efa25e442523e96589e7e0f8cd8b78020a0ba7d9453163a6403fe6
# Diagnostic-only same-process selector. Mode changes only at controller idle.
import mmap as _glm53_mmap, os as _glm53_os, json as _glm53_json
from pathlib import Path as _glm53_Path
_glm53_root = _glm53_Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0250')
_glm53_fd = _glm53_os.open(_glm53_root / 'gather_mode.bin', _glm53_os.O_RDONLY)
_glm53_mode = _glm53_mmap.mmap(_glm53_fd, 1, access=_glm53_mmap.ACCESS_READ)
_glm53_os.close(_glm53_fd)
_glm53_last_mode = None
_glm53_stock_finalize = PrepareAndFinalizeWithAll2All.finalize

def _glm53_matched_finalize(self, hidden_states, reduce_results, padded_hidden_states_shape=None):
    global _glm53_last_mode
    mode = _glm53_mode[0]
    assert mode in (0,1,2,3), 'invalid diagnostic mode'
    method = _glm53_direct_finalize if mode % 2 else _glm53_stock_finalize
    if mode != _glm53_last_mode:
        assert self.tp_size == 16 and not self.replace_allreduce
        assert padded_hidden_states_shape is not None and hidden_states.shape[-1] == 6144
        equal = None
        if mode % 2:
            before = hidden_states.clone()
            expected = _glm53_stock_finalize(self, hidden_states, reduce_results, padded_hidden_states_shape)
            result = _glm53_direct_finalize(self, hidden_states, reduce_results, padded_hidden_states_shape)
            equal = bool(torch.equal(expected, result))
            assert equal and torch.equal(before, hidden_states), 'actual HCCL hidden mismatch'
            assert result.data_ptr() != hidden_states.data_ptr(), 'output aliases input'
        else:
            result = method(self, hidden_states, reduce_results, padded_hidden_states_shape)
        rank = dist.get_rank()
        row = dict(rank=rank, pid=_glm53_os.getpid(), mode=mode,
                   actual_hidden_bitwise_equal=equal, shape=list(hidden_states.shape),
                   padded_shape=list(padded_hidden_states_shape), dtype=str(hidden_states.dtype),
                   local_bytes=hidden_states.numel()*hidden_states.element_size(),
                   input_preserved=True if mode%2 else None,
                   artifact='candidate:'+'26e52eef2714f9e3246d001b9ffc4e05d453a79cf95c3c38c0a2ba860498f3f9')
        (_glm53_root/'witnesses'/('mode%d_rank%d.json'%(mode,rank))).write_text(_glm53_json.dumps(row)+'\n')
        _glm53_last_mode = mode
        return result
    return method(self, hidden_states, reduce_results, padded_hidden_states_shape)

PrepareAndFinalizeWithAll2All.finalize = _glm53_matched_finalize
