# Excerpt from actual frozen shim; SHA256 f2efbec324fc95fb5bc3b5076e9676a22b14d050595cfb8a3ea76f632d11db27
# Diagnostic-only same-process selector. Mode changes only at controller idle.
import mmap as _glm53_mmap, os as _glm53_os, json as _glm53_json
from pathlib import Path as _glm53_Path
_glm53_root = _glm53_Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0251')
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
            before_cpu = hidden_states.detach().cpu()
            before_bits = before_cpu.contiguous().view(torch.uint8).clone()
            original_num_tokens = self.num_tokens
            try:
                self.num_tokens = padded_hidden_states_shape[0]
                expected = _glm53_stock_finalize(self, hidden_states, reduce_results, padded_hidden_states_shape)
                result = _glm53_direct_finalize(self, hidden_states, reduce_results, padded_hidden_states_shape)
            finally:
                self.num_tokens = original_num_tokens
            expected_cpu = expected.detach().cpu()
            result_cpu = result.detach().cpu()
            equal = bool(torch.equal(expected_cpu.contiguous().view(torch.uint8), result_cpu.contiguous().view(torch.uint8)))
            input_equal = bool(torch.equal(before_bits, hidden_states.detach().cpu().contiguous().view(torch.uint8)))
            output_private = result.data_ptr() != hidden_states.data_ptr()
            flag = torch.tensor([int(equal and input_equal and output_private)], dtype=torch.int32, device=hidden_states.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.moe_config.tp_group.device_group)
            all_equal = bool(flag.item())
            result = result[:original_num_tokens]
        else:
            result = method(self, hidden_states, reduce_results, padded_hidden_states_shape)
        rank = dist.get_rank()
        row = dict(rank=rank, pid=_glm53_os.getpid(), mode=mode,
                   actual_hidden_bitwise_equal=equal, input_bytes_equal=input_equal if mode%2 else None, all_ranks_bytes_equal=all_equal if mode%2 else None, input_nan_count=int(torch.isnan(before_cpu).sum()) if mode%2 else None, input_format=torch_npu.get_npu_format(hidden_states), shape=list(hidden_states.shape),
                   padded_shape=list(padded_hidden_states_shape), dtype=str(hidden_states.dtype),
                   local_bytes=hidden_states.numel()*hidden_states.element_size(),
                   input_preserved=input_equal if mode%2 else None,
                   artifact='candidate:'+'26e52eef2714f9e3246d001b9ffc4e05d453a79cf95c3c38c0a2ba860498f3f9')
        (_glm53_root/'witnesses'/('mode%d_rank%d.json'%(mode,rank))).write_text(_glm53_json.dumps(row)+'\n')
        _glm53_last_mode = mode
        if mode%2 and not all_equal: raise RuntimeError('all-rank full-gather byte gate failed; witnesses retained')
        return result
    return method(self, hidden_states, reduce_results, padded_hidden_states_shape)

PrepareAndFinalizeWithAll2All.finalize = _glm53_matched_finalize
