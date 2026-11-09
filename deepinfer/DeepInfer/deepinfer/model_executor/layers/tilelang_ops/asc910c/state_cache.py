"""NPU byte transport for the original circular state-cache API."""
import torch


def _state_cache_copy(block_size_kvstore, block_size_inference, num_layers,
                      feature_dim, *, cache_to_tensor):
    if (any(type(value) is not int or value < 1 for value in
            (block_size_kvstore, block_size_inference, feature_dim))
            or type(num_layers) is not int or num_layers < 0
            or block_size_inference > block_size_kvstore):
        raise ValueError('state cache must cover a positive inference window')

    def run(state_tensor, state_cache, last_hit_position):
        for value, rows in ((state_tensor, block_size_inference), (state_cache, block_size_kvstore)):
            if (not isinstance(value, torch.Tensor) or value.device.type != 'npu'
                    or value.dtype != torch.uint8 or tuple(value.shape) != (num_layers, rows, feature_dim)
                    or value.stride(-1) != 1):
                raise TypeError('original state-cache copies require NPU uint8 token records')
        if state_tensor.device != state_cache.device:
            raise ValueError('state-cache buffers must belong to the same NPU')
        if type(last_hit_position) is not int:
            raise TypeError('state-cache endpoint must be a host integer')
        positions = torch.arange(last_hit_position-block_size_inference, last_hit_position,
                                 dtype=torch.int64, device=state_cache.device)
        cache_rows = positions.remainder(block_size_kvstore)
        tensor_rows = positions.remainder(block_size_inference)
        # Signed views preserve bytes for INT8 gather and retain arbitrary
        # layer/token padding strides.
        target, cache = state_tensor.view(torch.int8), state_cache.view(torch.int8)
        if cache_to_tensor:
            target.index_copy_(1, tensor_rows, cache.index_select(1, cache_rows))
        else:
            cache.index_copy_(1, cache_rows, target.index_select(1, tensor_rows))

    return run


def select_state_slice_tl(block_size_kvstore, block_size_inference, num_layers, feature_dim):
    return _state_cache_copy(block_size_kvstore, block_size_inference, num_layers,
                             feature_dim, cache_to_tensor=True)


def load_state_to_cache_tl(block_size_kvstore, block_size_inference, num_layers, feature_dim):
    return _state_cache_copy(block_size_kvstore, block_size_inference, num_layers,
                             feature_dim, cache_to_tensor=False)
