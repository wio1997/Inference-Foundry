from collections.abc import Iterator, Sequence

from deepinfer import AttrDict


def iter_runtime_attn_modules(
    attn_module_list: Sequence[AttrDict | None],
    n_main_layers: int,
) -> Iterator[tuple[int, int, AttrDict]]:
    """遍历 attention list。注意 lm head 也会出现在 attention list 中，特判掉这一层，不计入 layer id。config_idx 为原始的 id，一般不会被使用。"""
    for config_idx, attn_module in enumerate(attn_module_list):
        if config_idx == n_main_layers:
            continue
        assert attn_module is not None
        runtime_layer_id = config_idx if config_idx < n_main_layers else config_idx - 1
        yield runtime_layer_id, config_idx, attn_module
