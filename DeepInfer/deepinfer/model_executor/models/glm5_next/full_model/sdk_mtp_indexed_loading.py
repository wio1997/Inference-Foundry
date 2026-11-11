"""Select only shards containing weights consumed by the pinned MTP loader.

Uses existing DefaultModelLoader Source hooks. No checkpoint tensor is changed,
no loaded-parameter set is fabricated, and the original strict loader remains
responsible for coverage. This is initialization work, not inference speedup.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path

CONFIG_SHA256 = '33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943'
INDEX_SHA256 = 'dfaf0918f4c088c98b011def38475d9b1cb395f35ed5e279abc0c57610e23be4'


def select_mtp_shards(root, config, get_spec_layer, *, sort_key):
    root = Path(root)
    raw = (root / 'quant_model_weights.safetensors.index.json').read_bytes()
    if (hashlib.sha256(raw).hexdigest() != INDEX_SHA256
            or hashlib.sha256((root / 'config.json').read_bytes()).hexdigest() != CONFIG_SHA256):
        raise ValueError('MTP indexed loading requires the audited checkpoint metadata')
    index = json.loads(raw)['weight_map']
    selected, accepted = set(), 0
    for name, shard in index.items():
        normalized = name.replace('model.language_model.', 'model.', 1) if name.startswith('model.language_model.') else name
        if name == 'rot.weight' or get_spec_layer(config, normalized) is not None:
            if Path(shard).name != shard or not shard.endswith('.safetensors'):
                raise ValueError('MTP index shard is not an ordinary checkpoint filename')
            selected.add(shard)
            accepted += 1
    # Preserve the original iterator's natural file order, not glob order.
    ordered = sorted((p.name for p in root.glob('*.safetensors') if p.name in selected), key=sort_key)
    if not accepted or set(ordered) != selected:
        raise ValueError('MTP index has missing or empty required shard coverage')
    return ordered, accepted


def install_mtp_indexed_loading(*, expected_loader_sha256, expected_mtp_sha256,
                               expected_model_sha256, expected_weight_utils_sha256):
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.weight_utils import _natural_sort_key
    from vllm_ascend.models.glm5next.mtp import Glm5NextMTP
    from vllm_ascend.models.glm5next.model import get_spec_layer_idx_from_weight_name

    for obj, expected in ((DefaultModelLoader, expected_loader_sha256),
                          (Glm5NextMTP, expected_mtp_sha256),
                          (get_spec_layer_idx_from_weight_name, expected_model_sha256),
                          (_natural_sort_key, expected_weight_utils_sha256)):
        if hashlib.sha256(Path(inspect.getsourcefile(obj)).read_bytes()).hexdigest() != expected:
            raise RuntimeError('MTP indexed loading source differs from audited SDK')
    original = Glm5NextMTP.__init__
    if getattr(original, '_glm53_indexed_loading', False):
        raise RuntimeError('MTP indexed loading already installed')

    @functools.wraps(original)
    def initialize(model, *, vllm_config, prefix=''):
        original(model, vllm_config=vllm_config, prefix=prefix)
        if getattr(model, 'allow_patterns_overrides', None) is not None or getattr(model, 'secondary_weights', ()):
            raise ValueError('MTP model already has a custom weight source')
        cfg = vllm_config.model_config
        shards, accepted = select_mtp_shards(cfg.model, model.config, get_spec_layer_idx_from_weight_name,
                                           sort_key=_natural_sort_key)
        model.allow_patterns_overrides = [shards[0]]
        model.fall_back_to_pt_during_load = False
        # Each Source has one exact pattern because _prepare_weights treats a
        # pattern list as ordered alternatives, not as a union of files.
        model.secondary_weights = tuple(DefaultModelLoader.Source(
            model_or_path=cfg.model, revision=cfg.revision, prefix='',
            fall_back_to_pt=False, allow_patterns_overrides=[name]) for name in shards[1:])
        model._glm53_indexed_loading = {
            'shards': shards, 'source_keys': accepted, 'index_sha256': INDEX_SHA256,
            'scope': 'original strict loader;unchanged tensors;initialization only'}

    initialize._glm53_indexed_loading = True
    Glm5NextMTP.__init__ = initialize
