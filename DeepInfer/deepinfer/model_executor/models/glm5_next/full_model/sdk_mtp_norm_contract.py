"""Scope optional RMSNorm bias registration to this checkpoint's MTP head.

The pinned Ascend RMSNorm registers zero bias for every norm when *any*
quant-description entry has norm.bias. This MTP head exports seven FLOAT
RMSNorm weights without bias. Remove only those absent optional parameters;
do not invent loaded weights or drop the real indexer LayerNorm bias.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path

CHECKPOINT_CONFIG = '33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943'
NORM_PATHS = (
    'model.layers.45.enorm', 'model.layers.45.hnorm',
    'model.layers.45.shared_head.norm',
    'model.layers.45.mtp_block.input_layernorm',
    'model.layers.45.mtp_block.post_attention_layernorm',
    'model.layers.45.mtp_block.self_attn.q_a_layernorm',
    'model.layers.45.mtp_block.self_attn.kv_a_layernorm',
)


def install_mtp_norm_contract(expected_layernorm_sha256):
    from vllm_ascend.models.glm5next.mtp import Glm5NextMTP
    from vllm_ascend.ops.layernorm import AscendRMSNorm
    source = Path(inspect.getsourcefile(AscendRMSNorm))
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_layernorm_sha256:
        raise RuntimeError('MTP norm registration source differs from audited SDK')
    original = Glm5NextMTP.__init__
    if getattr(original, '_glm53_optional_norm_contract', False):
        raise RuntimeError('MTP norm contract already installed')

    @functools.wraps(original)
    def initialize(model, *, vllm_config, prefix=''):
        root = Path(vllm_config.model_config.model)
        if hashlib.sha256((root / 'config.json').read_bytes()).hexdigest() != CHECKPOINT_CONFIG:
            raise ValueError('MTP norm contract applies only to the audited checkpoint')
        description = json.loads((root / 'quant_model_description.json').read_text())
        index_path = root / 'quant_model_weights.safetensors.index.json'
        index = json.loads(index_path.read_text())['weight_map']
        original(model, vllm_config=vllm_config, prefix=prefix)
        selected = []
        for name in NORM_PATHS:
            source_name = name.replace('.mtp_block.', '.')
            source_name = source_name.replace('model.', 'model.language_model.', 1)
            if (description.get(source_name + '.weight') != 'FLOAT'
                    or source_name + '.weight' not in index
                    or source_name + '.bias' in index
                    or source_name + '.bias' in description):
                raise ValueError('checkpoint no-bias norm contract changed: ' + name)
            norm = model.get_submodule(name)
            if not isinstance(norm, AscendRMSNorm) or norm.bias is None or norm.bias_loaded:
                raise ValueError('unexpected MTP norm registration: ' + name)
            selected.append((name, norm))
        for _, norm in selected:
            norm.register_parameter('bias', None)
        model._glm53_mtp_norm_contract = {
            'removed_absent_optional_biases': [name + '.bias' for name, _ in selected],
            'checkpoint_config_sha256': CHECKPOINT_CONFIG,
            'layernorm_source_sha256': expected_layernorm_sha256,
            'indexer_bias_preserved': 'model.language_model.layers.45.self_attn.indexer.k_norm.bias' in index,
            'scope': 'registration only;no tensor values or weight-loaded set fabricated',
        }

    initialize._glm53_optional_norm_contract = True
    Glm5NextMTP.__init__ = initialize
