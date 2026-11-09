# SPDX-License-Identifier: Apache-2.0
"""Eager native 910C text backbone: resident INT8 experts, BF16 mixed modules.

This first engine path accepts a single short text sequence, greedy decoding
and EP sharding. It does not call vLLM at runtime. While max_tokens <= index_topk,
all available compressed keys are selected: the indexer and candidate filter
cannot exclude any key. Longer contexts fail before loading or computation.
MTP, vision, speculative decoding, ACL graphs and serving are outside this path.
"""
import torch
import torch.nn.functional as F

from .ascend_w8a8_layer import AscendW8A8Layer
from .w8a8_engram910c import AscendW8A8Engram, CpuEngramHistory, load_rotation
from .w8a8_math910c import W8A8SequenceCache, rms_norm_mixed


def validate_short_context(config, max_tokens):
    text = config['text_config']
    if not 1 <= max_tokens <= text['index_topk']:
        raise ValueError('eager W8A8 max_tokens must be positive and <= index_topk')
    if max_tokens > text['candidate_topk_blocks'] * text['candidate_block_size']:
        raise ValueError('candidate capacity is too small for select-all compressed keys')
    if text['n_shared_experts'] != 1 or text['scoring_func'] != 'sqrtsoftplus' or not text['norm_topk_prob']:
        raise ValueError('unsupported MoE routing contract')
    if text.get('hc_shifted', True) is not True:
        raise ValueError('eager W8A8 requires shifted mHC')
    if any(r not in (0, 1, 2) for r in text['compress_ratios'][:text['num_hidden_layers']]):
        raise ValueError('unsupported backbone compression ratio')


class AscendW8A8Model(torch.nn.Module):
    def __init__(self, reader, device, *, max_tokens=512, group=None, progress=None):
        super().__init__()
        validate_short_context(reader.config, max_tokens)
        self.reader, self.device, self.max_tokens = reader, torch.device(device), max_tokens
        self.cfg = reader.config['text_config']
        self.progress = progress or (lambda event: None)
        self.progress({'stage': 'embedding_and_head'})
        # Chunked host staging also handles the 1.32 GB dense vocabulary matrices.
        self.register_buffer('embedding', reader.materialize('embed.weight', device))
        self.register_buffer('head', reader.materialize('head.weight', device))
        self.register_buffer('norm', reader.tensor('norm.weight')[0].to(device))
        self.engram_history = CpuEngramHistory(reader) if self.cfg.get('engram_layer_ids') else None
        self.register_buffer('engram_rotation', load_rotation(reader, device) if self.engram_history else None)
        self.engrams = torch.nn.ModuleDict({str(i): AscendW8A8Engram(reader, i, device) for i in self.cfg.get('engram_layer_ids', [])})
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.cfg['num_hidden_layers']):
            layer = AscendW8A8Layer(reader, device, layer_id=layer_id, group=group)
            layer.load_local_experts()
            self.layers.append(layer)
            self.progress({'stage': 'resident_layer', 'layer': layer_id, 'local_experts': len(layer.expert_cache)})
        self.reset()

    def reset(self):
        self.cache = W8A8SequenceCache(self.max_tokens)
        if self.engram_history:
            self.engram_history.reset()

    @torch.inference_mode()
    def step(self, token_id):
        position = self.cache.next_position
        if position >= self.max_tokens:
            raise ValueError('sequence exceeds eager W8A8 context limit')
        if not 0 <= token_id < self.embedding.shape[0]:
            raise ValueError('token ID outside vocabulary')
        image_ids = {self.reader.config.get('image_token_id'), self.reader.config.get('image_pad_token_id')}
        if token_id in image_ids:
            raise ValueError('this W8A8 engine entry accepts text tokens only')
        hashes = self.engram_history.step(token_id) if self.engram_history else {}
        trunk = self.embedding[token_id:token_id+1, None, :].expand(-1, self.cfg['hc_mult'], -1).contiguous()
        previous = torch.zeros((1, self.cfg['hc_mult'], 1), dtype=torch.float32, device=self.device)
        previous[:, 0] = 1.
        for layer_id, layer in enumerate(self.layers):
            if str(layer_id) in self.engrams:
                trunk = self.engrams[str(layer_id)](trunk, hashes[layer_id], self.engram_rotation)
            trunk, previous = layer(trunk, previous, position=position, cache=self.cache)
        hidden = (trunk.float() * previous).sum(-2).bfloat16()
        hidden = rms_norm_mixed(hidden, self.norm, self.cfg['rms_norm_eps'])[0]
        logits = F.linear(hidden, self.head).float()
        self.cache.next_position += 1
        return logits

    def resident_bytes(self):
        """Logical tensor bytes, excluding allocator/NZ padding and workspace."""
        result = {}
        for tensor in list(self.parameters()) + list(self.buffers()):
            dtype = str(tensor.dtype)
            result[dtype] = result.get(dtype, 0) + tensor.numel() * tensor.element_size()
        return result
