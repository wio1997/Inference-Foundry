# SPDX-License-Identifier: Apache-2.0
"""CPU-owned Engram hash history and selected-row staging for ModelSlim."""
import torch

from deepinfer.hugging_face.config import _build_engram
from deepinfer.model_executor.models.v4_modules.engram_metadata import CompressedVocabulary, NgramHashMapping
from deepinfer.npu_ops.engram910c import hashes
from .w8a8_math910c import rotated_engram_gate


class CpuEngramHistory:
    def __init__(self, reader):
        self.reader = reader
        self.cfg = reader.config['text_config']
        cfg = _build_engram(reader.config, reader.root)
        vocabulary = CompressedVocabulary(cfg.tokenizer_name_or_path)
        if len(vocabulary) != self.cfg['engram_compressed_vocab_size']:
            raise ValueError('Engram compressed vocabulary differs from checkpoint config')
        self.token_map = vocabulary.old2new
        self.token_lookup = torch.from_numpy(vocabulary.lookup_table)
        self.pad_id = self.token_map[cfg.engram_pad_token_id]
        mapping = NgramHashMapping(cfg, device='cpu')
        self.layers = mapping.layer_ids
        self.multipliers = mapping.ngram_multipliers
        self.primes = mapping.ngram_vocab_sizes
        self.offsets = mapping.ngram_offsets
        totals = self.primes.flatten(1).long().sum(-1).tolist()
        if totals != self.cfg['engram_num_embeddings']:
            raise ValueError('Engram prime bucket totals differ from checkpoint tables')
        self.image_ids = {reader.config.get('image_token_id'), reader.config.get('image_pad_token_id')}
        self.reset()

    def reset(self):
        self.history = []

    def step(self, token_id):
        if not 0 <= token_id < len(self.token_map):
            raise ValueError('token outside tokenizer vocabulary')
        self.history.append(token_id)
        values = []
        blocked = False
        for shift in range(self.cfg['engram_max_ngram_size']):
            source = self.history[-shift-1] if shift < len(self.history) else None
            blocked = blocked or source is None or source in self.image_ids
            values.append(self.pad_id if blocked else self.token_map[source])
        tokens = torch.tensor([values], dtype=torch.int32)
        image_mask = torch.tensor([token_id in self.image_ids], dtype=torch.bool)
        output = hashes(tokens, self.multipliers, self.primes, self.offsets, image_mask)
        return {layer: output[i, 0] for i, layer in enumerate(self.layers)}

    def batch(self, token_ids):
        """Hash a whole prefill chunk once, preserving step() history semantics."""
        token_ids = list(token_ids)
        if not token_ids or any(type(t) is not int or not 0 <= t < len(self.token_map) for t in token_ids):
            raise ValueError('a nonempty chunk of valid integer tokens is required')
        width = self.cfg['engram_max_ngram_size']
        tail = self.history[-(width-1):] if width > 1 else []
        context = torch.tensor(tail+token_ids, dtype=torch.int64)
        # Vocabulary normalization is immutable. Converting its129k Python
        # entries for each request/token dominated the CPU staging path.
        token_map = self.token_lookup
        target = torch.arange(len(token_ids))+len(tail)
        blocked = torch.zeros(len(token_ids), dtype=torch.bool)
        values = []
        for shift in range(width):
            source = target-shift
            ids = context[source.clamp_min(0)]
            image = torch.zeros_like(blocked)
            for image_id in self.image_ids-{None}:
                image |= ids == image_id
            blocked |= (source < 0) | image
            values.append(torch.where(blocked, self.pad_id, token_map[ids]))
        image_mask = torch.zeros_like(blocked)
        current = context[len(tail):]
        for image_id in self.image_ids-{None}:
            image_mask |= current == image_id
        output = hashes(torch.stack(values, -1), self.multipliers, self.primes, self.offsets, image_mask)
        self.history.extend(token_ids)
        return {layer: output[i] for i, layer in enumerate(self.layers)}


def load_rotation(reader, device):
    rotation_cfg = reader.config.get('engram_rotation_config', {})
    if not rotation_cfg.get('value_projection_rotated', False):
        return None
    if (rotation_cfg.get('value_basis') != 'quarot_global'
            or rotation_cfg.get('key_and_gate_basis') != 'original'
            or rotation_cfg.get('runtime_delta_rotation', False)):
        raise ValueError('unsupported Engram checkpoint rotation contract')
    from safetensors import safe_open
    path = reader.root / 'optional/quarot.safetensors'
    if not path.is_file():
        raise ValueError('rotated Engram requires optional/quarot.safetensors')
    with safe_open(path, framework='pt') as f:
        rotation = f.get_tensor('global_rotation')
    width = reader.config['text_config']['hidden_size']
    block = rotation[:32, :32].float().contiguous()
    if rotation.shape != (width, width) or width % 32:
        raise ValueError('invalid global rotation shape')
    if not torch.equal(rotation, torch.block_diag(*[block.to(rotation.dtype)] * (width//32))):
        raise ValueError('Engram rotation must repeat the same block32')
    return block.to(device)


class AscendW8A8Engram(torch.nn.Module):
    def __init__(self, reader, layer_id, device):
        super().__init__()
        self.reader = reader
        self.layer_id = layer_id
        self.device = device
        self.eps = reader.config['text_config']['rms_norm_eps']
        prefix = f'layers.{layer_id}.engram'
        # The index intentionally selects the BF16 rotated projection overrides.
        self.projection = reader.load_linear(prefix+'.wkv', device)
        for key in ('q_weight', 'k_weight'):
            self.register_buffer(key, reader.tensor(prefix+'.'+key)[0].to(device))

    def forward(self, trunk, indices, rotation):
        # The global table stays on host/storage. Only 24 selected group32
        # rows (6144 BF16 values) are transferred per sequence position.
        if indices.ndim == 1:
            indices = indices[None]
        if indices.shape[0] != trunk.shape[0]:
            raise ValueError('one Engram row-index group per packed token required')
        rows = self.reader.engram_rows(self.layer_id, indices, self.device).reshape(trunk.shape[0], -1)
        valid = (indices >= 0).all(-1).to(trunk.device)
        return self.forward_rows(trunk,rows,rotation,valid)

    def forward_rows(self,trunk,rows,rotation,valid):
        """Device-only gate for persistent decode graphs; CPU stages rows."""
        kv = self.projection(rows)
        return rotated_engram_gate(trunk, kv, self.q_weight, self.k_weight, rotation, valid, self.eps)
