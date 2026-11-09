"""CPU integer emulation exercises model wiring, not NPU operator correctness."""
import copy
import importlib

import torch

from deepinfer.model_executor.models.ascend_w8a8 import AscendW8A8Model
from deepinfer.hugging_face.modelslim910c import FloatLinear
from tests.test_ascend910c_w8a8 import CONFIG, write_checkpoint


def integer_linear(x, weight, scale):
    maximum = x.float().abs().amax(-1)
    sx = maximum/127.
    codes = torch.round(x.float()*127./maximum.clamp_min(1e-30)[:, None]).clamp(-128, 127).int()
    return ((codes@weight.int().T).float()*sx[:, None]*scale.float()[None]).bfloat16()


class CpuIntLinear(torch.nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('scale', scale)

    def forward(self, x):
        return integer_linear(x, self.weight, self.scale)


class CpuIntExpert(torch.nn.Module):
    def __init__(self, w1, s1, w3, s3, w2, s2, *, limit):
        super().__init__()
        self.limit = limit
        self.first = CpuIntLinear(torch.cat((w1, w3)), torch.cat((s1, s3)))
        self.second = CpuIntLinear(w2, s2)

    def forward(self, x):
        gate, up = self.first(x).float().chunk(2, -1)
        hidden = (torch.nn.functional.silu(gate.clamp_max(self.limit))*up.clamp(-self.limit, self.limit)).bfloat16()
        return self.second(hidden)


def tiny_model(tmp_path, monkeypatch):
    torch.manual_seed(431)
    config = copy.deepcopy(CONFIG)
    c = config['text_config']
    c.update(hidden_size=32, num_hidden_layers=6, vocab_size=64, num_attention_heads=4,
        head_dim=8, qk_rope_head_dim=4, q_lora_rank=8, o_groups=2, o_lora_rank=4,
        hc_mult=2, hc_sinkhorn_iters=5, n_routed_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=8, engram_layer_ids=[], kv_source_layer_ids=[2, 4],
        compress_ratios=[0, 0, 2, 2, 1, 1], sliding_window=3, index_topk=8)
    config.pop('engram_rotation_config')
    entries, quant = {}, {}
    def dense(name, shape, *, dtype=torch.bfloat16, ones=False):
        value = torch.ones(shape) if ones else torch.randn(shape)*.1
        entries[name] = (value.to(dtype), 'BF16' if dtype==torch.bfloat16 else 'F32')
        if name.endswith('.weight') and len(shape)==2:
            quant[name] = 'FLOAT'
    def int8(name, shape):
        entries[name+'.weight'] = (torch.randint(-127, 128, shape, dtype=torch.int8), 'I8')
        entries[name+'.weight_scale'] = (torch.rand(shape[0], 1)*.002+.001, 'F32')
        entries[name+'.weight_offset'] = (torch.zeros(shape[0], 1), 'F32')
        quant[name+'.weight'] = 'W8A8_DYNAMIC'
    dense('embed.weight', (64, 32))
    dense('head.weight', (64, 32))
    dense('norm.weight', (32,), ones=True)
    for i in range(6):
        p = f'layers.{i}'
        for kind in ('attn', 'ffn'):
            dense(p+f'.hc_{kind}_fn', (8, 64), dtype=torch.float32)
            dense(p+f'.hc_{kind}_base', (8,), dtype=torch.float32)
            dense(p+f'.hc_{kind}_scale', (3,), dtype=torch.float32)
            dense(p+f'.{kind}_norm.weight', (32,), ones=True)
        for name, shape in [('wq_a', (8, 32)), ('wkv', (8, 32)), ('wo_a', (8, 16)), ('wo_b', (32, 8))]:
            dense(p+f'.attn.{name}.weight', shape)
        int8(p+'.attn.wq_b', (32, 8))
        dense(p+'.attn.q_norm.weight', (8,), ones=True)
        dense(p+'.attn.kv_norm.weight', (8,), ones=True)
        dense(p+'.attn.attn_sink', (4,))
        dense(p+'.ffn.gate.weight', (4, 32))
        # The real checkpoint stores this BF16; the router requires FP32.
        dense(p+'.ffn.gate.bias', (4,), dtype=torch.bfloat16)
        for expert in ['shared_experts']+[f'experts.{j}' for j in range(4)]:
            for matrix, shape in [('w1', (8, 32)), ('w3', (8, 32)), ('w2', (32, 8))]:
                int8(p+f'.ffn.{expert}.{matrix}', shape)
        if i in (2, 4):
            dense(p+'.attn.compressor.wkv.weight', (8, 32))
            dense(p+'.attn.compressor.norm.weight', (8,), ones=True)
            if i==2:
                dense(p+'.attn.compressor.wgate.weight', (8, 32))
    reader = write_checkpoint(tmp_path, entries, quant, config)
    def load_linear(prefix, device, **kwargs):
        weight, scale = reader.linear_payload(prefix)
        return FloatLinear(weight) if scale is None else CpuIntLinear(weight, scale)
    reader.load_linear = load_linear
    layer_module = importlib.import_module('deepinfer.model_executor.models.ascend_w8a8_layer')
    monkeypatch.setattr(layer_module, 'AscendW8A8Expert', CpuIntExpert)
    return AscendW8A8Model(reader, 'cpu', max_tokens=8)


def test_full_backbone_resident_int8_cache_sharing_and_reset(tmp_path, monkeypatch):
    model = tiny_model(tmp_path, monkeypatch)
    assert len(model.layers)==6 and all(len(layer.expert_cache)==4 for layer in model.layers)
    assert model.resident_bytes()['torch.int8']>0
    first = []
    for token_id in [0, 3, 5, 7, 9]:
        first.append(model.step(token_id))
    assert model.cache.next_position==5
    assert all(value.shape==(3, 8) for value in model.cache.swa.values())
    assert model.cache.compressed[2].shape==(2, 8)
    assert model.cache.compressed[4].shape==(5, 8)
    assert len(model.cache.pending[2])==2
    assert set(model.cache.compressed)=={2, 4}  # consumers do not own duplicates
    assert all(torch.isfinite(logits).all() for logits in first)
    model.reset()
    assert not model.cache.swa and not model.cache.pending
    for expected, token_id in zip(first, [0, 3, 5, 7, 9], strict=True):
        torch.testing.assert_close(model.step(token_id), expected, rtol=0, atol=0)


def test_streaming_generation_does_not_reuse_previous_request(tmp_path, monkeypatch):
    model = tiny_model(tmp_path, monkeypatch)
    model.step(0)
    expected = model.step(3)
    model.step(8)  # future token cannot change the already published result
    model.reset()
    model.step(0)
    torch.testing.assert_close(model.step(3), expected, rtol=0, atol=0)
