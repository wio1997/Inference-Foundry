"""CPU contract tests. None of these executes/initializes an NPU."""
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import torch

from deepinfer.hugging_face.config import load_hf_model_config
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.ascend_w8a8 import validate_short_context
from deepinfer.model_executor.models.w8a8_math910c import compress_step, rms_norm_mixed, rope_frequencies, rotate_partial, rotated_engram_gate
from deepinfer.npu_ops.moe910c import dispatch_combine


CONFIG = json.loads((Path(__file__).parent/'data/ascend_w8a8_config.json').read_text())


def write_checkpoint(root, entries, quant, config=None):
    payload = bytearray()
    header = {}
    for name, (value, dtype) in entries.items():
        raw = value.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {'dtype': dtype, 'shape': list(value.shape), 'data_offsets': [len(payload), len(payload)+len(raw)]}
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    (root/'shard').write_bytes(struct.pack('<Q', len(encoded))+encoded+payload)
    (root/'quant_model_weights.safetensors.index.json').write_text(json.dumps({'weight_map': dict.fromkeys(entries, 'shard')}))
    (root/'quant_model_description.json').write_text(json.dumps(quant))
    (root/'config.json').write_text(json.dumps(config or CONFIG))
    return ModelSlimCheckpoint(root)


@pytest.fixture
def quant_checkpoint(tmp_path):
    weight = torch.tensor([[-128, -1, 127], [3, -17, 9], [1, 2, 3], [-8, 2, 0]], dtype=torch.int8)
    scales = torch.tensor([[.12345], [.5], [.25], [.75]])
    entries = {'linear.weight': (weight, 'I8'), 'linear.weight_scale': (scales, 'F32'), 'linear.weight_offset': (torch.zeros_like(scales), 'F32')}
    return write_checkpoint(tmp_path, entries, {'linear.weight': 'W8A8_DYNAMIC'}), weight, scales


def test_signed_int8_and_bf16_scales(quant_checkpoint):
    reader, weight, scales = quant_checkpoint
    actual, scale = reader.linear_payload('linear')
    assert actual.dtype == torch.int8
    torch.testing.assert_close(actual, weight, rtol=0, atol=0)
    torch.testing.assert_close(scale, scales.flatten().bfloat16(), rtol=0, atol=0)


def test_sharding_reads_only_requested_rows(quant_checkpoint):
    reader, weight, scale = quant_checkpoint
    reader.max_read_bytes = 16  # Full 16-byte scales fit; shard weight uses 6 bytes.
    actual, scales = reader.linear_payload('linear', output_slice=slice(1, 3))
    torch.testing.assert_close(actual, weight[1:3], rtol=0, atol=0)
    torch.testing.assert_close(scales, scale[1:3].flatten().bfloat16(), rtol=0, atol=0)
    with pytest.raises(ValueError, match='contiguous'):
        reader.linear_payload('linear', output_slice=slice(None, None, 2))


@pytest.mark.parametrize('kind', ['negative_scale', 'nan_scale', 'offset', 'shape', 'bf16_underflow'])
def test_invalid_quantization_rejected(tmp_path, kind):
    scale = torch.ones(4, 1)
    offset = torch.zeros_like(scale)
    if kind == 'negative_scale':
        scale[0] = -1
    elif kind == 'nan_scale':
        scale[0] = float('nan')
    elif kind == 'offset':
        offset[0] = .1
    elif kind == 'shape':
        offset = torch.zeros(4)
    else:
        scale[0] = 1e-43
    reader = write_checkpoint(tmp_path, {'x.weight': (torch.ones(4, 3, dtype=torch.int8), 'I8'),
        'x.weight_scale': (scale, 'F32'), 'x.weight_offset': (offset, 'F32')}, {'x.weight': 'W8A8_DYNAMIC'})
    with pytest.raises(ValueError):
        reader.linear_payload('x')


def test_chunk_materialization_preserves_dense_and_int8(quant_checkpoint):
    reader, weight, _ = quant_checkpoint
    reader.max_read_bytes = 6
    torch.testing.assert_close(reader.materialize('linear.weight', 'cpu', chunk_bytes=6), weight, rtol=0, atol=0)
    with pytest.raises(ValueError, match='smaller than one row'):
        reader.materialize('linear.weight', 'cpu', chunk_bytes=2)


def test_engram_signed_groups_sentinel_and_duplicate_order(tmp_path):
    codes = torch.full((3, 256), -7, dtype=torch.int8)
    codes[1] = 11
    exponents = torch.tensor([[127]*8, [126]*8, [128]*8], dtype=torch.uint8)
    reader = write_checkpoint(tmp_path, {'layers.1.engram.embed.weight': (codes, 'I8'),
        'layers.1.engram.embed.scale': (exponents, 'F8_E8M0')}, {})
    values = reader.engram_rows(1, torch.tensor([[2, 1], [-1, 2]]), 'cpu')
    expected = torch.tensor([[-14., 5.5], [0., -14.]]).bfloat16()[..., None].expand(2, 2, 256)
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    for invalid in ([-2], [3], [1.5]):
        with pytest.raises(ValueError):
            reader.engram_rows(1, invalid, 'cpu')


def test_ascend_config_keeps_original_yoco_metadata_without_fp8(quant_checkpoint):
    reader, _, _ = quant_checkpoint
    cfg = load_hf_model_config(reader.root/'config.json')
    assert cfg.gpt.ascend_w8a8
    assert not any((cfg.gpt.fp8_attn, cfg.gpt.fp8_mlp, cfg.gpt.mxfp8, cfg.gpt.mxfp4, cfg.gpt.use_fp4_sparse_kv))
    assert cfg.gpt.n_prefill_layer == 20
    assert cfg.gpt.engram.cpu_offload and cfg.gpt.engram.scale_group_size == 32


def test_original_fp8_config_preserved(quant_checkpoint):
    reader, _, _ = quant_checkpoint
    config = copy.deepcopy(CONFIG)
    config['quantization_config'] = {'quant_method': 'fp8', 'expert_dtype': 'fp4', 'activation_scheme': 'dynamic', 'weight_block_size': [32, 32], 'scale_fmt': 'ue8m0'}
    (reader.root/'config.json').write_text(json.dumps(config))
    cfg = load_hf_model_config(reader.root/'config.json')
    assert cfg.gpt.fp8_attn and cfg.gpt.mxfp4 and cfg.gpt.use_fp4_sparse_kv
    assert cfg.gpt.n_prefill_layer == 20 and not cfg.gpt.ascend_w8a8


def test_unknown_quantization_rejected(quant_checkpoint):
    reader, _, _ = quant_checkpoint
    config = copy.deepcopy(CONFIG)
    config['quantization_config'] = {'quant_method': 'unknown'}
    (reader.root/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='quant_method'):
        load_hf_model_config(reader.root/'config.json')


def test_route_weights_applied_after_expert_output():
    hidden = torch.tensor([[1., 3.], [2., -4.]]).bfloat16()
    ids = torch.tensor([[0, 1], [1, -1]])
    weights = torch.tensor([[.23, .77], [.5, 0.]])
    seen = []
    def expert(expert_id, tokens, route_weights):
        seen.append(expert_id)
        return (tokens.float() * (expert_id+1) + .13).bfloat16()
    output = dispatch_combine(hidden, ids, weights, 2, expert, weight_on_output=True)
    expected = torch.stack([((hidden[0].float()+.13).bfloat16().float()*.23 + (hidden[0].float()*2+.13).bfloat16().float()*.77),
        (hidden[1].float()*2+.13).bfloat16().float()*.5]).bfloat16()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert seen == [0, 1]


def test_c2_streaming_pool_axis_rounding_and_position():
    x = torch.tensor([[1., 2.], [-3., 4.]]).bfloat16()
    wkv = torch.tensor([[.17, .53], [-.37, .11]])
    wgate = torch.tensor([[.2, -.8], [-1.1, .3]])
    norm = torch.tensor([.5, 2.]).bfloat16()
    pending = []
    assert compress_step(x[:1], 0, 2, wkv, wgate, norm, 1e-20, pending) == (None, None)
    actual, position = compress_step(x[1:], 1, 2, wkv, wgate, norm, 1e-20, pending)
    k, scores = x.float()@wkv.T, x.float()@wgate.T
    # Independent two-element softmax formula, per output channel.
    p0 = torch.sigmoid(scores[0]-scores[1])
    pooled = (k[0]*p0+k[1]*(1-p0)).bfloat16().float()
    expected = (pooled*torch.rsqrt(pooled.square().mean()+1e-20)*norm.float()).bfloat16()[None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert position == 0 and not pending
    with pytest.raises(ValueError, match='preceding'):
        compress_step(x[:1], 3, 2, wkv, wgate, norm, 1e-20, pending)


def test_c1_bf16_projection_before_norm():
    x = torch.tensor([[.25, -1.5]]).bfloat16()
    weight = torch.tensor([[.13, 2.1], [-.27, .14]]).bfloat16()
    norm = torch.ones(2).bfloat16()
    actual, position = compress_step(x, 7, 1, weight, None, norm, 1e-20, [])
    expected = rms_norm_mixed((x.float()@weight.float().T).bfloat16(), norm, 1e-20)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert position == 7


def test_rope_interleaved_inverse_and_nope_preservation():
    cfg = CONFIG['text_config']
    freq = rope_frequencies(cfg, compressed=False)
    expected = 1/(10000**(torch.arange(0, 64, 2).float()/64))
    torch.testing.assert_close(freq, expected, rtol=0, atol=0)
    x = torch.randn(2, 3, 512).bfloat16()
    rotated = rotate_partial(x, 41, freq)
    torch.testing.assert_close(rotated[..., :-64], x[..., :-64], rtol=0, atol=0)
    torch.testing.assert_close(rotate_partial(rotated, 41, freq, inverse=True), x, rtol=.01, atol=.008)
    yarn = rope_frequencies(cfg, compressed=True)
    assert not torch.equal(yarn, freq) and bool((yarn > 0).all())


def test_rotated_engram_matches_original_basis_and_mask():
    torch.manual_seed(1729)
    hidden = torch.randn(2, 3, 8).bfloat16()
    kv = torch.randn(2, 32).bfloat16()
    q, k = torch.randn(3, 8).bfloat16(), torch.randn(3, 8).bfloat16()
    rotation = torch.tensor([[0., 1.], [-1., 0.]])
    mask = torch.tensor([True, False])
    actual = rotated_engram_gate(hidden, kv, q, k, rotation, mask, 1e-20)
    original = hidden.float().reshape(2, 3, 4, 2)@rotation.T
    original = original.flatten(-2)
    keys, value = kv[:, :24].reshape(2, 3, 8).float(), kv[:, 24:].float()
    a = original*torch.rsqrt(original.square().mean(-1, keepdim=True)+1e-20)
    b = keys*torch.rsqrt(keys.square().mean(-1, keepdim=True)+1e-20)
    dot = (a*q.float()*b*k.float()).sum(-1)/math.sqrt(8)
    score = torch.sigmoid(torch.where(torch.signbit(dot), -1., 1.)*dot.abs().clamp_min(1e-6).sqrt())
    expected = (hidden.float()+score[..., None]*value[:, None]).bfloat16()
    expected[1] = hidden[1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_engram_dot_uses_positive_clamped_magnitude():
    hidden = torch.ones(1, 2, 4).bfloat16()
    kv = torch.cat((torch.zeros(1, 8), torch.ones(1, 4)), -1).bfloat16()
    # Large values amplify the +sqrt(epsilon) vs sign(0)==0 distinction.
    kv[:, -4:] = 128
    out = rotated_engram_gate(hidden, kv, torch.ones(2, 4), torch.ones(2, 4), None, torch.tensor([True]), 1e-20)
    expected = (hidden.float()+torch.sigmoid(torch.tensor(.001))*128).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize('max_tokens', [0, 513, 4096])
def test_context_limit_rejects_indexer_approximation(max_tokens):
    with pytest.raises(ValueError):
        validate_short_context(CONFIG, max_tokens)


def test_import_does_not_initialize_platform_or_load_npu():
    # This image auto-registers torch_npu as a PyTorch backend on `import
    # torch`. Backend registration is not a device context. Check lazy device
    # initialization directly, instead of forbidding the registered module.
    script = "import torch, deepinfer; assert not getattr(torch.npu, '_initialized', False); import deepinfer.engine.ascend_w8a8; assert 'PLATFORM' not in vars(deepinfer); assert not getattr(torch.npu, '_initialized', False)"
    subprocess.run([sys.executable, '-c', script], check=True)


def test_cpu_quant_preserves_exact_half_maximum_ties_and_zero_row():
    from examples.ascend910c_w8a8_probe import cpu_quant
    x = torch.tensor([[3.265625, 1.6328125, -1.6328125, 0.], [0., 0., 0., 0.]]).bfloat16()
    codes, scales = cpu_quant(x)
    torch.testing.assert_close(codes, torch.tensor([[127, 64, -64, 0], [0, 0, 0, 0]], dtype=torch.int8), rtol=0, atol=0)
    assert scales[1] == 0 and torch.isfinite(scales).all()
