"""W8A8 primitive oracle with explicit quantization and compute boundaries.

The ideal CPU end-to-end nearest-even oracle is still reported. CANN's fast
dynamic-quant path can pick the other nearest integer at exact half ties.
Do not treat those codes as GEMM errors or substitute CPU quantization into
the production path. Validate quantization independently, then validate real
INT8 matmuls and activation against CPU using their actual integer inputs.
"""
import argparse
import json
from pathlib import Path

import torch

from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from examples.ascend910c_w8a8_probe import compare, cpu_expert, cpu_linear, cpu_quant


def quantization_contract(x, codes, scales):
    if x.dtype != torch.bfloat16 or codes.dtype != torch.int8 or codes.shape != x.shape:
        raise AssertionError('quantization dtype/shape mismatch')
    if scales.dtype != torch.float32 or scales.shape != x.shape[:-1]:
        raise AssertionError('quantization scale dtype/shape mismatch')
    maximum = x.double().abs().amax(-1)
    assert torch.isfinite(x).all() and torch.isfinite(scales).all()
    expected_scale = maximum/127.
    torch.testing.assert_close(scales.double(), expected_scale, rtol=2e-6, atol=1e-8)
    ideal = x.double()*127./maximum.clamp_min(1e-30)[..., None]
    error = (codes.double()-ideal).abs()
    # This checks the nearest-integer cell, including both valid half-tie
    # endpoints, with only a small FP32 arithmetic allowance in code units.
    assert float(error.max()) <= .50002, f'non-nearest integer: max error {float(error.max())}'
    assert bool((codes[x == 0] == 0).all())
    assert not bool((codes == -128).any())
    ideal_codes = ideal.round().to(torch.int8)
    differences = codes != ideal_codes
    return {'max_rounding_error_code_units': float(error.max()),
            'different_from_ideal_nearest_even': int(differences.sum()),
            'scope': 'nearest-integer cell and per-token scale; half ties may differ from CPU nearest-even'}


def integer_matmul(codes, weights, scales, token_scales):
    accum = codes.int() @ weights.int().T
    return (accum.float()*token_scales[:, None]*scales.float()[None]).bfloat16()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch_npu
    from deepinfer.npu_ops.w8a8 import AscendW8A8Expert, configure_native_runtime
    torch.npu.set_device(0)
    configure_native_runtime()
    torch.set_num_threads(4)
    reader = ModelSlimCheckpoint(args.model_dir)
    generator = torch.Generator().manual_seed(20261026)
    results = {}
    for prefix in ['layers.0.attn.wq_b', 'layers.0.ffn.experts.0.w1',
                   'layers.0.ffn.experts.0.w2', 'layers.0.ffn.shared_experts.w1']:
        weight, scale = reader.linear_payload(prefix)
        x = torch.randn((3, weight.shape[1]), generator=generator).bfloat16()
        device_x=x.npu()
        device_codes,device_scale=torch_npu.npu_dynamic_quant(device_x,dst_type=torch.int8)
        codes,token_scale=device_codes.cpu(),device_scale.cpu()
        quant_check=quantization_contract(x,codes,token_scale)
        expected=integer_matmul(codes,weight,scale,token_scale)
        ideal=cpu_linear(x,weight,scale)
        for nz in (False, True):
            op = reader.load_linear(prefix, 'npu:0', nz=nz)
            actual = op(device_x)
            key = prefix+('/NZ' if nz else '/ND')
            results[key] = compare(actual, expected, key)
            results[key]['quantization']=quant_check
            results[key]['ideal_cpu_relative_l2_diagnostic']=float((actual.cpu().float()-ideal.float()).norm()/ideal.float().norm())
            assert op.weight.dtype == torch.int8
            if nz:
                assert torch_npu.get_npu_format(op.weight) == 29
                results[key]['npu_format'] = 29
            del op
    for expert_id in (0, 17):
        payload = [reader.linear_payload(f'layers.0.ffn.experts.{expert_id}.w{i}') for i in (1, 3, 2)]
        op = AscendW8A8Expert(*[value.npu() for pair in payload for value in pair], limit=10.)
        x = torch.randn((4, 5120), generator=generator).bfloat16()
        actual, device_stages = op(x.npu(), return_stages=True)
        stages = {name: value.cpu() for name, value in device_stages.items()}
        actual = actual.cpu()
        entry = {'quant1': quantization_contract(x, stages['input_codes'], stages['input_scale'])}
        w13 = torch.cat((payload[0][0], payload[1][0]))
        s13 = torch.cat((payload[0][1], payload[1][1]))
        first = integer_matmul(stages['input_codes'], w13, s13, stages['input_scale'])
        entry['gmm1'] = compare(stages['combined'], first, 'gmm1')
        a, b = stages['combined'].float().chunk(2, -1)
        expected_activation = (torch.nn.functional.silu(a.clamp_max(10.))*b.clamp(-10., 10.)).bfloat16()
        entry['activation'] = compare(stages['activated'], expected_activation, 'activation')
        entry['quant2'] = quantization_contract(stages['activated'], stages['second_codes'], stages['second_scale'])
        second = integer_matmul(stages['second_codes'], payload[2][0], payload[2][1], stages['second_scale'])
        entry['gmm2'] = compare(actual, second, 'gmm2')
        ideal = cpu_expert(x, payload)
        error = actual.float()-ideal.float()
        entry['ideal_cpu_end_to_end_diagnostic'] = {
            'relative_l2': float(error.norm()/ideal.float().norm()),
            'max_abs': float(error.abs().max()),
            'used_as_acceptance': False,
            'reason': 'Actual CANN integer quantization is independently checked above; this is the ideal nearest-even chain.'}
        results[f'expert-{expert_id}'] = entry
        del op, device_stages
    report = {'status': 'pass', 'checkpoint': args.model_dir.name, 'results': results,
              'boundary': 'Real INT8 ND/NZ linears and two experts: independent quantization-cell/scale checks, integer CPU GMM and activation checks. Ideal-chain diagnostics retained. Not complete layer or model proof.'}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
