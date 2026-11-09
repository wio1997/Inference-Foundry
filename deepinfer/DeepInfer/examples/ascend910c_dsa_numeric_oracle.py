"""CPU-only diagnostics for source-defined CANN precision boundaries.

Keep the ideal FP32 oracle failure. These independent integer/tensor formulas
test a different contract, justified by the pinned operator source, without
changing tolerances or feeding CPU results into the production implementation.
"""
import argparse
import json
from pathlib import Path

import torch


def qli_scores(q, k, query_scales, key_scales, weights):
    # quant_lightning_indexer_v2_service_cube_arch22.h::FixpSToL1
    # uses ReLU + DEQF16 with 0x3a800000 (1/1024); ComputeWs consumes
    # FP16 scores and FP16 weights. ProcessVec0 multiplies weights*qscale
    # in FP16; ProcessVec1 multiplies the FP32 head sum by the key scale.
    values = []
    for query, scale, weight in zip(q, query_scales, weights):
        dot = query.int() @ k.int().T
        rounded = (dot.float().clamp_min(0)/1024.).half().float()
        combined_weights = (weight.float()*scale.float()).half().float()
        values.append((rounded*combined_weights[:, None]).sum(0)*key_scales.float())
    return torch.stack(values)


def swa_softmax_pv(q, rows, sink):
    # sparse_flash_mla_swa_block_vector.h::SoftmaxFlashV2Compute,
    # ProcessVec1SingleBuf and ProcessVec2: unnormalized exp(P) is rounded
    # to BF16 before cube PV; FP32 sum (including sink) divides PV later.
    scores = q.float() @ rows.float().T * 512**-.5
    maximum = torch.maximum(scores.amax(-1), sink)
    exponentials = (scores-maximum[:, None]).exp()
    denominator = exponentials.sum(-1)+(sink-maximum).exp()
    return (exponentials.bfloat16().float() @ rows.float()/denominator[:, None]).bfloat16()


def compare(actual, expected):
    delta = (actual.float()-expected.float()).abs()
    relative = float(delta.norm()/expected.float().norm())
    failures = int((delta > .0005+.008*expected.float().abs()).sum())
    return {'relative_l2': relative, 'pointwise_failures': failures,
            'max_absolute_error': float(delta.max()),
            'pass': not failures and relative <= .006}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = {'scope': 'CPU reproduction of pinned CANN precision boundaries, not end-to-end model accuracy', 'cases': {}}
    for path in sorted(args.input_dir.glob('index-*.pt')):
        data = torch.load(path, map_location='cpu', weights_only=True)
        score = qli_scores(data['q'], data['k'], data['qs'], data['ks'], data['weights'])
        score.masked_fill_(~data['valid'], float('-inf'))
        indices = score.topk(512, -1).indices.sort(-1).values.int()
        original = data['actual']
        results['cases'][path.stem] = {'exact_source_precision_topk': bool(torch.equal(original, indices)),
            'source_precision_recall': [len(set(a.tolist()) & set(b.tolist()))/512 for a, b in zip(original, indices)],
            'ideal_FP32_exact_topk_diagnostic': bool(torch.equal(original, data['expected']))}
    for path in sorted(args.input_dir.glob('attn-r0-*.pt')):
        data = torch.load(path, map_location='cpu', weights_only=True)
        expected = []
        for row, position in enumerate(data['positions'].tolist()):
            rows = data['swa'][max(0, position-127):position+1, 0]
            expected.append(swa_softmax_pv(data['q'][row], rows, data['sink']))
        expected = torch.stack(expected)
        results['cases'][path.stem] = compare(data['actual'], expected)
        results['cases'][path.stem]['ideal_FP32_diagnostic'] = compare(data['actual'], data['expected'])
    import sys
    results['npu_initialized'] = bool('torch_npu' in sys.modules and torch.npu.is_initialized())
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
