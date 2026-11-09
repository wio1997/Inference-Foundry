"""Independent CPU-vs-real-CANN paged DSA test, including 32K and C2 tail.

This is operator correctness evidence, never a model performance benchmark.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

import torch

from deepinfer.npu_ops.dsa910c import DSABatch, load_ops, page_rows, quantize_query, select, sparse_mla


def shuffled_pages(rows, page_size, generator):
    cache, _ = page_rows(rows, page_size)
    permutation = torch.randperm(cache.shape[0], generator=generator)
    inverse = permutation.argsort().int()[None]
    return cache[permutation].npu(), inverse.npu()


def index_scores(q, k, qs, ks, weights):
    # Independent INT32 GEMM, dequantization, ReLU and head weighting.
    return torch.stack([((row.int() @ k.int().T).float()
                         * qscale.float()[:, None] * ks.float()[None])
                        .relu().mul(w.float()[:, None]).sum(0)
                        for row, qscale, w in zip(q, qs, weights)])


def index_case(seq_len, query_len, ratio, mode, generator):
    count = seq_len//ratio
    heads = 32
    q = torch.randint(-127, 128, (query_len, heads, 128), generator=generator, dtype=torch.int8)
    k = torch.randint(-127, 128, (count, 128), generator=generator, dtype=torch.int8)
    qs = (torch.rand((query_len, heads), generator=generator)*.01+.01).half()
    ks = (torch.rand(count, generator=generator)*.01+.01).half()
    weights = torch.randn((query_len, heads), generator=generator).half()
    candidates = None
    if mode == 2:
        # Restrict to 2048 nonadjacent eight-key blocks in the 32K source.
        candidates = (torch.arange(2048, dtype=torch.int32)*2)[None, None].expand(query_len, 1, -1).contiguous()
    scores = index_scores(q, k, qs, ks, weights)
    positions = torch.arange(seq_len-query_len, seq_len)
    visible = (positions+1)//ratio
    valid = torch.arange(count)[None] < visible[:, None]
    if candidates is not None:
        allowed = torch.zeros((query_len, count), dtype=torch.bool)
        for row in range(query_len):
            selected_positions = (candidates[row, 0, :, None].long()*8+torch.arange(8)).flatten()
            allowed[row, selected_positions[selected_positions < count]] = True
        valid &= allowed
    scores.masked_fill_(~valid, float('-inf'))
    expected = scores.topk(512, dim=-1).indices.sort(-1).values.int()
    paged, table = shuffled_pages(k[:, None], 128//ratio, generator)
    # Scales must use exactly the same physical page permutation as keys.
    scale_pages = torch.zeros(paged.shape[:3], dtype=torch.float16)
    for logical, physical in enumerate(table.cpu()[0].tolist()):
        start = logical*(128//ratio)
        values = ks[start:start+128//ratio]
        scale_pages[physical, :values.numel(), 0] = values
    batch = DSABatch.build([query_len], [seq_len], 'npu:0')
    actual, emitted = select(q.npu(), paged, weights.npu(), qs.npu(), scale_pages.npu(),
                             batch, table, ratio=ratio, candidate_mode=mode,
                             candidates=None if candidates is None else candidates.npu())
    actual = actual.cpu()
    torch.save({'q':q,'k':k,'qs':qs,'ks':ks,'weights':weights,'scores':scores,'valid':valid,'actual':actual,'expected':expected,'candidates':candidates}, str(DUMP_DIR/f'index-r{ratio}-mode{mode}.pt'))
    recalls = [len(set(a.tolist()) & set(b.tolist()))/512 for a, b in zip(actual, expected)]
    violations = int(((actual < 0) | (actual >= visible[:, None])).sum())
    result = {'seq_len': seq_len, 'query_len': query_len, 'ratio': ratio,
              'candidate_mode': mode, 'recall': recalls, 'causal_violations': violations,
              'exact_topk_set': bool(torch.equal(actual, expected))}
    if mode == 1:
        emitted = emitted.cpu()
        result['candidate_shape'] = list(emitted.shape)
        result['candidate_unique'] = [int(row.unique().numel()) for row in emitted[:, 0]]
        result['candidate_range_valid'] = bool(((emitted >= 0) & (emitted < (count+7)//8)).all())
    # Save mismatch details instead of accepting approximate recall as success.
    if not result['exact_topk_set']:
        result['differences'] = [{'actual_only': sorted(set(a.tolist())-set(b.tolist())),
                                  'expected_only': sorted(set(b.tolist())-set(a.tolist()))}
                                 for a, b in zip(actual, expected)]
    result['pass'] = result['exact_topk_set'] and not violations
    if mode == 1:
        result['pass'] &= result['candidate_range_valid'] and all(n == 2048 for n in result['candidate_unique'])
    return result


def attention_case(seq_len, query_len, ratio, generator):
    heads = 8
    query = (torch.randn((query_len, heads, 512), generator=generator)*.5).bfloat16()
    swa = torch.randn((seq_len, 1, 512), generator=generator).bfloat16()
    # Future rows are deliberately large, so a missing causal mask is visible.
    swa[-1] *= 3
    compressed = torch.randn((max(seq_len//max(ratio, 1), 1), 1, 512), generator=generator).bfloat16() if ratio else None
    indices = torch.full((query_len, 512), -1, dtype=torch.int32) if ratio else None
    positions = torch.arange(seq_len-query_len, seq_len)
    expected = []
    sink = torch.linspace(-2, 2, heads)
    for row, position in enumerate(positions.tolist()):
        rows = swa[max(0, position-127):position+1, 0]
        if ratio:
            visible = (position+1)//ratio
            picked = torch.randperm(visible, generator=generator)[:512].sort().values
            indices[row, :picked.numel()] = picked.int()
            rows = torch.cat((rows, compressed[picked, 0]))
        logits = query[row].float() @ rows.float().T * 512**-.5
        normalization = torch.logsumexp(torch.cat((logits, sink[:, None]), -1), -1)
        expected.append((torch.exp(logits-normalization[:, None]) @ rows.float()).bfloat16())
    expected = torch.stack(expected)
    swa_cache, swa_table = shuffled_pages(swa, 128, generator)
    kwargs = {}
    if ratio:
        cache, table = shuffled_pages(compressed, 128//ratio, generator)
        kwargs = dict(compressed_cache=cache, compressed_table=table, compressed_indices=indices.npu())
    batch = DSABatch.build([query_len], [seq_len], 'npu:0')
    actual = sparse_mla(query.npu(), swa_cache, batch, swa_table, sink=sink.npu(), ratio=ratio, **kwargs).cpu()
    torch.save({'q':query,'swa':swa,'compressed':compressed,'indices':indices,'positions':positions,'sink':sink,'actual':actual,'expected':expected}, str(DUMP_DIR/f'attn-r{ratio}-s{seq_len}.pt'))
    delta = (actual.float()-expected.float()).abs()
    l2 = float(delta.norm()/expected.float().norm())
    pointwise_failures = int((delta > .0005+.008*expected.float().abs()).sum())
    return {'seq_len': seq_len, 'query_len': query_len, 'ratio': ratio,
            'relative_l2': l2, 'max_absolute_error': float(delta.max()),
            'pointwise_failures': pointwise_failures,
            'pass': not pointwise_failures and l2 <= .006}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    global DUMP_DIR
    DUMP_DIR=args.output.parent/'dsa-numeric-v3';DUMP_DIR.mkdir(exist_ok=True)
    import torch_npu  # noqa: F401
    from deepinfer.npu_ops.w8a8 import configure_native_runtime
    load_ops()
    torch.npu.set_device(0)
    configure_native_runtime()
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(20261027)
    results = {'scope': 'operator correctness only; not model throughput', 'cases': {}}
    cases = [(f'qli-r{ratio}-mode{mode}', index_case, (s, q, ratio, mode))
             for s, q, ratio, mode in [(1024, 8, 1, 3), (32769, 2, 2, 3),
                                        (32768, 1, 1, 1), (32768, 2, 1, 2)]]
    cases += [(f'mla-r{ratio}-s{s}', attention_case, (s, q, ratio))
              for s, q, ratio in [(17, 17, 0), (32769, 2, 0),
                                 (32768, 2, 1), (32769, 2, 2)]]
    for name, operation, parameters in cases:
        try:
            result = operation(*parameters, generator)
            torch.npu.synchronize()
        except Exception:
            result = {'pass': False, 'exception': traceback.format_exc()}
        results['cases'][name] = result
        print(name, json.dumps(result), flush=True)
        args.output.write_text(json.dumps(results, indent=2))
    results['vllm_python_modules'] = [n for n in sys.modules if n == 'vllm' or n.startswith('vllm.')]
    results['pass'] = all(r['pass'] for r in results['cases'].values()) and not results['vllm_python_modules']
    args.output.write_text(json.dumps(results, indent=2))
    if not results['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
