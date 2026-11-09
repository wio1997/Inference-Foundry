"""Eight-request paged cache/indexer/attention integration on actual 910C.

Tests different query/chunk lengths, C2 odd tails and ring page remapping.
This is synthetic operator integration, not eight-request model serving.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

from deepinfer.model_executor.models.w8a8_paged_cache910c import W8A8PagedCache
from deepinfer.npu_ops.dsa910c import DSABatch, load_ops, select, sparse_mla
from examples.ascend910c_dsa_numeric_oracle import compare, qli_scores, swa_softmax_pv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch_npu  # noqa: F401
    from deepinfer.npu_ops.w8a8 import configure_native_runtime
    load_ops()
    torch.npu.set_device(0)
    configure_native_runtime()
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    config['text_config']['num_hidden_layers'] = 1
    config['text_config']['kv_source_layer_ids'] = [2]
    cache = W8A8PagedCache(config, 'npu:0', max_tokens=32769, requests=8, max_query_tokens=4)
    qlens = [1, 2, 3, 4, 1, 2, 3, 4]
    slens = [32769, 32768, 30001, 29000, 8193, 4096, 2049, 1024]
    slots = [7, 2, 5, 0, 6, 1, 4, 3]
    batch = DSABatch.build(qlens, slens, 'npu:0')
    g = torch.Generator().manual_seed(272610)
    q = torch.randint(-127, 128, (sum(qlens), 32, 128), dtype=torch.int8, generator=g)
    qs = (torch.rand((sum(qlens), 32), generator=g)*.01+.01).half()
    weights = torch.randn((sum(qlens), 32), generator=g).half()
    query = (torch.randn((sum(qlens), 8, 512), generator=g)*.5).bfloat16()
    sink = torch.linspace(-2, 2, 8)
    expected_indices, expected_mla, expected_swa = [], [], []
    cursor = 0
    for slot, qlen, slen in zip(slots, qlens, slens):
        count = slen//2
        key = torch.randint(-127, 128, (count, 128), dtype=torch.int8, generator=g)
        scale = (torch.rand(count, generator=g)*.01+.01).half()
        comp = torch.randn((count, 1, 512), generator=g).bfloat16()
        history_start = max(0, slen-qlen-127)
        swa = torch.randn((slen-history_start, 1, 512), generator=g).bfloat16()
        swa[-1] *= 3
        # Write all current query keys before evaluating the first query.
        # Its preceding history must survive the SWA physical ring reuse.
        plane = cache.swa[0]
        plane.write(swa.npu(), plane.plan(range(history_start, slen), [slot]*swa.shape[0]))
        plans = cache.compressed_plans(2, range(0, count*2, 2), [slot]*count)
        cache.compressed[2].write(comp.npu(), plans[0])
        cache.index_key[2].write(key[:, None].npu(), plans[1])
        cache.index_scale[2].write(scale[:, None, None].npu(), plans[2])
        scores = qli_scores(q[cursor:cursor+qlen], key, qs[cursor:cursor+qlen], scale, weights[cursor:cursor+qlen])
        for local, position in enumerate(range(slen-qlen, slen)):
            visible = (position+1)//2
            score = scores[local].clone()
            score[visible:] = float('-inf')
            selected = score.topk(min(512, visible)).indices.sort().values
            padded = torch.full((512,), -1, dtype=torch.int32)
            padded[:selected.numel()] = selected.int()
            expected_indices.append(padded)
            window = swa[max(0, position-127)-history_start:position+1-history_start, 0]
            rows = torch.cat((window, comp[selected, 0]))
            logits = query[cursor+local].float() @ rows.float().T * 512**-.5
            denominator = torch.logsumexp(torch.cat((logits, sink[:, None]), -1), -1)
            expected_mla.append((torch.exp(logits-denominator[:, None]) @ rows.float()).bfloat16())
            expected_swa.append(swa_softmax_pv(query[cursor+local], window, sink))
        cursor += qlen
    selected, _ = select(q.npu(), cache.index_key[2].cache, weights.npu(), qs.npu(),
                         cache.index_scale[2].cache.squeeze(-1), batch,
                         cache.index_key[2].table(slots), ratio=2)
    compressed_output = sparse_mla(query.npu(), cache.swa[0].cache, batch, cache.swa[0].table(slots),
                                   sink=sink.npu(), ratio=2, compressed_cache=cache.compressed[2].cache,
                                   compressed_table=cache.compressed[2].table(slots), compressed_indices=selected)
    swa_output = sparse_mla(query.npu(), cache.swa[0].cache, batch, cache.swa[0].table(slots), sink=sink.npu())
    torch.npu.synchronize()
    results = {'scope': 'synthetic C8 paged DSA integration only; no model serving/throughput claim',
               'query_lengths': qlens, 'sequence_lengths': slens, 'request_slots': slots,
               'exact_topk_sets': bool(torch.equal(selected.cpu(), torch.stack(expected_indices))),
               'compressed_mla': compare(compressed_output.cpu(), torch.stack(expected_mla)),
               'swa_mla': compare(swa_output.cpu(), torch.stack(expected_swa)),
               'cache_resident_bytes': cache.resident_bytes(),
               'vllm_python_modules': [n for n in sys.modules if n == 'vllm' or n.startswith('vllm.')]}
    results['pass'] = results['exact_topk_sets'] and results['compressed_mla']['pass'] and results['swa_mla']['pass'] and not results['vllm_python_modules']
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if not results['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
