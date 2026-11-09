"""910C source-defined DSA precision contract and real W8A8 indexer integration.

The original FP32 probe and its failing evidence remain separate. No model
throughput or full40-layer accuracy claim is made by this operator suite.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

import torch

from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.w8a8_indexer910c import AscendW8A8Indexer
from deepinfer.npu_ops.dsa910c import DSABatch, load_ops, quantize_query
from examples.ascend910c_dsa_numeric_oracle import qli_scores, swa_softmax_pv
from examples.ascend910c_dsa_probe import attention_case, index_case, shuffled_pages


def real_indexer_case(model_dir, layer_id, generator):
    reader = ModelSlimCheckpoint(model_dir)
    c = reader.config['text_config']
    consumer = layer_id > c['candidate_source_layer_id']
    source_id = c['candidate_source_layer_id'] if consumer else layer_id
    source = AscendW8A8Indexer(reader, 'npu:0', layer_id=source_id)
    op = AscendW8A8Indexer(reader, 'npu:0', layer_id=layer_id) if consumer else source
    ratio = c['compress_ratios'][layer_id]
    seq_len, queries = (32769 if ratio == 2 else 32768), 2
    count = seq_len//ratio
    latent = torch.randn((count, 512), generator=generator).bfloat16().npu()
    source_positions = (torch.arange(count, dtype=torch.int32)*ratio).npu()
    key, key_scale = source.key(latent, source_positions)
    paged, table = shuffled_pages(key.cpu(), 128//ratio, generator)
    scale_pages = torch.zeros(paged.shape[:3], dtype=torch.float16)
    for logical, physical in enumerate(table.cpu()[0].tolist()):
        values = key_scale.cpu()[logical*(128//ratio):(logical+1)*(128//ratio), 0]
        scale_pages[physical, :values.numel(), 0] = values
    hidden = torch.randn((queries, c['hidden_size']), generator=generator).bfloat16().npu()
    qr = torch.randn((queries, c['q_lora_rank']), generator=generator).bfloat16().npu()
    batch = DSABatch.build([queries], [seq_len], 'npu:0')
    q, qs, weights = op.query(qr, hidden, batch.positions())
    candidates = (torch.arange(2048, dtype=torch.int32)*2)[None, None].expand(queries, 1, -1).contiguous() if consumer else None
    actual, emitted = op.select(qr, hidden, batch, paged, scale_pages.npu(), table,
                               candidates=None if candidates is None else candidates.npu())
    scores = qli_scores(q.cpu(), key.cpu()[:, 0], qs.cpu(), key_scale.cpu()[:, 0], weights.cpu())
    valid = torch.arange(count)[None] < ((batch.positions().cpu()+1)//ratio)[:, None]
    if consumer:
        allowed = torch.zeros_like(valid)
        ids = (candidates[0, 0, :, None].long()*8+torch.arange(8)).flatten()
        allowed[:, ids[ids < count]] = True
        valid &= allowed
    scores.masked_fill_(~valid, float('-inf'))
    expected = scores.topk(512, -1).indices.sort(-1).values.int()
    result = {'layer': layer_id, 'source': source_id, 'seq_len': seq_len,
              'query_projection_weight_dtype': str(op.query_projection.weight.dtype),
              'exact_source_precision_topk': bool(torch.equal(actual.cpu(), expected)),
              'query_dtype': str(q.dtype), 'key_dtype': str(key.dtype),
              'candidate_emitted_shape': None if emitted is None else list(emitted.shape)}
    result['pass'] = result['exact_source_precision_topk'] and op.query_projection.weight.dtype == torch.int8
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch_npu  # noqa: F401
    from deepinfer.npu_ops.w8a8 import configure_native_runtime
    load_ops()  # CANN custom OPP must be registered before set_device.
    torch.npu.set_device(0)
    configure_native_runtime()
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(20261027)
    results = {'scope': 'source-defined operator precision and real indexer integration only', 'cases': {}}
    cases = [(f'qli-r{ratio}-mode{mode}', index_case, (s, q, ratio, mode), {'score_oracle': qli_scores})
             for s, q, ratio, mode in [(1024, 8, 1, 3), (32769, 2, 2, 3),
                                        (32768, 1, 1, 1), (32768, 2, 1, 2)]]
    cases += [(f'mla-r{ratio}-s{s}', attention_case, (s, q, ratio), {'swa_oracle': swa_softmax_pv})
              for s, q, ratio in [(17, 17, 0), (32769, 2, 0), (32768, 2, 1), (32769, 2, 2)]]
    cases += [(f'real-indexer-{i}', real_indexer_case, (args.model_dir, i), {}) for i in (2, 20, 24)]
    for name, operation, parameters, options in cases:
        try:
            result = operation(*parameters, generator, **options)
            torch.npu.synchronize()
        except Exception:
            result = {'pass': False, 'exception': traceback.format_exc()}
        results['cases'][name] = result
        args.output.write_text(json.dumps(results, indent=2))
        print(name, json.dumps(result), flush=True)
    results['vllm_python_modules'] = [n for n in sys.modules if n == 'vllm' or n.startswith('vllm.')]
    results['pass'] = all(r['pass'] for r in results['cases'].values()) and not results['vllm_python_modules']
    args.output.write_text(json.dumps(results, indent=2))
    if not results['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
