"""Compare the pinned single-head kernel and paired-head candidate on Ascend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch_npu  # Registers the NPU backend before tensor allocation.


BASELINE_SHA256 = '08a8abfc7ba34ebd493b82a9c570487efafa9a9387b24ab34cacbbeb8003fcac'
HIGH_ROW = (1 << 22) + 257
WIDTH = 512
HEADS = 8
HOST_KV_ROWS = 16384
SHAPES = {'prefill': (('prefill', 8192),), 'decode': (('decode', 16), ('draft', 80), ('verify', 96))}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(tensor):
    data = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def save(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def physical(ids):
    return torch.where((ids >= 0) & ((ids & 1) != 0), ids + HIGH_ROW, ids)


def write_pool(pool, values):
    low, high = values.clone(), values.clone()
    low[1::2] = -11.0
    high[0::2] = 13.0
    pool[:len(values)].copy_(low)
    pool[HIGH_ROW:HIGH_ROW + len(values)].copy_(high)


def fixture(rows, ratio, generator):
    original = torch.full((rows, 128), -1, dtype=torch.int32)
    compressed = torch.full((rows, 512), -1, dtype=torch.int32)
    for row in range(rows - 1):
        offset = (row % 16) * 1024
        position = 96 + (row * 17) % 900
        recent = torch.arange(max(0, position - 127), position + 1, dtype=torch.int32)
        original[row, :len(recent)] = recent + offset
        if ratio:
            legal_count = (position + 1) // ratio
            indices = torch.randperm(legal_count, generator=generator)[:512]
            compressed[row, :len(indices)] = indices + offset
    q = (torch.randn((rows, HEADS, WIDTH), generator=generator) * 0.2).bfloat16()
    sink = torch.linspace(-2.0, 3.0, HEADS)
    sink[1] = float('-inf')
    return q, original, compressed, sink


def oracle(q, ori, cmp, original, compressed, sink, selected):
    expected = []
    for row in selected.tolist():
        kv = torch.cat([ori[original[row][original[row] >= 0].long()],
                        cmp[compressed[row][compressed[row] >= 0].long()]], dim=0).double()
        if len(kv) == 0:
            expected.append(torch.zeros_like(q[row], dtype=torch.float64))
            continue
        scores = q[row].double() @ kv.T / WIDTH**0.5
        weights = torch.softmax(torch.cat([scores, sink.double()[:, None]], dim=1), dim=1)
        expected.append(weights[:, :-1] @ kv)
    return torch.stack(expected).float()


def compare(actual, reference):
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.003)
    return float((actual - reference).abs().max())


def verify_pair(results, reference, selected):
    baseline, candidate = (results[key].float().cpu() for key in ('baseline', 'two_heads'))
    assert torch.isfinite(baseline).all() and torch.isfinite(candidate).all()
    return {
        'candidate_vs_baseline_all_rows_max_abs_error': compare(candidate, baseline),
        'baseline_vs_FP64_sample_max_abs_error': compare(baseline[selected], reference),
        'candidate_vs_FP64_sample_max_abs_error': compare(candidate[selected], reference),
    }


def run_case(mode, rows, ratio, device, pools, host_pools, kernels, generator):
    q, oi, ci, sink = fixture(rows, ratio, generator)
    original, compressed = pools
    host_original, host_compressed = host_pools
    device_q, device_oi, device_ci, device_sink = (
        value.to(device) for value in (q, physical(oi), physical(ci), sink)
    )
    args = (device_q, original, compressed, device_oi, device_ci, device_sink)
    selected = torch.arange(rows) if rows <= 96 else torch.linspace(0, rows - 1, 32).long()
    for pool, data in zip(pools, host_pools):
        write_pool(pool, data)
    torch.npu.synchronize()

    eager = {}
    warmup_seconds = {key: [] for key in kernels}
    for key, kernel in kernels.items():
        for _ in range(2):
            started = time.perf_counter()
            eager[key] = kernel(*args)
            torch.npu.synchronize()
            warmup_seconds[key].append(time.perf_counter() - started)
    errors = verify_pair(eager, oracle(q, *host_pools, oi, ci, sink, selected), selected)
    del eager

    orders = [('baseline', 'two_heads', 'two_heads', 'baseline'),
              ('two_heads', 'baseline', 'baseline', 'two_heads')]
    timings = []
    for order in orders:
        for key in order:
            started = time.perf_counter()
            output = kernels[key](*args)
            torch.npu.synchronize()
            timings.append({'arm': key, 'synchronized_call_seconds': time.perf_counter() - started})
            del output

    graphs, captured = {}, {}
    for key, kernel in kernels.items():
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), auto_dispatch_capture=True):
            captured[key] = kernel(*args)
        graphs[key] = graph
    replays = []
    for index in range(3):
        changed_q = (torch.randn(q.shape, generator=generator) * (0.15 + 0.1 * index)).bfloat16()
        changed_original = (host_original.float() + 0.03125 * (index + 1)).bfloat16()
        changed_compressed = (host_compressed.float() - 0.015625 * (index + 1)).bfloat16()
        changed_oi, changed_ci = oi.clone(), ci.clone()
        changed_oi[:, index] = -1
        changed_ci[:, index] = -1
        changed_sink = sink + 0.25 * (index + 1)
        device_q.copy_(changed_q)
        device_oi.copy_(physical(changed_oi))
        device_ci.copy_(physical(changed_ci))
        device_sink.copy_(changed_sink)
        write_pool(original, changed_original)
        write_pool(compressed, changed_compressed)
        for graph in graphs.values():
            graph.replay()
        torch.npu.synchronize()
        reference = oracle(changed_q, changed_original, changed_compressed,
                           changed_oi, changed_ci, changed_sink, selected)
        replays.append({'errors': verify_pair(captured, reference, selected),
                        'output_sha256': {key: tensor_sha(value) for key, value in captured.items()}})
    for key in kernels:
        assert len({row['output_sha256'][key] for row in replays}) == 3

    return {'mode': mode, 'query_rows': rows, 'compression_ratio': ratio,
            'attention_heads': HEADS, 'FP64_checked_rows': selected.tolist(),
            'eager_errors': errors, 'warmup_seconds': warmup_seconds,
            'timing_samples': timings, 'changed_input_replays': replays}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--role', choices=SHAPES, required=True)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    assert args.output.is_absolute() and args.output.parent.is_dir() and not args.output.exists()
    baseline_path = args.baseline_dir.resolve(strict=True) / 'lowratio_sparse_attention.py'
    assert file_sha(baseline_path) == BASELINE_SHA256
    sys.path.insert(0, str(baseline_path.parent))
    import lowratio_sparse_attention as baseline
    import lowratio_sparse_attention_two_heads as candidate

    assert Path(baseline.__file__).resolve() == baseline_path
    assert Path(candidate.__file__).resolve().parent == Path(__file__).resolve().parent
    report = {'phase': 'initializing', 'device_id': args.device, 'role': args.role,
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'torch': torch.__version__,
              'torch_npu': getattr(torch_npu, '__version__', None),
              'baseline_source_sha256': BASELINE_SHA256, 'candidate_source_sha256': file_sha(Path(candidate.__file__)),
              'high_physical_row': HIGH_ROW,
              'cases': [], 'scope': 'Same-device sparse-attention component experiment. Times are synchronized calls.'}
    save(args.output, report)
    try:
        torch.set_num_threads(1)
        torch.npu.set_device(args.device)
        device = f'npu:{args.device}'
        generator = torch.Generator().manual_seed(1300)
        host_pools = tuple((torch.randn((HOST_KV_ROWS, WIDTH), generator=generator) * 0.2).bfloat16()
                           for _ in range(2))
        pools = tuple(torch.empty((HIGH_ROW + HOST_KV_ROWS, WIDTH), dtype=torch.bfloat16, device=device)
                      for _ in host_pools)
        for pool in pools:
            baseline.prepare_pool(pool)
        kernels = {'baseline': baseline.sparse_shared_kv, 'two_heads': candidate.sparse_shared_kv_two_heads}
        report.update(device=torch.npu.get_device_name(args.device),
                      pool_bytes_each=pools[0].numel() * pools[0].element_size(), phase='running')
        for mode, rows in SHAPES[args.role]:
            for ratio in (0, 1, 2):
                report['current_case'] = {'mode': mode, 'query_rows': rows, 'compression_ratio': ratio}
                save(args.output, report)
                case = run_case(mode, rows, ratio, device, pools, host_pools, kernels, generator)
                report['cases'].append(case)
        report.update(phase='component_passed', current_case=None)
        save(args.output, report)
    except BaseException as error:
        report.update(phase='failed_requires_review', error=repr(error))
        save(args.output, report)
        raise


if __name__ == '__main__':
    main()
