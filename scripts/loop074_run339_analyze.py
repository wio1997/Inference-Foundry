#!/usr/bin/env python3
"""Parse three independent CANN 9.1.0 HCCL Test TP8 payload runs."""

import argparse
import json
import re
import statistics
from pathlib import Path

CASES = {
    'ag_hidden_bf16': {'binary': 'all_gather_test', 'test_bytes': 720896,
                       'api_input_bytes': 90112, 'dtype': 'bfp16', 'count_per_forward': 90,
                       'note': '47 wrapped plus 43 source-inferred DSA AG'},
    'ag_router_fp32': {'binary': 'all_gather_test', 'test_bytes': 90112,
                       'api_input_bytes': 11264, 'dtype': 'fp32', 'count_per_forward': 43},
    'ag_extra_bf16': {'binary': 'all_gather_test', 'test_bytes': 2883584,
                      'api_input_bytes': 360448, 'dtype': 'bfp16', 'count_per_forward': 1},
    'rs_bf16': {'binary': 'reduce_scatter_test', 'test_bytes': 720896,
                'api_input_bytes': 720896, 'dtype': 'bfp16', 'count_per_forward': 87},
    'a2a_bf16': {'binary': 'alltoall_test', 'test_bytes': 720896,
                 'api_input_bytes': 720896, 'dtype': 'bfp16', 'count_per_forward': 43},
}
LINE = re.compile(r'^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\w+)\s*$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    root = args.run_dir
    repeats = (root, root / 'repeat2', root / 'repeat3')
    per_case = {name: [] for name in CASES}
    for idx, rep in enumerate(repeats, 1):
        status = {}
        for line in (rep / 'cases.tsv').read_text().splitlines():
            name, binary, size, dtype, code = line.split('\t')
            status[name] = (binary, int(size), dtype, int(code))
        if set(status) != set(CASES):
            raise RuntimeError(f'repeat{idx} missing or extra case')
        for name, meta in CASES.items():
            if status[name] != (meta['binary'], meta['test_bytes'], meta['dtype'], 0):
                raise RuntimeError(f'repeat{idx} {name} configuration/exit failed')
            raw = (rep / f'{name}.log').read_text()
            if f'the minbytes is {meta["test_bytes"]}, maxbytes is {meta["test_bytes"]}, iters is 30, warmup_iters is 5' not in raw:
                raise RuntimeError(f'repeat{idx} {name} header mismatch')
            rows = [LINE.match(line) for line in raw.splitlines()]
            rows = [x for x in rows if x is not None]
            if len(rows) != 1 or int(rows[0][1]) != meta['test_bytes'] or rows[0][4] != 'success':
                raise RuntimeError(f'repeat{idx} {name} result invalid')
            per_case[name].append({'repeat': idx, 'test_reported_bytes': int(rows[0][1]),
                                   'device_time_us': float(rows[0][2]),
                                   'test_algorithm_bandwidth_GBps': float(rows[0][3])})
    output = {
        'status': 'complete_three_independent_hccl_test_runs',
        'environment': {'npu': '8x Ascend 910B3', 'driver': '26.0.rc1',
                        'cann': '9.1.0', 'mpi': 'Open MPI 4.1.2',
                        'HCCL_BUFFSIZE': 1024, 'TASK_QUEUE_ENABLE': 1,
                        'HCCL_OP_EXPANSION_MODE': 'AIV', 'HCCL_INTRA_ROCE_ENABLE': 1,
                        'HCCL_RDMA_CONNECT_TIMEOUT': 17, 'device_time_flag': 1,
                        'warmup_iters': 5, 'measured_iters': 30,
                        'result_check': 1},
        'scope': 'isolated same-topology/dtype/size HCCL Test service, not model DAG or physical link floor',
        'cases': {}, 'limits': [
            'HCCL Test `data_size` equals full AG output, full RS input, and A2A input/output; not universally per-rank API input.',
            'Tool may choose a different communication algorithm or buffer lifecycle from the Extreme Runtime.',
            'No producer-arrival, model compute/HBM contention, Graph scheduling, 264-op chain or Product E2E included.',
            'Do not add isolated service medians and call the sum a critical-path or hardware lower bound.',
            'Run337 wrapper omitted 43 AG; their count/size attribution is source and profiled-sequence inference, not captured by Run337 Host wrapper.',
        ]}
    for name, meta in CASES.items():
        vals = [x['device_time_us'] for x in per_case[name]]
        output['cases'][name] = {**meta, 'repeats': per_case[name],
                                 'min_us': min(vals), 'median_us': statistics.median(vals),
                                 'max_us': max(vals)}
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
