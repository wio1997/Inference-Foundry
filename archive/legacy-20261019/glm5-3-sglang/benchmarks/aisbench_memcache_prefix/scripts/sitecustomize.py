"""Scoped to this benchmark's PYTHONPATH; no installed sources are changed."""
import os
if 'AIS_BENCH_REQUEST_TIMEOUT' in os.environ:
    from ais_bench.benchmark import global_consts
    global_consts.REQUEST_TIME_OUT = int(os.environ['AIS_BENCH_REQUEST_TIMEOUT'])
