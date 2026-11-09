"""Original Worker/ModelRunner classes with the existing NPU execution resources."""
from contextlib import contextmanager
import time
import torch


class CpuTimer:
    def __init__(self):
        self.samples = []

    @contextmanager
    def record(self, name):
        start = time.monotonic_ns()
        try:
            yield
        finally:
            self.samples.append((name, start, time.monotonic_ns()))


class NativeEventRegistry:
    def __init__(self):
        self.latest = {}

    def set(self, name, pair):
        self.latest[name] = pair

    def latest_event_pair(self, name):
        return self.latest.get(name)

    def flush_out(self, *args, **kwargs):
        # Native graph events are retained by the original WorkerStepState.
        return None


def create_worker(model_runner, callbacks, flags, registry, *, is_driver, nextn=5,
                  backend=None, gc_interval=10000, gc_step=0, gc_backend=None,
                  cache_config=None, model_config=None):
    from deepinfer.worker.model_runner import ModelRunner
    from deepinfer.worker.worker import Worker
    from deepinfer.worker.cache_engine import CacheEngine

    backend = torch if backend is None else backend
    runner = ModelRunner.from_execution_backend(model_runner, is_driver_worker=is_driver, nextn=nextn)
    storage = getattr(model_runner, 'cache_storage', None)
    cache_engine = (None if storage is None else
                    CacheEngine.from_execution_storage(storage, cache_config=cache_config))
    timer = CpuTimer()
    worker = Worker.from_execution_backend(runner, callbacks, is_driver_worker=is_driver,
        nextn=nextn, options=flags, cpu_timer_backend=timer, device_timer_backend=registry,
        graph_timer_backend=registry, event_factory=backend.npu.Event, gc_backend=gc_backend,
        gc_interval=gc_interval, gc_step=gc_step, cache_engine=cache_engine,
        cache_config=cache_config, model_config=model_config,
        global_rank=getattr(model_runner.runner, 'rank', None) if hasattr(model_runner, 'runner') else None)
    # Keep existing per-session phase timing while using the real Worker class.
    worker._local_cpu_timer = timer
    worker._post_process_step = timer.record("worker.post_process")(worker._post_process_step)
    return worker
