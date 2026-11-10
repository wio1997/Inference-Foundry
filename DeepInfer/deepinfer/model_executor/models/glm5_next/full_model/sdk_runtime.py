"""Experimental compute-only SDK worker behind the native DeepInfer Worker.

No vLLM engine, request scheduler, HTTP frontend or executor is created here.
The existing native process group and its device claim remain authoritative.
Not deployed or hardware-validated yet.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib


@dataclass
class SDKComputeRuntime:
    config: object
    compute_worker: object
    physical_cache_config: object
    scheduler_cache_config: object
    cache_manager: object
    max_requests: int
    memory_bytes_per_rank: list
    native_group: object

    @property
    def max_context_tokens(self):
        return self.config.model_config.max_model_len

    @property
    def max_step_tokens(self):
        return self.config.scheduler_config.max_num_batched_tokens

    @property
    def vocab_size(self):
        return self.config.model_config.get_vocab_size()

    def install(self, worker):
        from .sdk_schedule import SDKBatchSchedule
        from .sdk_dispatch import SDKBatchWorkerAdapter
        controller = None
        if worker.is_driver_worker:
            controller = SDKBatchSchedule(self.cache_manager,
                max_context=self.max_context_tokens, max_batch_tokens=self.max_step_tokens,
                max_requests=self.max_requests, vocab_size=self.vocab_size,
                num_cache_groups=len(self.scheduler_cache_config.kv_cache_groups))
        self.adapter = SDKBatchWorkerAdapter(self, controller, self.native_group)
        worker.install_glm_single_request_adapter(self.adapter)
        worker._glm_worker_runtime = self

    def execute(self, descriptor):
        from vllm.config import set_current_vllm_config
        with set_current_vllm_config(self.config):
            output = self.compute_worker.execute_model(descriptor)
            if output is None:
                output = self.compute_worker.sample_tokens(None)
            return output

    def drain(self):
        # Do not call SDKWorker.shutdown here: its distributed groups share
        # the native worker process. Native Worker lifecycle owns final exit.
        from vllm.model_executor.offloader.base import get_offloader
        import torch
        get_offloader().join_after_forward()
        torch.npu.synchronize(self.compute_worker.device)


def prepare_sdk_compute(model_dir, *, device, native_group, local_rank,
                        init_method, max_context=2048, max_step=1792,
                        max_requests=8, cache_bytes=2*1024**3, graph_decode=False, breakable_graph=False):
    import torch
    import torch.distributed as dist
    from deepinfer.model_executor.models.glm5_next.full_model.reference_factory import (
        verify_installed_reference_sources, CHECKPOINT_PINS)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import set_current_vllm_config
    from vllm_ascend.worker.worker import NPUWorker
    from vllm.distributed.parallel_state import get_tp_group, get_pp_group, get_dp_group
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_configs, generate_scheduler_kv_cache_config,
        resolve_kv_cache_block_sizes, get_max_concurrency_for_kv_cache_config)
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.model_executor.offloader.base import get_offloader, NoopOffloader

    if (not dist.is_initialized() or torch.device(device).type != 'npu'
            or min(max_context, max_step, max_requests, cache_bytes) < 1
            or max_step > max_context):
        raise ValueError('initialized native NPU worker and explicit positive bounds required')
    ranks = dist.get_process_group_ranks(native_group)
    if ranks != list(range(dist.get_world_size())):
        raise ValueError('candidate requires native Worker to own the whole TP world')
    verify_installed_reference_sources()
    root = Path(model_dir).resolve(strict=True)
    for name, expected in CHECKPOINT_PINS.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('checkpoint identity mismatch: '+name)
    args = EngineArgs(model=str(root), tensor_parallel_size=len(ranks),
        pipeline_parallel_size=1, data_parallel_size=1,
        enable_expert_parallel=False, distributed_executor_backend='external_launcher',
        dtype='bfloat16', seed=0, enforce_eager=not graph_decode,
        compilation_config=({'mode': 0, 'cudagraph_mode': ('PIECEWISE' if breakable_graph else 'FULL_DECODE_ONLY'),
            'cudagraph_capture_sizes': [1, 2, 4, 8]} if graph_decode else {}),
        max_model_len=max_context, max_num_seqs=max_requests,
        max_num_batched_tokens=max_step, enable_chunked_prefill=True,
        enable_prefix_caching=False, block_size=128, mamba_block_size=128,
        mamba_cache_mode='none', additional_config={'enable_dsa_cp': False},
        async_scheduling=False, skip_mm_profiling=True,
        kv_cache_memory_bytes=cache_bytes, cpu_offload_gb=0,
        offload_backend='auto', offload_group_size=0,
        trust_remote_code=False, disable_log_stats=True)
    config = args.create_engine_config()
    rank = dist.get_rank(native_group)
    with set_current_vllm_config(config):
        worker = NPUWorker(vllm_config=config, local_rank=local_rank,
            rank=rank, distributed_init_method=init_method,
            is_driver_worker=rank==0)
        worker.init_device()
        tp = get_tp_group()
        if (list(tp.ranks) != ranks or tp.rank_in_group != rank
                or get_pp_group().world_size != 1 or get_dp_group().world_size != 1):
            raise ValueError('SDK compute TP membership differs from native Worker')
        worker.load_model()
        if not isinstance(get_offloader(), NoopOffloader):
            raise ValueError('candidate requires full NPU residency')
        model = worker.get_model()
        if any(p.numel() and p.device.type != 'npu' for p in model.parameters()):
            raise ValueError('nonempty model parameter outside NPU')
        spec = worker.get_kv_cache_spec()
        available = int(worker.determine_available_memory())
        # This one-off CPU exchange uses the native Gloo world, not a new
        # scheduler/executor. Each real worker contributes its own cache spec.
        all_specs = [None]*len(ranks)
        all_memory = [None]*len(ranks)
        dist.all_gather_object(all_specs, spec, group=dist.group.WORLD)
        dist.all_gather_object(all_memory, available, group=dist.group.WORLD)
        cache_configs = get_kv_cache_configs(config, all_specs, all_memory)
        physical = cache_configs[rank]
        scheduler_cache = generate_scheduler_kv_cache_config(cache_configs)
        capacity = get_max_concurrency_for_kv_cache_config(config, scheduler_cache)
        if capacity < max_requests:
            raise ValueError('physical cache cannot honor the declared native sequence capacity')
        worker.initialize_from_config(physical)
        worker.compile_or_warm_up_model()
        if graph_decode:
            from vllm.config.compilation import CUDAGraphMode
            if (config.model_config.enforce_eager or
                    config.compilation_config.cudagraph_mode != (CUDAGraphMode.PIECEWISE if breakable_graph else CUDAGraphMode.FULL_DECODE_ONLY)):
                raise RuntimeError("requested decode graph fell back; do not score as graph")
        scheduler_block, hash_block = resolve_kv_cache_block_sizes(scheduler_cache, config)
        manager = None
        if rank == 0:
            manager = KVCacheManager(kv_cache_config=scheduler_cache,
                max_model_len=max_context, scheduler_block_size=scheduler_block,
                hash_block_size=hash_block, max_in_flight_tokens=config.max_in_flight_tokens,
                enable_caching=False, use_eagle=False, num_prefill_lookahead=0,
                log_stats=False, dcp_world_size=1, pcp_world_size=1,
                watermark=config.scheduler_config.watermark,
                enable_mamba_fine_grained_prefix_cache=False)
    return SDKComputeRuntime(config, worker, physical, scheduler_cache,
                             manager, max_requests, all_memory, native_group)
