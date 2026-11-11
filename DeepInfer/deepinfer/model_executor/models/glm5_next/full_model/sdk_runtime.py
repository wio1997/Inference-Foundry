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
    request_block_hasher: object = None

    @property
    def max_context_tokens(self):
        return self.config.model_config.max_model_len

    @property
    def max_step_tokens(self):
        return self.config.scheduler_config.max_num_batched_tokens

    @property
    def vocab_size(self):
        return self.config.model_config.get_vocab_size()

    @property
    def num_speculative_tokens(self):
        return self.config.num_speculative_tokens

    def install(self, worker):
        from .sdk_schedule import SDKBatchSchedule
        from .sdk_dispatch import SDKBatchWorkerAdapter
        controller_kwargs = {}
        if self.num_speculative_tokens:
            from .sdk_async_mtp_schedule import SDKAsyncMTPBatchSchedule
            from .sdk_async_mtp_dispatch import SDKAsyncMTPBatchWorkerAdapter
            if worker.nextn != 1 or self.num_speculative_tokens != 1:
                raise ValueError('native and SDK MTP widths must both equal one')
            SDKBatchSchedule, SDKBatchWorkerAdapter = SDKAsyncMTPBatchSchedule, SDKAsyncMTPBatchWorkerAdapter
            controller_kwargs['num_lookahead_tokens'] = self.config.num_lookahead_tokens
        elif self.config.scheduler_config.async_scheduling:
            from .sdk_async_schedule import SDKAsyncBatchSchedule
            from .sdk_async_dispatch import SDKAsyncBatchWorkerAdapter
            SDKBatchSchedule, SDKBatchWorkerAdapter = SDKAsyncBatchSchedule, SDKAsyncBatchWorkerAdapter
        controller = None
        if worker.is_driver_worker:
            controller = SDKBatchSchedule(self.cache_manager,
                max_context=self.max_context_tokens, max_batch_tokens=self.max_step_tokens,
                max_requests=self.max_requests, vocab_size=self.vocab_size,
                num_cache_groups=len(self.scheduler_cache_config.kv_cache_groups),
                request_block_hasher=self.request_block_hasher,
                prefill_cache_block_size=self.config.cache_config.block_size, **controller_kwargs)
        self.adapter = SDKBatchWorkerAdapter(self, controller, self.native_group)
        worker.install_glm_single_request_adapter(self.adapter)
        worker._glm_worker_runtime = self

    def execute(self, descriptor):
        from vllm.config import set_current_vllm_config
        with set_current_vllm_config(self.config):
            output = self.compute_worker.execute_model(descriptor)
            if output is None:
                output = self.compute_worker.sample_tokens(None)
            if self.num_speculative_tokens:
                from .sdk_mtp_output import capture_mtp_output
                output = capture_mtp_output(self, output, descriptor)
            return output

    def drain(self):
        # Do not call SDKWorker.shutdown here: its distributed groups share
        # the native worker process. Native Worker lifecycle owns final exit.
        from vllm.model_executor.offloader.base import get_offloader
        import torch
        get_offloader().join_after_forward()
        torch.npu.synchronize(self.compute_worker.device)
        if getattr(self, "kda_capture", None) is not None:
            import json
            self.kda_capture.close()
            self.kda_capture_stats_path.write_text(json.dumps(self.kda_capture.snapshot())+"\n")

    @staticmethod
    def resolve_output(output):
        from vllm.v1.outputs import AsyncModelRunnerOutput
        from .sdk_mtp_output import PendingMTPOutput
        if isinstance(output, PendingMTPOutput):
            return output.resolve()
        return output.get_output() if isinstance(output, AsyncModelRunnerOutput) else output


def prepare_sdk_compute(model_dir, *, device, native_group, local_rank,
                        init_method, max_context=2048, max_step=1792,
                        max_requests=8, cache_bytes=2*1024**3, graph_decode=False, breakable_graph=False, kda_decode_capture=False,
                        prefix_caching=False, async_scheduling=False,
                        expert_parallel=False, shared_expert_overlap=False,
                        mtp_tokens=0):
    import torch
    import torch.distributed as dist
    from deepinfer.model_executor.models.glm5_next.full_model.reference_factory import (
        verify_installed_reference_sources, CHECKPOINT_PINS)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import set_current_vllm_config
    from vllm_ascend.worker.worker import NPUWorker
    from vllm.distributed.parallel_state import get_tp_group, get_pp_group, get_dp_group, get_ep_group
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_configs, generate_scheduler_kv_cache_config,
        resolve_kv_cache_block_sizes, get_max_concurrency_for_kv_cache_config)
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.model_executor.offloader.base import get_offloader, NoopOffloader

    if (not dist.is_initialized() or torch.device(device).type != 'npu'
            or min(max_context, max_step, max_requests, cache_bytes) < 1
            or max_step > max_context):
        raise ValueError('initialized native NPU worker and explicit positive bounds required')
    if mtp_tokens not in (0, 1) or (mtp_tokens and (breakable_graph or kda_decode_capture or not async_scheduling)):
        raise ValueError('initial MTP bridge requires one draft, SDK async sampling and no breakable capture')
    ranks = dist.get_process_group_ranks(native_group)
    if ranks != list(range(dist.get_world_size())):
        raise ValueError('candidate requires native Worker to own the whole TP world')
    verify_installed_reference_sources()
    root = Path(model_dir).resolve(strict=True)
    for name, expected in CHECKPOINT_PINS.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('checkpoint identity mismatch: '+name)
    additional_config = {'enable_dsa_cp': False}
    if shared_expert_overlap:
        additional_config['multistream_overlap_shared_expert'] = True
    compilation = ({'mode': 0, 'cudagraph_mode': ('PIECEWISE' if breakable_graph else 'FULL_DECODE_ONLY'),
        'cudagraph_capture_sizes': [1, 2, 4, 8]} if graph_decode else {})
    extra_args = {}
    if mtp_tokens:
        additional_config['enable_flashcomm1'] = False
        extra_args.update(quantization='ascend', speculative_config={
            'method': 'deepseek_mtp', 'num_speculative_tokens': 1, 'enforce_eager': True})
        if graph_decode:
            compilation = {'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [2, 4, 8, 16]}
        from .sdk_mtp_norm_contract import install_mtp_norm_contract
        install_mtp_norm_contract('d2fda42ad99935725c98b9505a8d2e94696d9cdb2906e5cb2f1e37bf21fb27bc')
    args = EngineArgs(model=str(root), tensor_parallel_size=len(ranks),
        pipeline_parallel_size=1, data_parallel_size=1,
        enable_expert_parallel=expert_parallel, distributed_executor_backend='external_launcher',
        dtype='bfloat16', seed=0, enforce_eager=not graph_decode,
        compilation_config=compilation,
        max_model_len=max_context, max_num_seqs=max_requests,
        max_num_batched_tokens=max_step, enable_chunked_prefill=True,
        enable_prefix_caching=prefix_caching, block_size=128, mamba_block_size=128,
        mamba_cache_mode='align' if prefix_caching else 'none', additional_config=additional_config,
        async_scheduling=async_scheduling, skip_mm_profiling=True,
        kv_cache_memory_bytes=cache_bytes, cpu_offload_gb=0,
        offload_backend='auto', offload_group_size=0,
        trust_remote_code=False, disable_log_stats=True, **extra_args)
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
        ep = get_ep_group()
        ascend_config = get_ascend_config()
        if expert_parallel and (ep.world_size != len(ranks) or list(ep.ranks) != ranks):
            raise ValueError('SDK expert group differs from the owned native TP8 world')
        if shared_expert_overlap and not ascend_config.multistream_overlap_shared_expert:
            raise ValueError('requested shared expert overlap was disabled by SDK config')
        worker.load_model()
        moe_modules = [module for module in worker.model_runner.get_model().modules()
                       if type(module).__name__ == 'Glm5NextMoE']
        shared_profile = {
            'expert_parallel': config.parallel_config.enable_expert_parallel,
            'expert_group_size': ep.world_size,
            'expert_group_ranks': list(ep.ranks),
            'moe_group_sizes': sorted({module.ep_size for module in moe_modules}),
            'moe_local_expert_counts': sorted({module.n_local_physical_experts for module in moe_modules}),
            'shared_expert_overlap': ascend_config.multistream_overlap_shared_expert,
            'fused_mc2': ascend_config.enable_fused_mc2,
            'compilation_mode': str(config.compilation_config.mode),
            'quantization_type': type(config.quant_config).__name__,
            'ascend_graph_config': {
                name: getattr(ascend_config.ascend_compilation_config, name)
                for name in ('enable_npugraph_ex', 'enable_static_kernel', 'enable_super_kernel')},
        }
        if expert_parallel and (not moe_modules or shared_profile['moe_group_sizes'] != [len(ranks)]):
            raise ValueError('GLM expert modules did not use the requested EP group')
        kda_capture = None
        if mtp_tokens:
            from .sdk_mtp_observer import install_mtp_observer
            kda_capture = install_mtp_observer(config, worker.model_runner, max_requests=max_requests)
        elif kda_decode_capture:
            if not (graph_decode and breakable_graph):
                raise ValueError("KDA capture requires PIECEWISE graph mode")
            from .kda_graph_capture import install_kda_decode_capture
            kda_capture = install_kda_decode_capture(config, worker.model_runner, max_requests=max_requests)
        elif graph_decode and not breakable_graph:
            # Share the reference arm's exact-decode FULL replay contract.
            # The original SDK still owns all model math and metadata.
            from .full_graph_observer import install_full_graph_observer
            kda_capture = install_full_graph_observer(config, worker.model_runner,
                max_requests=max_requests)
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
        if mtp_tokens:
            snapshot = kda_capture.snapshot()
            if not snapshot['draft_enforce_eager'] or snapshot['draft_use_cuda_graph']:
                raise RuntimeError('shared MTP draft eager configuration did not take effect')
            shared_profile['mtp'] = snapshot
            # Configured policy; request-stage counters must separately prove
            # actual overlap. A True flag alone is not a mechanism result.
            shared_profile['native_mtp_host_output_overlap'] = bool(async_scheduling)
        if graph_decode:
            from vllm.config.compilation import CUDAGraphMode
            if (config.model_config.enforce_eager or
                    config.compilation_config.cudagraph_mode != (CUDAGraphMode.PIECEWISE if breakable_graph else CUDAGraphMode.FULL_DECODE_ONLY)):
                raise RuntimeError("requested decode graph fell back; do not score as graph")
        scheduler_block, hash_block = resolve_kv_cache_block_sizes(scheduler_cache, config)
        manager = None
        request_block_hasher = None
        if rank == 0:
            if prefix_caching:
                from vllm.utils.hashing import get_hash_fn_by_name
                from vllm.v1.core.kv_cache_utils import init_none_hash, get_request_block_hasher
                hash_fn = get_hash_fn_by_name(config.cache_config.prefix_caching_hash_algo)
                init_none_hash(hash_fn)
                request_block_hasher = get_request_block_hasher(hash_block, hash_fn)
            speculative = config.speculative_config
            use_eagle = speculative is not None and speculative.use_eagle_block_drop()
            prefill_lookahead = (config.num_speculative_tokens if speculative.use_multi_module_mtp() else 1) if speculative is not None and speculative.use_eagle() else 0
            manager = KVCacheManager(kv_cache_config=scheduler_cache,
                max_model_len=max_context, scheduler_block_size=scheduler_block,
                hash_block_size=hash_block, max_in_flight_tokens=config.max_in_flight_tokens,
                enable_caching=prefix_caching, use_eagle=use_eagle, num_prefill_lookahead=prefill_lookahead,
                log_stats=False, dcp_world_size=1, pcp_world_size=1,
                watermark=config.scheduler_config.watermark,
                enable_mamba_fine_grained_prefix_cache=False)
    result = SDKComputeRuntime(config, worker, physical, scheduler_cache,
                             manager, max_requests, all_memory, native_group)
    result.kda_capture = kda_capture
    result.shared_profile = shared_profile
    result.request_block_hasher = request_block_hasher
    return result
