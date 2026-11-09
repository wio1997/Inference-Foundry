"""Original Worker resources over the already loaded W8A8 model; no scheduler."""
import torch
import torch.distributed as dist
from pathlib import Path
import numpy as np
import os

from deepinfer.model_executor.models.w8a8_cache_storage910c import W8A8CacheStorage
from deepinfer.engine.zos910c.official_phase_core910c import NativeEventRegistry


class OriginalDeviceWork:
    """Protect graph/cache addresses until the latest original device work ends."""
    def __init__(self, event_factory):
        self.event_factory = event_factory
        self.event = None

    def record(self):
        if self.event is None:
            self.event = self.event_factory()
        self.event.record()

    def drain(self):
        if self.event is not None:
            self.event.synchronize()
            self.event = None


class ExistingW8A8Resources:
    def __init__(self, runner, registry=None):
        self.runner = runner
        self.model = runner.model
        self.registry = NativeEventRegistry() if registry is None else registry
        self.cache_storage = W8A8CacheStorage.from_model(self.model)
        self.worker = None
        self.device_work = None
        self.engram_lookup_gpu = None

    def record_device_work(self):
        if self.device_work is not None:
            self.device_work.record()

    def _bootstrap_vote(self, error, *, groups_initialized=None):
        # Use the already initialized default group. A local validation error
        # must reach every rank before any rank creates original subgroups.
        device = 'cpu' if str(dist.get_backend()).lower() == 'gloo' else self.model.device
        values = [int(error is not None)]
        if groups_initialized is not None:
            values.append(int(groups_initialized))
        failed = torch.tensor(values, dtype=torch.int32, device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.SUM)
        mixed_groups = groups_initialized is not None and int(failed[1].item()) not in (0, dist.get_world_size())
        if failed[0].item() or mixed_groups:
            raise RuntimeError(f'original resource bootstrap rejected on a rank: {error or "peer validation failed"}')

    def warm_up_model(self, *, capture_max_graph_only=False):
        return self.runner.warm_up_model(capture_max_graph_only=capture_max_graph_only)


    @property
    def local_rank(self):
        index = self.model.device.index
        if index is None:
            raise ValueError('initialized backend requires an explicit local device index')
        return index


    def worker_runtime(self):
        from deepinfer.engine.zos910c.official_phase_core910c import CpuTimer
        return dict(cpu_timer_backend=CpuTimer(), device_timer_backend=self.registry,
                    graph_timer_backend=self.registry, event_factory=torch.npu.Event)


    def validate_model_config(self, model_config, scheduler_config):
        """Verify original config against already loaded checkpoint resources."""
        reader = getattr(self.model, 'reader', None)
        if reader is None:
            # Synthetic control fixtures have no checkpoint or model weights.
            return
        if Path(model_config.model).resolve() != Path(reader.root).resolve():
            raise ValueError('original ModelConfig and resident checkpoint directory differ')
        config = model_config.model_cfg.gpt
        for key in ('vocab_size','hidden_size','num_hidden_layers','num_attention_heads',
                    'head_dim','q_lora_rank','o_lora_rank','o_groups'):
            if getattr(config, key) != self.model.cfg[key]:
                raise ValueError(f'original ModelConfig {key} differs from resident model')
        if model_config.runtime_max_model_len > self.model.max_tokens:
            raise ValueError('original runtime positions exceed resident cache capacity')
        if scheduler_config.max_num_batched_tokens > self.model.chunk_size:
            raise ValueError('original prefill budget exceeds resident chunk capacity')


    def initialize_engram(self, tokenizer_path):
        """Bind original CPU/C++ lookup to the existing immutable vocabulary."""
        from deepinfer import envs
        from deepinfer.model_executor.models.v4_modules.engram import CompressedTokenizer
        histories = getattr(self.model, 'histories', ())
        history = histories[0] if histories else None
        if history is not None and hasattr(history, 'token_map'):
            if Path(tokenizer_path).resolve() != (Path(self.model.reader.root)/'tokenizer.json').resolve():
                raise ValueError('original Engram tokenizer differs from resident vocabulary')
            table = history.token_map
            array = history.token_lookup.numpy()
        else:
            from deepinfer.model_executor.models.v4_modules.engram_metadata import CompressedVocabulary
            vocabulary = CompressedVocabulary(tokenizer_path)
            table, array = vocabulary.old2new, vocabulary.lookup_table
        if len(table) != self.model.cfg['vocab_size'] or array.dtype != np.int32:
            raise ValueError('original compressed tokenizer shape/dtype differs from resident vocabulary')
        pad = table[envs.DEEPINFER_ENGRAM_PAD_TOKEN]
        if history is not None and hasattr(history, 'pad_id') and history.pad_id != pad:
            raise ValueError('original Engram pad token differs from resident history')
        if self.engram_lookup_gpu is None:
            self.engram_lookup_gpu = torch.from_numpy(array).to(device=self.model.device)
        CompressedTokenizer.lookup_table = table
        CompressedTokenizer.lookup_table_np = array
        CompressedTokenizer.lookup_table_gpu = self.engram_lookup_gpu
        CompressedTokenizer.compressed_pad_token = pad

    def publish_ready(self, worker, control_dir):
        from deepinfer import envs
        from deepinfer.entrypoints.file_transport import PROTOCOL, atomic_json, process_birth, source_identity
        error, receipt = None, None
        try:
            if self.worker is not worker or self.model.device.type != 'npu':
                raise ValueError('ready requires the initialized original NPU Worker owner')
            pool = getattr(self, 'hidden_graphs', None)
            draft_pool = getattr(getattr(self, 'mtp_module', None), 'decode_graph_runner', None)
            directory = Path(control_dir)
            directory.mkdir(parents=True, exist_ok=True)
            receipt = dict(protocol=PROTOCOL, state='READY', source=source_identity(),
                rank=worker.global_rank, pid=os.getpid(), starttime=process_birth(os.getpid()),
                dp_rank=worker.parallel_config.data_parallel_rank,
                data_parallel_size=worker.parallel_config.data_parallel_size,
                context_parallel_size=worker.parallel_config.context_parallel_size,
                world_size=worker.parallel_config.world_size,
                max_model_len=worker.model_config.max_model_len,
                runtime_max_model_len=worker.model_config.runtime_max_model_len,
                vocab_size=worker.model_config.get_vocab_size(),
                cache=self.cache_storage.cache_parameters(),
                graph_sizes=[] if pool is None else pool.batch_sizes,
                draft_graph_sizes=[] if draft_pool is None else draft_pool.batch_sizes,
                decode_perf_curve=worker.decode_perf_curve,
                capabilities=dict(vision=bool(getattr(self, 'supports_multimodal', False)),
                                  eplb=getattr(self.model, 'eplb_runtime', None) is not None,
                                  batch_invariant=bool(envs.DEEPINFER_BATCH_INVARIANT
                                      and getattr(self, 'supports_batch_invariant', False))),
                nextn=worker.nextn,
                execution_options=dict(yoco_mode=worker.model_runner.yoco_prefill_truncation_mode,
                    yoco_keep_tokens=envs.DEEPINFER_YOCO_PREFILL_KEEP_TOKENS,
                    mtp_varlen=bool(worker.execution_options.DEEPINFER_ENABLE_MTP_VARLEN),
                    zos=bool(worker.execution_options.DEEPINFER_ZERO_OVERHEAD_SCHEDULING),
                    enforce_eager=bool(envs.DEEPINFER_ENFORCE_EAGER)),
                seed=worker.model_config.seed+worker.global_rank)
            atomic_json(directory/f'ready-rank{worker.global_rank}.json', receipt)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        # Every receipt is written after normal warmup/seed reset. The vote
        # returns only after all original ranks have published or rejected it.
        self._bootstrap_vote(error)
        return receipt


    def initialize_worker(self, worker):
        from deepinfer import envs
        from deepinfer.model_executor.parallel_utils.parallel_state import PG, init_parallel_groups

        if not dist.is_initialized():
            raise RuntimeError('existing backend default process group must be initialized')
        config = worker.parallel_config
        error = None
        try:
            if getattr(self, 'worker', None) is not None and self.worker is not worker:
                raise ValueError('initialized backend already belongs to another Worker')
            self.cache_storage._assert_drained()
            self.validate_model_config(worker.model_config, worker.scheduler_config)
            if (dist.get_rank() != worker.global_rank or self.runner.rank != worker.global_rank
                    or dist.get_world_size() != config.world_size
                    or self.runner.tp_size != config.context_parallel_size
                    or self.runner.dp_rank != config.data_parallel_rank):
                raise ValueError('original Worker parallel coordinates differ from existing backend')
            nextn = 0 if self.model.dspark is None else self.model.dspark.nextn
            if worker.nextn != nextn:
                raise ValueError('original Worker nextn differs from existing draft resources')
            if worker.scheduler_config.max_num_seqs > self.cache_storage.num_state_seqs - 1:
                raise ValueError('original Scheduler exceeds existing physical state capacity')
            if envs.DEEPINFER_BATCH_INVARIANT and not getattr(self, 'supports_batch_invariant', False):
                raise NotImplementedError('batch-invariant compute interfaces still require backend adaptation')
            runtime = getattr(self.model, 'eplb_runtime', None)
            if runtime is not None:
                runtime.validate_original_worker(worker)
            if self.model.device.type == 'npu' and torch.npu.current_device() != self.local_rank:
                raise ValueError('current device differs from existing backend cache device')
            if PG.world is not None and (
                    PG.world.size() != config.world_size or PG.get_rank() != worker.global_rank
                    or PG.context_parallel_group().size() != config.context_parallel_size
                    or PG.data_parallel_group().size() != config.data_parallel_size
                    or PG.expert_parallel_group().size() != config.expert_parallel_size
                    or PG.engram_parallel_group().size() != config.engram_parallel_size):
                raise ValueError('existing original process groups differ from backend topology')
        except Exception as exc:
            error = str(exc)
        self._bootstrap_vote(error, groups_initialized=PG.world is not None)
        if PG.world is None:
            init_parallel_groups(config.data_parallel_size, ep_size=config.expert_parallel_size,
                                 cp_size=config.context_parallel_size, enp_size=config.engram_parallel_size)
        if self.device_work is None and self.model.device.type == 'npu':
            self.device_work = OriginalDeviceWork(torch.npu.Event)
            self.model._original_device_work = self.device_work
        worker.device = self.model.device
        self.worker = worker
        if runtime is not None:
            runtime.attach_original_worker(worker)


    def destroy_cuda_graphs(self):
        if self.device_work is not None:
            self.device_work.drain()
        self.runner.destroy_cuda_graphs()
        runtime = getattr(self.model, 'eplb_runtime', None)
        if runtime is not None and self.worker is not None and runtime.original_worker is self.worker:
            runtime.detach_original_worker(self.worker)
        if self.engram_lookup_gpu is not None:
            from deepinfer.model_executor.models.v4_modules.engram import CompressedTokenizer
            if CompressedTokenizer.lookup_table_gpu is self.engram_lookup_gpu:
                CompressedTokenizer.release()
            self.engram_lookup_gpu = None
        self.worker = None
