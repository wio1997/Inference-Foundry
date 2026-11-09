"""Attach original Engine/Worker entry points to externally loaded W8A8 ranks.

All ranks call ``attach_original_engine``. CP drivers receive the normal
LLMEngine; CP followers enter the normal Worker RPC loop. Drivers must drive
the original DP collectives together, as with the original offline API.
This module has no queue, schedule, sampling or model-loading implementation.
"""
import torch.distributed as dist

from deepinfer.engine.llm_engine import LLMEngine
from deepinfer.hugging_face import get_hf_tokenizer
from deepinfer.model_executor.model_loader import _get_entry_architecture
from deepinfer.model_executor.parallel_utils.parallel_state import PG
from deepinfer.worker.worker import Worker


def attach_original_engine(backend, engine_args):
    """Preflight data/providers on all ranks, then use original constructors."""
    if not dist.is_initialized():
        raise RuntimeError('initialize the existing resident rank before attaching its Engine')
    error, configs, tokenizer = None, None, None
    try:
        configs = engine_args.create_engine_configs()
        model_config, cache_config, parallel_config, scheduler_config = configs
        cp_size = parallel_config.context_parallel_size
        cp_rank = backend.runner.rank % cp_size
        if (parallel_config.engine_rank != backend.runner.rank // cp_size
                or parallel_config.world_size != dist.get_world_size()
                or cp_size != backend.runner.tp_size
                or parallel_config.data_parallel_rank != backend.runner.dp_rank):
            raise ValueError('EngineArgs parallel coordinates differ from the resident rank')
        backend.validate_model_config(model_config, scheduler_config)
        architecture = _get_entry_architecture(model_config)
        if cp_rank == 0:
            tokenizer = get_hf_tokenizer(model_config.tokenizer,
                trust_remote_code=model_config.trust_remote_code,
                tokenizer_revision=model_config.tokenizer_revision, revision=model_config.revision)
            if hasattr(architecture, 'multi_modal_mapper') and engine_args.engine_type.name != 'COMPLETION':
                from deepinfer.hugging_face import load_processor_config, build_mm_processor
                build_mm_processor(load_processor_config(model_config.model))
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    backend._bootstrap_vote(error, groups_initialized=PG.world is not None)
    if cp_rank == 0:
        backend.prepared_tokenizer = tokenizer
        return LLMEngine(*configs, log_stats=not engine_args.disable_log_stats,
            engine_type=engine_args.engine_type, offline_inference=engine_args.offline_inference,
            execution_backend=backend)
    return Worker(engine_args.engine_type, model_config, parallel_config, scheduler_config, cache_config,
        backend.local_rank, cp_rank, '', False, execution_backend=backend)
