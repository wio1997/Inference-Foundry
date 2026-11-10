# SPDX-License-Identifier: Apache-2.0
"""Bind the pinned reference TP runtime to the world initialized by Worker.

This does not turn a DeepInfer CP object into a reference TP object. Reference
GroupCoordinators are built through their real initialization functions, then
their membership and order are checked against Worker. Only a whole-world TP
deployment is supported here; subgroup/multi-model deployments are rejected.
"""
import torch.distributed as dist


def bind_worker_tensor_parallel(vllm_config, *, worker_group, local_rank):
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        get_dcp_group, get_dp_group, get_pcp_group, get_pp_group,
        get_tp_group, get_world_group, init_distributed_environment,
        initialize_model_parallel, model_parallel_is_initialized,
    )

    if not dist.is_initialized():
        raise RuntimeError('DeepInfer Worker must initialize torch.distributed first')
    if type(local_rank) is not int or local_rank < 0:
        raise ValueError('explicit process-visible local device rank required')
    ranks = dist.get_process_group_ranks(worker_group)
    if ranks != list(range(dist.get_world_size())):
        raise ValueError('reference runtime currently requires Worker to own the whole world')
    p = vllm_config.parallel_config
    if (p.tensor_parallel_size != len(ranks) or p.pipeline_parallel_size != 1
            or p.data_parallel_size != 1 or p.use_sequence_parallel_moe
            or p.prefill_context_parallel_size != 1
            or p.decode_context_parallel_size != 1
            or p.distributed_executor_backend != 'external_launcher'
            or vllm_config.speculative_config is not None):
        raise ValueError('requires external launcher TP=Worker world, PP/DP/PCP/DCP=1, no SP/speculation')
    backend = dist.get_backend(worker_group)
    with set_current_vllm_config(vllm_config):
        # Existing torch WORLD is reused by init_distributed_environment. It
        # additionally constructs the reference world coordinator, not a new
        # request executor or scheduler.
        init_distributed_environment(
            world_size=len(ranks), rank=dist.get_rank(), local_rank=local_rank,
            backend=backend)
        if not model_parallel_is_initialized():
            initialize_model_parallel(
                tensor_model_parallel_size=len(ranks),
                pipeline_model_parallel_size=1,
                prefill_context_model_parallel_size=1,
                decode_context_model_parallel_size=1, backend=backend)
    world, tp = get_world_group(), get_tp_group()
    if (list(world.ranks) != ranks or list(tp.ranks) != ranks
            or tp.world_size != len(ranks)
            or tp.rank_in_group != dist.get_rank(worker_group)):
        raise ValueError('existing reference world/TP differs from Worker membership/order')
    if any(group.world_size != 1 for group in
           (get_pp_group(), get_dp_group(), get_pcp_group(), get_dcp_group())):
        raise ValueError('existing reference parallel context has unsupported extra dimensions')
    return tp
