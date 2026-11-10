# SPDX-License-Identifier: Apache-2.0
"""Explicit assembly for the bounded four-rank GLM Worker adapter.

The caller supplies the DSA operations, their binding, installed sampler,
loaded operator libraries and all memory/context bounds. No backend is selected
implicitly. This does not register a model, allocate a serving cache, start a
Worker, or claim model quality or production readiness.
"""
from dataclasses import dataclass

import torch
import torch.distributed as dist

from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.parallel_utils.parallel_state import PG

from .native_model import load_native_text_runner
from .rank_text_install import install_rank_text_seams
from .worker_bridge import SingleRequestWorkerBridge
from .worker_dispatch import GLMSingleRequestWorkerAdapter
from .worker_initial_state import initial_worker_text_state


@dataclass(frozen=True)
class BoundedWorkerRuntime:
    runner: object
    adapter: GLMSingleRequestWorkerAdapter
    rank_install: object

    def install(self, worker):
        """Install on an explicitly prepared Worker; caller owns lifetime."""
        worker.install_glm_single_request_adapter(self.adapter)


@torch.inference_mode()
def prepare_bounded_worker_runtime(reader, device, *, dsa_operations, bind_dsa, sampler,
        max_expert_checkpoint_bytes, max_expert_resident_bytes, max_context_tokens, max_step_tokens,
        rank_cache_bytes, max_assembled_bytes, max_intermediate_rows, max_packed_rows,
        hc_backend, kda_backend, shard_dense_down=False, dense_activation_backend='torch_fp32'):
    """Build one unpaged, single-request text runtime using explicit components.

    All four CP ranks must call with corresponding rank-local inputs, in the
    same order, after process-group setup. The supplied ``bind_dsa`` returns
    a callable operation for each configured DSA layer. Known failures before
    rank sharding are agreed; device/process loss and errors inside component
    collectives are not recovered. The caller must release active request owners
    and the process group; no system resources or services are managed here.
    """
    cp=PG.context_parallel_group()
    if not dist.is_initialized() or cp.size()!=4 or dist.get_world_size(cp)!=4:
        raise ValueError('initialized four-rank CP group required')
    rank=cp.rank();failure=None;runner=None;config=None
    try:
        if type(rank) is not int or not 0<=rank<4 or not callable(bind_dsa):
            raise ValueError('explicit rank and DSA binding callback required')
        if rank==0 and sampler is None:
            raise ValueError('driver requires an explicit installed sampler')
        if hc_backend!='ascend_installed' or kda_backend not in ('ascendc','ascendc_installed_decode','ascendc_chunk_prefill'):
            raise ValueError('explicit supported native mHC and KDA backends required')
        if type(max_step_tokens) is not int or not 1<=max_step_tokens<=8:
            raise ValueError('explicit step bound must be 1..8')
        config=Glm5NextConfig.from_checkpoint(reader.root)
        runner=load_native_text_runner(reader,device,dsa_operations=dsa_operations,
            max_expert_checkpoint_bytes=max_expert_checkpoint_bytes,max_expert_resident_bytes=max_expert_resident_bytes,
            vocab_start=0,vocab_stop=config.vocab_size,max_head_rows=config.vocab_size,
            max_tokens=max_step_tokens,max_context_tokens=max_context_tokens,logical_block_size=128,
            expert_policy='selected_per_token',hc_backend=hc_backend,kda_backend=kda_backend,max_requests=1,
            dense_activation_backend=dense_activation_backend)
        for spec,layer in zip(config.layers,runner.stack.layers,strict=True):
            if spec.attention=='deepseek_sparse_attention':
                operation=bind_dsa(layer,dsa_operations[spec.layer_id])
                if not callable(operation):raise ValueError('DSA binding must return a callable operation')
                layer.attention.operation=operation
    except Exception as exc:
        failure=exc
    status=torch.tensor([int(failure is not None)],dtype=torch.int32,device=device)
    dist.all_reduce(status,op=dist.ReduceOp.MAX,group=cp)
    if int(status.item()):
        raise RuntimeError('bounded Worker component assembly failed on a rank') from failure
    installed=install_rank_text_seams(runner,reader,rank=rank,world_size=4,group=cp,
        cache_bytes=rank_cache_bytes,max_assembled_bytes=max_assembled_bytes,
        max_intermediate_rows=max_intermediate_rows,max_packed_rows=max_packed_rows,shard_dense_down=shard_dense_down)
    runner.enable_rank_commit_agreement(group=cp)
    if max_step_tokens>1:runner.enable_rank_batch_commit(max_tokens=max_step_tokens)
    bridge=SingleRequestWorkerBridge(runner,lambda request_id:initial_worker_text_state(runner,request_id))
    return BoundedWorkerRuntime(runner,GLMSingleRequestWorkerAdapter(bridge,sampler),installed)
