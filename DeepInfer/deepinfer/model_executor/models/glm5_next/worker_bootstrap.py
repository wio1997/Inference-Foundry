# SPDX-License-Identifier: Apache-2.0
"""Explicitly bootstrap an already initialized four-rank GLM text Worker.

No backend, service, cache allocator or model registration is chosen here.
The caller supplies factories and budgets and owns Worker/process-group lifetime.
"""
import torch
import torch.distributed as dist

from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.parallel_utils.parallel_state import PG

from .worker_runtime import prepare_bounded_worker_runtime


@torch.inference_mode()
def bootstrap_bounded_glm_worker(worker, reader, device, *, dsa_factory, bind_dsa, sampler_factory,
        max_expert_checkpoint_bytes, max_expert_resident_bytes, max_context_tokens, max_step_tokens,
        rank_cache_bytes, max_assembled_bytes, max_intermediate_rows, max_packed_rows, hc_backend, kda_backend):
    """Prepare and install a single-request adapter on all four CP ranks.

    The Worker must already have initialized its device and process group. The
    driver alone creates the explicitly supplied sampler. Local preparation and
    installation failures are agreed; installed adapters are rolled back only
    if this call created them and no request has started. Process loss or
    failure inside a collective is outside this contract.
    """
    cp=PG.context_parallel_group()
    if not dist.is_initialized() or cp.size()!=4 or dist.get_world_size(cp)!=4:
        raise ValueError('initialized four-rank CP group required')
    rank=cp.rank();failure=None;ops=None;sampler=None
    try:
        if (worker.model_config.architecture!='Glm5NextForConditionalGeneration' or worker.nextn!=0
                or worker.prev_step_state is not None or bool(worker.is_driver_worker)!=(rank==0)
                or getattr(worker,'_glm_single_request_adapter',None) is not None):
            raise ValueError('fresh compatible non-speculative GLM Worker required')
        if not callable(dsa_factory) or not callable(bind_dsa) or not callable(sampler_factory):
            raise ValueError('explicit DSA, binding and sampler factories required')
        config=Glm5NextConfig.from_checkpoint(reader.root)
        ops=dsa_factory(config)
        if rank==0:sampler=sampler_factory()
    except Exception as exc:
        failure=exc
    status=torch.tensor([int(failure is not None)],dtype=torch.int32,device=device)
    dist.all_reduce(status,op=dist.ReduceOp.MAX,group=cp)
    if int(status.item()):
        raise RuntimeError('GLM Worker bootstrap preflight failed on a rank') from failure
    runtime=prepare_bounded_worker_runtime(reader,device,dsa_operations=ops,bind_dsa=bind_dsa,sampler=sampler,
        max_expert_checkpoint_bytes=max_expert_checkpoint_bytes,max_expert_resident_bytes=max_expert_resident_bytes,
        max_context_tokens=max_context_tokens,max_step_tokens=max_step_tokens,rank_cache_bytes=rank_cache_bytes,
        max_assembled_bytes=max_assembled_bytes,max_intermediate_rows=max_intermediate_rows,max_packed_rows=max_packed_rows,
        hc_backend=hc_backend,kda_backend=kda_backend)
    failure=None
    try:
        runtime.install(worker)
    except Exception as exc:
        failure=exc
    status=torch.tensor([int(failure is not None)],dtype=torch.int32,device=device)
    dist.all_reduce(status,op=dist.ReduceOp.MAX,group=cp)
    if int(status.item()):
        if getattr(worker,'_glm_single_request_adapter',None) is runtime.adapter:
            del worker._glm_single_request_adapter
        raise RuntimeError('GLM Worker adapter installation failed on a rank') from failure
    return runtime


@torch.inference_mode()
def bootstrap_native_sfa_glm_worker(worker, device, *, sfa_library, max_read_bytes,
        max_expert_checkpoint_bytes, max_expert_resident_bytes, max_context_tokens, max_step_tokens,
        rank_cache_bytes, max_assembled_bytes, max_intermediate_rows, max_packed_rows, hc_backend, kda_backend):
    """Open the Worker's checkpoint and select the installed SFA explicitly.

    Every rank reads the same configured checkpoint independently. Reader
    failures are agreed before any model component is allocated. This is an
    opt-in bounded text path, not a production model registration or cache.
    """
    cp=PG.context_parallel_group()
    if not dist.is_initialized() or cp.size()!=4 or dist.get_world_size(cp)!=4:
        raise ValueError('initialized four-rank CP group required')
    reader=None;failure=None
    try:
        from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
        if not isinstance(sfa_library,str) or not sfa_library or type(max_read_bytes) is not int or max_read_bytes<1:
            raise ValueError('explicit SFA library and checkpoint read bound required')
        reader=ModelSlimCheckpoint(worker.model_config.model,max_read_bytes=max_read_bytes)
    except Exception as exc:
        failure=exc
    status=torch.tensor([int(failure is not None)],dtype=torch.int32,device=device)
    dist.all_reduce(status,op=dist.ReduceOp.MAX,group=cp)
    if int(status.item()):
        raise RuntimeError('GLM checkpoint reader preparation failed on a rank') from failure
    def dsa_factory(config):
        from .native_dsa import make_dsa_operations
        return make_dsa_operations(config,sfa_library=sfa_library,max_total_tokens=max_context_tokens)
    def bind_dsa(layer,operation):
        # NativeShortDSA is bound by the existing decoder loader.
        return operation
    def sampler_factory():
        from vllm_ascend.sample.sampler import AscendSampler
        return AscendSampler()
    return bootstrap_bounded_glm_worker(worker,reader,device,dsa_factory=dsa_factory,bind_dsa=bind_dsa,
        sampler_factory=sampler_factory,max_expert_checkpoint_bytes=max_expert_checkpoint_bytes,
        max_expert_resident_bytes=max_expert_resident_bytes,max_context_tokens=max_context_tokens,
        max_step_tokens=max_step_tokens,rank_cache_bytes=rank_cache_bytes,max_assembled_bytes=max_assembled_bytes,
        max_intermediate_rows=max_intermediate_rows,max_packed_rows=max_packed_rows,hc_backend=hc_backend,kda_backend=kda_backend)
