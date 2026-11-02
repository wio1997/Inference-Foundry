import os
from collections.abc import Iterable, Sequence

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing

from deepinfer._C import cuda_utils
from deepinfer.config import ParallelConfig
from deepinfer.model_executor.eplb import (
    expert_model_parallel_shuffle_inplace,
    expert_model_parallel_shuffle_inplace_intranode,
)
from deepinfer.model_executor.parallel_utils.parallel_state import PG
from deepinfer.utils import CudaIPCWrapper, get_open_port
from deepinfer.worker.base import _init_distributed_environment

ep_size = 8
num_local_experts = 2
num_experts = 5

PYTEST_STEPS = 200


def generate_weight(expert_ids: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build two distinguishable weight copies by expert id (values are id / -id) so the result can be checked directly after shuffling."""
    return (
        torch.tensor([[expert_id] for expert_id in expert_ids], dtype=torch.int32, device="cuda"),
        torch.tensor([[expert_id, -expert_id] for expert_id in expert_ids], dtype=torch.int32, device="cuda"),
    )


def weight_equal(weight1: Sequence[torch.Tensor], weight2: Sequence[torch.Tensor]) -> bool:
    if len(weight1) != len(weight2):
        return False
    return all(w1.equal(w2) for w1, w2 in zip(weight1, weight2, strict=False))


def new_shuffled_expert_indices() -> torch.Tensor:
    """Randomly generate the global expert placement over [1, ep_size*num_local_experts]: every expert appears at least once and the remaining slots repeat randomly."""
    indices = np.concatenate(
        [
            np.arange(num_experts, dtype=np.int32),
            np.random.randint(0, num_experts, size=ep_size * num_local_experts - num_experts, dtype=np.int32),
        ],
        dtype=np.int32,
    )
    np.random.shuffle(indices)
    return torch.from_numpy(indices).cuda().view(1, ep_size * num_local_experts)


def local_expert_ids(global_expert_indices: torch.Tensor, rank: int) -> torch.Tensor:
    """Take the local expert-id segment of the global placement that belongs to a rank."""
    return global_expert_indices[0][rank * num_local_experts : (rank + 1) * num_local_experts]


def _run_expert_shuffle(rank: int, step: int, port: str, impl: str) -> None:
    torch.cuda.set_device(torch.cuda.device(rank))
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")

    parallel_config = ParallelConfig(
        data_parallel_size=ep_size,
        context_parallel_size=1,
        expert_parallel_size=ep_size,
        engram_parallel_size=ep_size,
        engine_rank=rank,
        engine_group_rank=0,
    )
    _init_distributed_environment(parallel_config, 0, f"tcp://{ip}:{port}")

    if rank == 0:
        global_expert_indices_tensor = new_shuffled_expert_indices()
    else:
        global_expert_indices_tensor = torch.empty(1, ep_size * num_local_experts, dtype=torch.int32, device="cuda")
        # receive buffers for non-rank0; rank 0 regenerates the whole tensor every step, so no preallocation is needed
        current_global_expert_indices_tensor = torch.empty_like(global_expert_indices_tensor)
    dist.broadcast(global_expert_indices_tensor, 0)

    # the outer list is the layer dimension; a single layer is enough to cover the shuffle logic
    weight = [generate_weight(local_expert_ids(global_expert_indices_tensor, rank))]

    ep_group = PG.expert_parallel_group()

    if impl == "intranode":
        # each rank allocates a mirror buffer with the same layout as its local weights, exposed over CUDA IPC so the whole group can read and write it directly
        current_experts_weight = [cuda_utils.alloc_tensor(w.nbytes).view(w.dtype).view_as(w) for w in weight[0]]
        current_experts_weight_ipc_handle = [CudaIPCWrapper(tensor) for tensor in current_experts_weight]
        ipc_experts_weight = [None] * ep_group.size()
        dist.all_gather_object(ipc_experts_weight, current_experts_weight_ipc_handle, ep_group)
        for i in range(ep_group.size()):
            ipc_experts_weight[i] = current_experts_weight if i == ep_group.rank() else [x.to_tensor() for x in ipc_experts_weight[i]]

    for i in range(step):
        if rank == 0:
            if i % 100 == 0:
                print(f"current step: {i}/{step}")
            current_global_expert_indices_tensor = new_shuffled_expert_indices()
        dist.broadcast(current_global_expert_indices_tensor, 0)

        if impl == "p2p":
            expert_model_parallel_shuffle_inplace(
                global_expert_indices_tensor,
                weight,
                current_global_expert_indices_tensor,
                ep_group,  # matches the production path; P2P CUDA tensors need an NCCL group
            )
        else:
            expert_model_parallel_shuffle_inplace_intranode(
                ipc_experts_weight,
                global_expert_indices_tensor,
                weight,
                current_global_expert_indices_tensor,
                ep_group,
            )

        expected_local_ids = local_expert_ids(current_global_expert_indices_tensor, rank)
        if not weight_equal(weight[0], generate_weight(expected_local_ids)):
            print(f"[{rank}] origin={global_expert_indices_tensor.tolist()}")
            print(f"[{rank}] current={current_global_expert_indices_tensor.tolist()}")
            print(f"[{rank}] {weight[0]=} should be expert {expected_local_ids.tolist()}")
            raise AssertionError

        global_expert_indices_tensor.copy_(current_global_expert_indices_tensor)


def run_expert_shuffle(step: int, impl: str) -> None:
    port = str(get_open_port())
    torch.multiprocessing.spawn(_run_expert_shuffle, args=(step, port, impl), nprocs=ep_size, join=True)


def test_expert_shuffle() -> None:
    """NCCL batched P2P shuffle: after several random shuffle steps, each rank's local expert weights stay consistent with the global expert index."""
    if torch.cuda.device_count() < ep_size:
        pytest.skip(f"requires {ep_size} GPUs")
    run_expert_shuffle(PYTEST_STEPS, "p2p")


def test_expert_shuffle_intranode() -> None:
    """Intranode (CUDA IPC) shuffle: after several random shuffle steps, each rank's local expert weights stay consistent with the global expert index."""
    if torch.cuda.device_count() < ep_size:
        pytest.skip(f"requires {ep_size} GPUs")
    run_expert_shuffle(PYTEST_STEPS, "intranode")
