import inspect
import os
import random

import torch
import torch.distributed as dist


__all__ = ['init_dist', 'init_seed', 'dist_print']


_local_rank = None


def init_seed(global_seed: int) -> None:
    local_seed = global_seed + dist.get_rank()
    torch.manual_seed(local_seed)
    random.seed(local_seed)


def dist_print(s: str = '', once_in_node: bool = False) -> None:
    assert _local_rank is not None
    if not once_in_node or _local_rank == 0:
        print(s, flush=True)
    dist.barrier()


def init_dist(local_rank: int, num_local_ranks: int, seed: int = 0) -> tuple[int, int, dist.ProcessGroup]:
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    global _local_rank
    _local_rank = local_rank

    params = {
        'backend': 'hccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in inspect.signature(dist.init_process_group).parameters:
        params['device_id'] = torch.device(f'npu:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('npu')
    torch.npu.set_device(local_rank)
    init_seed(seed)

    group = dist.distributed_c10d._get_default_group()
    return dist.get_rank(), dist.get_world_size(), group
