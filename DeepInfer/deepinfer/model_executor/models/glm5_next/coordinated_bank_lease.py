# SPDX-License-Identifier: Apache-2.0
"""Coordinate synchronous bank-load exceptions among live ranks.

Synchronous exceptions during lease entry and successful-body cleanup are coordinated. A lost
process, a failed device/collective, or later asynchronous compute failure
still requires the distributed backend timeout and external recovery.
"""
from contextlib import contextmanager, ExitStack
import torch
import torch.distributed as dist


@contextmanager
def coordinated_bank_lease(lease, *, device, rank, world_size, group=None):
    if type(world_size) is not int or world_size not in (1, 4) or type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError('explicit supported rank/world required')
    if world_size == 4 and (not dist.is_initialized() or dist.get_rank(group) != rank or dist.get_world_size(group) != world_size):
        raise ValueError('matching initialized group required')
    with ExitStack() as stack:
        failure = None
        value = None
        try:
            value = stack.enter_context(lease)
        except Exception as exc:
            failure = exc
        if world_size == 4:
            status = torch.tensor([int(failure is not None)], dtype=torch.int32, device=device)
            dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
            if int(status.item()) != 0:
                raise RuntimeError('one or more ranks failed to load expert bank') from failure
        elif failure is not None:
            raise failure
        try:
            yield value
        except Exception as body_failure:
            # Preserve the coordinated body failure if cleanup also fails.
            try:
                stack.__exit__(type(body_failure), body_failure, body_failure.__traceback__)
            except Exception as cleanup_failure:
                body_failure.add_note('lease cleanup also failed: ' + repr(cleanup_failure))
            raise
        else:
            coordinated_local_stage(stack.close, device=device, rank=rank, world_size=world_size, group=group)


def coordinated_local_stage(operation, *, device, rank, world_size, group=None):
    """Coordinate synchronous local-stage errors before later collectives.

    All participating ranks must call this in the same order. The operation
    must synchronize device work if asynchronous failures need observation.
    Process loss, a failed status collective and later errors remain outside
    this guarantee.
    """
    if type(world_size) is not int or world_size not in (1, 4) or type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError('explicit supported rank/world required')
    if world_size == 4 and (not dist.is_initialized() or dist.get_rank(group) != rank or dist.get_world_size(group) != world_size):
        raise ValueError('matching initialized group required')
    failure = None
    value = None
    try:
        value = operation()
    except Exception as exc:
        failure = exc
    if world_size == 4:
        status = torch.tensor([int(failure is not None)], dtype=torch.int32, device=device)
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
        if int(status.item()) != 0:
            raise RuntimeError('one or more ranks failed local expert execution') from failure
    elif failure is not None:
        raise failure
    return value
