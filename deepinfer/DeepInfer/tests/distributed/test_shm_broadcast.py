# SPDX-License-Identifier: Apache-2.0
# Derived from vLLM; modified by DeepSeek.

from __future__ import annotations

import multiprocessing
import os
import random
import time
from collections.abc import Callable
from multiprocessing import shared_memory

import numpy as np
import pytest
import torch.distributed as dist

from deepinfer.distributed.shm_broadcast import MessageQueue, ShmRingBuffer
from deepinfer.utils import get_ip, get_open_port


@pytest.mark.ci
@pytest.mark.ci_npu
@pytest.mark.parametrize("already_unlinked", [False, True])
def test_shm_ring_buffer_cleanup(already_unlinked: bool) -> None:
    buffer = ShmRingBuffer(n_reader=1, max_chunk_bytes=32, max_chunks=2)
    name = buffer.shared_memory.name
    if already_unlinked:
        buffer.shared_memory.unlink()

    # Call directly so an exception fails the test instead of being ignored by __del__.
    buffer.__del__()
    buffer.__del__()

    assert buffer.shared_memory.buf is None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def update_environment_variables(envs: dict[str, str]) -> None:
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            print(f"Overwriting environment variable {k} from '{os.environ[k]}' to '{v}'")
        os.environ[k] = v


def get_arrays(n: int, seed: int = 0) -> list[np.ndarray]:
    np.random.seed(seed)
    sizes = np.random.randint(1, 10_000, n)
    # on average, each array will have 5k elements
    # with int64, each array will have 40kb
    return [np.random.randint(1, 100, i) for i in sizes]


def distributed_run(fn: Callable[[dict[str, str]], None], world_size: int) -> None:
    master_port = str(get_open_port())
    processes = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
        }
        p = multiprocessing.Process(target=fn, args=(env,))
        processes.append(p)
        p.start()

    deadline = time.monotonic() + 120
    try:
        for p in processes:
            p.join(timeout=max(0, deadline - time.monotonic()))
        assert all(not p.is_alive() for p in processes), "Shared-memory broadcast timed out"
        for p in processes:
            assert p.exitcode == 0
    finally:
        for p in processes:
            if p.is_alive():
                p.kill()
            p.join()


def worker_fn_wrapper(fn: Callable[[], None]) -> Callable[[dict[str, str]], None]:
    # `multiprocessing.Process` cannot accept environment variables directly
    # so we need to pass the environment variables as arguments
    # and update the environment variables in the function
    def wrapped_fn(env: dict[str, str]) -> None:
        update_environment_variables(env)
        dist.init_process_group(backend="gloo")
        fn()

    return wrapped_fn


@worker_fn_wrapper
def worker_fn() -> None:
    rank = dist.get_rank()
    if rank == 0:
        port = get_open_port()
        ip = get_ip()
        dist.broadcast_object_list([ip, port], src=0)
    else:
        recv = [None, None]
        dist.broadcast_object_list(recv, src=0)
        ip, port = recv

    writer_rank = 2
    # max_chunk_bytes = 40KB (≈ the average size of one array), max_chunks = 2
    broadcaster = MessageQueue.create_from_process_group(dist.group.WORLD, 40 * 1024, 2, writer_rank)
    if rank == writer_rank:
        seed = random.randint(0, 1000)
        dist.broadcast_object_list([seed], writer_rank)
    else:
        recv = [None]
        dist.broadcast_object_list(recv, writer_rank)
        seed = recv[0]  # type: ignore

    dist.barrier()

    # in case we find a race condition
    # print the seed so that we can reproduce the error
    print(f"Rank {rank} got seed {seed}")
    # test broadcasting with about 400MB of data
    N = 10_000
    if rank == writer_rank:
        arrs = get_arrays(N, seed)
        for x in arrs:
            broadcaster.broadcast_object(x)
            time.sleep(random.random() / 1000)
    else:
        arrs = get_arrays(N, seed)
        for x in arrs:
            y = broadcaster.broadcast_object(None)
            assert np.array_equal(x, y)
            time.sleep(random.random() / 1000)

    print("torch distributed passed the test!")


@pytest.mark.ci
@pytest.mark.ci_npu
def test_shm_broadcast() -> None:
    """Correctness of MessageQueue shared-memory broadcast of a ~400MB random array across 4 processes (gloo)."""
    distributed_run(worker_fn, 4)
