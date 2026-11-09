# Filter and disable all annoying warnings
import os
import warnings
os.environ['TORCH_NPU_DISABLED_WARNING'] = '1'
# TODO(HUAWEI): fix permission warnings
warnings.filterwarnings('ignore', message='Permission mismatch')
warnings.filterwarnings('ignore', message=r'Warning: The .* owner does not match the current owner\.')
warnings.filterwarnings('ignore', message='Cannot create tensor with')

import argparse
import math
import random
import numpy as np
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.math import align


def test_session(group: dist.ProcessGroup) -> None:
    rank_idx, num_ranks = group.rank(), group.size()
    subgroup = dist.new_group(list(range(num_ranks)))
    buffer = deep_ep.BucketBuffer(
        [group, subgroup], deep_ep.get_num_allocation_alignment(), explicitly_destroy=True)
    src = torch.full((2, 8), rank_idx, dtype=torch.float32, device='npu')
    ref = torch.stack([torch.full_like(src, rank) for rank in range(num_ranks)])
    dst = torch.empty_like(ref)
    with buffer.session() as session:
        result = buffer.all_gather(src, dsts=dst, group=group).wait()
        assert result is dst
        torch.testing.assert_close(dst, ref, rtol=0, atol=0)
        tensors = session.allocate(((num_ranks, 2, 8), (num_ranks, 2, 8)), group=group)
        for tensor in tensors:
            tensor[rank_idx].copy_(src)
        outputs = buffer.all_gather([tensor[rank_idx] for tensor in tensors], group=group).wait()
        for output in outputs:
            torch.testing.assert_close(output, ref.view(-1), rtol=0, atol=0)
        try:
            session.allocate((8,), group=subgroup)
        except AssertionError:
            pass
        else:
            raise AssertionError('A session accepted different communication groups')

    try:
        with buffer.session():
            raise ValueError('session cleanup')
    except ValueError:
        pass
    assert 'all_gather' not in buffer.__dict__

    buffer.destroy()


def all_gather_ref(shape: tuple, rank_idx: int, num_ranks: int, round_idx: int = 0):
    ref_list = []
    for i in range(num_ranks):
        torch.manual_seed(42 + round_idx * 43 + i)
        ref_list.append(torch.randn(shape, dtype=torch.bfloat16, device='npu'))
    return ref_list[rank_idx], torch.stack(ref_list, dim=0)


def generate_stress_ops(
        num_ops: int,
        num_max_inflight_gathers: int,
        shape: tuple,
        rank_idx: int,
        num_ranks: int,
) -> tuple[list[tuple], tuple[torch.Tensor], tuple[torch.Tensor]]:
    tensors, refs = zip(
        *(all_gather_ref(shape, rank_idx, num_ranks, round_idx=i) for i in range(num_ops)),
        strict=True,
    )
    unprocessed = random.sample(range(num_ops), num_ops)
    inflight, ops = [], [('create_session', (-1, ))]
    limit = num_max_inflight_gathers
    while unprocessed or inflight:
        num_max_gathers = min(len(unprocessed), limit, 64)
        choices = []
        if num_max_gathers > 0:
            choices.append('ag')
        if inflight:
            choices.append('fetch')
        else:
            choices.append('destroy')
        op = random.choice(choices)
        if op == 'ag':
            indices = tuple(unprocessed[-random.randint(1, num_max_gathers):])
            limit -= len(indices)
            del unprocessed[-len(indices):]
            inflight.append(indices)
            ops.append(('ag', indices))
        elif op == 'fetch':
            ops.append(('fetch', inflight.pop(random.randrange(len(inflight)))))
        else:
            ops.extend([('destroy_session', (-1, )), ('create_session', (-1, ))])
            limit = num_max_inflight_gathers

    ops.append(('destroy_session', (-1, )))
    return ops, tensors, refs


def do_all_gather(buffer: deep_ep.BucketBuffer, session: deep_ep.BucketSession,
                  is_inplace: bool, is_batched: bool,
                  tensors: tuple[torch.Tensor, ...],
                  start_event: torch.npu.Event | None = None):
    if is_inplace:
        group = buffer.groups[0]
        gathered = session.allocate(
            tuple((group.size(), *tensor.shape) for tensor in tensors), torch.bfloat16)
        ag_tensors = tuple(tensor[group.rank()] for tensor in gathered)
        for dst, src in zip(ag_tensors, tensors, strict=True):
            dst.copy_(src)
    else:
        ag_tensors = tensors

    if start_event is not None:
        torch.zeros(int(256e6 // 4), dtype=torch.int, device='npu')
        start_event.record()

    return [buffer.all_gather(ag_tensors)] if is_batched else [buffer.all_gather(tensor) for tensor in ag_tensors]


# noinspection PyTypeChecker,PyCallingNonCallable,PyShadowingNames
@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    test_session(group)

    shape = (32, 64, 2048)
    num_max_inflight_gathers = args.num_max_inflight_gathers
    num_max_session_bytes = num_max_inflight_gathers * align(
        num_ranks * math.prod(shape) * torch.bfloat16.itemsize, deep_ep.get_num_rdma_alignment())
    dist_print(f'Config:\n'
               f' > Ranks: {num_ranks}\n'
               f' > Shape: {shape}\n'
               f' > Max inflight gathers: {num_max_inflight_gathers}\n',
               once_in_node=True)

    buffer = deep_ep.BucketBuffer(
        group, align(num_max_session_bytes, deep_ep.get_num_allocation_alignment()), explicitly_destroy=True)
    assert isinstance(buffer.get_comm_stream(), torch.npu.Stream)

    dist_print('Running stress tests:', once_in_node=True)
    for seed in range(args.num_stress_iterations):
        random.seed(42 + seed)
        num_ops = 128
        ops, tensors, refs = generate_stress_ops(
            num_ops, num_max_inflight_gathers, shape, rank_idx, num_ranks)
        results = [None] * num_ops
        handles = dict()
        torch.npu.synchronize()
        for op, indices in ops:
            if op == 'create_session':
                session = buffer.session()
                session.__enter__()
            elif op == 'destroy_session':
                session.__exit__(None, None, None)
                buffer.barrier()
            elif op == 'ag':
                is_inplace, is_batched = random.random() < 0.5, random.random() < 0.8
                handles[indices] = do_all_gather(
                    buffer, session, is_inplace, is_batched, tuple(tensors[i] for i in indices))
            elif op == 'fetch':
                out_tensors = []
                for handle in handles.pop(indices):
                    out = handle.wait()
                    out_tensors.extend(out if isinstance(out, list) else [out])
                for out, idx in zip(out_tensors, indices, strict=True):
                    results[idx] = out.view_as(refs[idx]).clone()

        for i in range(num_ops):
            if results[i] is None or not torch.equal(results[i], refs[i]):
                mismatched_ranks = [
                    j for j in range(num_ranks)
                    if results[i] is None or not torch.equal(results[i][j], refs[i][j])
                ]
                raise AssertionError(
                    f'Rank {rank_idx}: stress mismatch at seed={seed}, op={i}, '
                    f'peer ranks={mismatched_ranks}'
                )
        dist_print(f' > Seed {seed} passed ({num_ops} ops)', once_in_node=True)
    dist_print(once_in_node=True)

    dist_print('Profiling all-gather:', once_in_node=True)
    buffer.destroy()

    num_max_session_bytes = num_max_inflight_gathers * align(
        num_ranks * (2 ** 26) * torch.bfloat16.itemsize, deep_ep.get_num_rdma_alignment())
    buffer = deep_ep.BucketBuffer(
        group, align(num_max_session_bytes, deep_ep.get_num_allocation_alignment()), explicitly_destroy=True)
    for num_bytes in (2 ** p for p in range(20, 27)):
        shape = (num_bytes // 2, )
        tensors = tuple(
            torch.randn(shape, dtype=torch.bfloat16, device='npu')
            for _ in range(num_max_inflight_gathers)
        )

        for is_inplace in (False, True):
            for is_batched in (False, True):
                num_tests = 50
                start_events = [torch.npu.Event(enable_timing=True) for _ in range(num_tests)]
                end_events = [torch.npu.Event(enable_timing=True) for _ in range(num_tests)]
                torch.npu.synchronize()

                for i in range(num_tests):
                    with buffer.session() as session:
                        wait_handles = do_all_gather(
                            buffer, session, is_inplace, is_batched, tensors, start_event=start_events[i])
                        for handle in wait_handles:
                            handle.wait()
                    buffer.barrier()
                    end_events[i].record()
                torch.npu.synchronize()

                times = np.array([
                    start.elapsed_time(end) / 1e3
                    for start, end in zip(start_events, end_events, strict=True)
                ])[1:]
                avg_t = np.average(times)
                unit = ('MB', 1e6) if num_bytes >= 1e6 else ('KB', 1e3)
                bandwidth_info = (
                    f', {num_bytes * num_ranks * num_max_inflight_gathers / avg_t / 1e9:.3f} GB/s'
                    if num_ranks > 1 else ''
                )

                dist_print(
                    f' > Rank: {rank_idx:3}/{num_ranks:3} | '
                    f'{num_ranks} x {(num_bytes / unit[1]):.0f} {unit[0]} | '
                    f'avg: {avg_t / num_max_inflight_gathers * 1e6:.3f} us'
                    f'{bandwidth_info}'
                    f' (inplace={int(is_inplace)}, batched={int(is_batched)})')
    dist_print(once_in_node=True)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test BucketBuffer sessions')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-max-inflight-gathers', type=int, default=4)
    parser.add_argument('--num-stress-iterations', type=int, default=4)
    args = parser.parse_args()

    torch.multiprocessing.spawn(
        test, args=(args.num_processes, args), nprocs=args.num_processes)
