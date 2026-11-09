import argparse

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.testing import bench, parse_num_bytes


def test_performance(buffer: deep_ep.BucketBuffer, gathered: torch.Tensor,
                     group: dist.ProcessGroup, args: argparse.Namespace) -> None:
    rank_idx, num_ranks = group.rank(), group.size()
    gathered.fill_(float('nan'))
    src = gathered[rank_idx]
    # HCCL requires torch_npu format metadata absent from the registered from_blob view.
    hccl_src = torch.empty(src.shape, dtype=src.dtype, device='npu').normal_().add_(rank_idx)
    src.copy_(hccl_src)
    hccl_dst = torch.empty(gathered.numel(), dtype=gathered.dtype, device='npu')

    def run_hccl():
        dist.all_gather_into_tensor(hccl_dst, hccl_src, group=group)

    def run_deep_ep():
        # One tensor is one bucket; wait() includes the deferred CQ drain and barrier.
        return buffer.all_gather(src, group=group, num_sms=args.num_sms).wait()

    run_hccl()
    result = run_deep_ep()
    assert torch.equal(result, hccl_dst), 'Performance bucket mismatch'

    bench_kwargs = dict(num_warmups=args.num_warmups, num_tests=args.num_tests,
                        barrier=lambda: buffer.barrier(group=group, with_cpu_sync=True))
    hccl_t, _, _ = bench(run_hccl, **bench_kwargs)
    t, _, _ = bench(run_deep_ep, **bench_kwargs)
    durations = torch.tensor([hccl_t, t], dtype=torch.float32, device='npu')
    dist.all_reduce(durations, op=dist.ReduceOp.MAX, group=group)
    hccl_t, t = durations.tolist()
    assert torch.equal(gathered.view(-1), hccl_dst), 'Performance bucket mismatch after benchmarking'

    num_gathered_bytes = gathered.nbytes
    num_remote_bytes = src.nbytes * (num_ranks - 1)
    dist_print(f'Performance (1 bucket, in-place):\n'
               f' > Ranks: {num_ranks}, resident SMs: {args.num_sms}\n'
               f' > Bytes per rank: {src.nbytes} ({src.nbytes / (1 << 30):g} GiB), '
               f'total gathered bytes: {num_gathered_bytes}\n'
               f' > Timing: NPU events around the complete operation, max rank mean; '
               f'{args.num_warmups} warmups, {args.num_tests} iterations\n'
               f' > DeepEP: {t * 1e6:.1f} us | {num_gathered_bytes / t / 1e9:.1f} GB/s gathered | '
               f'{num_remote_bytes / t / 1e9:.1f} GB/s remote | {hccl_t / t:.2f}x HCCL\n'
               f' > HCCL:   {hccl_t * 1e6:.1f} us | {num_gathered_bytes / hccl_t / 1e9:.1f} GB/s gathered | '
               f'{num_remote_bytes / hccl_t / 1e9:.1f} GB/s remote', once_in_node=True)


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks >= 2 and num_ranks % 2 == 0
    subgroup = None
    for rank_begin_idx in range(0, num_ranks, 2):
        item = dist.new_group(list(range(rank_begin_idx, rank_begin_idx + 2)))
        if rank_begin_idx <= rank_idx < rank_begin_idx + 2:
            subgroup = item

    plan = deep_ep.BufferAllocator()
    gathered = plan.allocate((num_ranks, 2, 8), torch.float32)
    shard_sizes = (7, 127, 128, 129, 1025, 16385)
    unaligned = plan.allocate((1 + num_ranks * max(shard_sizes),), torch.float32)
    buckets = [plan.allocate((num_ranks, 16 if idx % 2 == 0 else 257 + idx), torch.float32)
               for idx in range(64)]
    perf_gathered = None if args.skip_perf_test else plan.allocate((num_ranks, args.num_bytes // 4), torch.float32)
    buffer = deep_ep.BucketBuffer([group, subgroup], plan, explicitly_destroy=True)

    for item in (group, subgroup):
        for iteration in range(2):
            src = torch.arange(16, dtype=torch.float32, device='npu').view(2, 8)
            src += item.rank() * 100 + iteration * 1000
            ref = torch.stack([src - item.rank() * 100 + rank * 100 for rank in range(item.size())])
            dst = gathered[:item.size()]
            # Poison every shard, including our own, to detect missing local copies.
            dst.fill_(float('nan'))
            handle = buffer.all_gather(src, dsts=dst, group=item, num_sms=args.num_sms)
            # The barrier is deferred until wait().
            # Simulate some work to be done while the all-gather is in progress.
            _ = torch.ones(32, dtype=torch.float32, device='npu') + 1
            result = handle.wait()
            assert result.ndim == 1
            assert torch.equal(result, ref.view(-1))

            src += 10000
            ref += 10000
            dst.fill_(float('nan'))
            dst[item.rank()].copy_(src)
            result = buffer.all_gather(dst[item.rank()], group=item, num_sms=args.num_sms).wait()
            assert torch.equal(result, ref.view(-1))
            buffer.barrier(group=item, with_cpu_sync=True)

    # Cover different shard sizes and unaligned addresses/tails in each context.
    for item in (group, subgroup):
        for num_elems in shard_sizes:
            src = torch.arange(num_elems + 1, dtype=torch.float32, device='npu')[1:]
            src += item.rank() * 100
            dst = unaligned[1:1 + item.size() * num_elems].view(item.size(), num_elems)
            dst.fill_(float('nan'))
            ref = torch.stack([src - item.rank() * 100 + rank * 100 for rank in range(item.size())])
            result = buffer.all_gather(src, dsts=dst, group=item, num_sms=args.num_sms).wait()
            assert torch.equal(result, ref.view(-1))

    # Exercise the complete BucketList with distinct inputs and registered destinations.
    srcs = [torch.full(bucket[rank_idx].shape, rank_idx + 100 * idx, dtype=torch.float32, device='npu')
            for idx, bucket in enumerate(buckets)]
    for bucket in buckets:
        bucket.fill_(float('nan'))
    outputs = buffer.all_gather(srcs, dsts=buckets, group=group, num_sms=args.num_sms).wait()
    assert len(outputs) == len(buckets)
    for idx, (output, bucket) in enumerate(zip(outputs, buckets, strict=True)):
        ref = torch.stack([torch.full_like(srcs[idx], rank + 100 * idx) for rank in range(num_ranks)])
        assert output.data_ptr() == bucket.data_ptr()
        assert torch.equal(output, ref.view(-1))

    dist_print('All-gather and 64-bucket batches passed', once_in_node=True)
    if not args.skip_perf_test:
        test_performance(buffer, perf_gathered, group, args)

    buffer.destroy()
    deep_ep.destroy_all_managed_hccl_comm()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test all-gather')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-sms', type=int, default=0, choices=(0,))
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=1 << 30,
                        help='Performance input size PER RANK, with binary suffixes: 1G (default), 64M, 512K, or bytes')
    parser.add_argument('--num-warmups', type=int, default=10)
    parser.add_argument('--num-tests', type=int, default=30)
    parser.add_argument('--skip-perf-test', action='store_true', help='Run correctness tests only')
    args = parser.parse_args()
    if args.num_bytes % 4:
        parser.error('--num-bytes must be divisible by 4 (float32)')
    if args.num_warmups < 0 or args.num_tests <= 0:
        parser.error('--num-warmups must be nonnegative and --num-tests must be positive')
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
