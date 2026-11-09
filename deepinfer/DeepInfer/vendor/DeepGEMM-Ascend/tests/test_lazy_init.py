import argparse
import os

import torch
import torch.multiprocessing as mp
import torch_npu
import deep_gemm


def main(local_rank: int):
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'
    torch_npu.npu.set_device(local_rank)
    value = torch.ones(1, device=f'npu:{local_rank}')
    torch_npu.npu.synchronize()
    assert value.item() == 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test NPU initialization after importing deep_gemm before fork')
    parser.add_argument('--num-processes', type=int, default=1, help='Number of processes/devices (default: 1)')
    args = parser.parse_args()
    if args.num_processes < 1:
        parser.error('--num-processes must be positive')

    print(f'deep_gemm extension: {deep_gemm._C.__file__}', flush=True)
    mp.start_processes(main, nprocs=args.num_processes, start_method='fork')
    print('PASS')
