import torch
import torch_npu  # noqa: F401

from generators import GemmType
from common import run_gemm_test


def test_fp8_gemm(
        gemm_type: GemmType = GemmType.Normal,
        unaligned_mn: bool = False,
        unaligned_k: bool = False,
        strided: bool = False,
        skip_prof: bool = False,
        test_recipes: bool = False,
):
    return run_gemm_test(
        torch.float8_e4m3fn, torch.float8_e4m3fn,
        gemm_type, unaligned_mn, unaligned_k, strided,
        skip_prof=skip_prof,
        test_recipes=test_recipes,
        threshold=1e-5,
    )


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('select', nargs='?', default='all', choices=['all', 'unaligned-k', 'unaligned-mn', 'strided', 'normal', 'mgrouped', 'kgrouped'])
    parser.add_argument('--skip-prof', '--no-prof', dest='skip_prof', action='store_true', help='Skip performance measurements and just test correctness')

    args = parser.parse_args()
    sel = args.select
    skip_prof = args.skip_prof
    if sel == 'unaligned-k':
        test_fp8_gemm(GemmType.Normal, unaligned_k=True, skip_prof=skip_prof)
    elif sel == 'unaligned-mn':
        test_fp8_gemm(GemmType.Normal, unaligned_mn=True, skip_prof=skip_prof)
    elif sel == 'strided':
        test_fp8_gemm(GemmType.Normal, strided=True, skip_prof=skip_prof)
    elif sel == 'normal':
        test_fp8_gemm(GemmType.Normal, skip_prof=skip_prof)
    elif sel == 'mgrouped':
        test_fp8_gemm(GemmType.MGroupedContiguousWithPsumLayout, skip_prof=skip_prof)
    elif sel == 'kgrouped':
        test_fp8_gemm(GemmType.KGroupedContiguousWithPsumLayout, skip_prof=skip_prof)
    else:
        test_fp8_gemm(GemmType.Normal, skip_prof=skip_prof)
        test_fp8_gemm(GemmType.Normal, unaligned_mn=True, skip_prof=skip_prof)
        test_fp8_gemm(GemmType.Normal, unaligned_k=True, skip_prof=skip_prof)
        test_fp8_gemm(GemmType.Normal, strided=True, skip_prof=skip_prof)

        test_fp8_gemm(GemmType.MGroupedContiguousWithPsumLayout, skip_prof=skip_prof)
        test_fp8_gemm(GemmType.KGroupedContiguousWithPsumLayout, skip_prof=skip_prof)
        test_fp8_gemm(GemmType.Normal, skip_prof=skip_prof, test_recipes=True)
