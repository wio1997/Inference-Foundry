import gc
import time
import torch
import torch_npu  # noqa: F401
from functools import partial

import deep_gemm
from deep_gemm.testing import bench_msprof
from deep_gemm.testing.par_compile import par_compile
from generators import (
    GemmType,
    Major,
    enumerate_k_grouped_psum_sf_layout,
    enumerate_sf_layout,
    enumerate_sf_layout_performance_tests,
    iter_psum_sep,
)


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def make_sf(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float32:
        exponent = torch.randint(1, 255, shape, dtype=torch.int32)
        return (exponent * (1 << 23)).view(torch.float32).npu()
    return torch.randint(-32768, 32768, shape, dtype=torch.int16).npu()


def with_major(sf: torch.Tensor, mn_major: bool) -> torch.Tensor:
    if mn_major and sf.size(-2) > 1 and sf.size(-1) > 1:
        sf = sf.transpose(-2, -1).contiguous().transpose(-2, -1)
        assert sf.stride(-2) == 1
    else:
        assert sf.stride(-1) == 1
    return sf

def reference(sf: torch.Tensor, mn: int, gran_mn: int) -> torch.Tensor:
    if sf.dtype == torch.float32:
        exponent = ((sf.view(torch.int32) // (1 << 23)) & 0xff).to(torch.uint8)
        if exponent.size(-1) % 2:
            exponent = torch.nn.functional.pad(exponent, (0, 1))
        sf = exponent.contiguous().view(torch.int16)
    return (
        sf.transpose(-2, -1)
        .repeat_interleave(gran_mn, -1)[..., :mn]
        .contiguous()
        .transpose(-2, -1)
    )


def check_layout(output: torch.Tensor, expected: torch.Tensor) -> None:
    assert output.dtype == torch.int16
    assert output.shape == expected.shape
    assert output.stride(-2) == 1
    assert output.stride(-1) == output.size(-2)
    assert torch.equal(output, expected)


def fill_group_padding_(sf: torch.Tensor, psum_layout: list[int], alignment: int, divisor: int, value) -> None:
    for span in iter_psum_sep(psum_layout, alignment, divisor):
        sf[..., span] = value


def bench_host_us(fn, num_warmups: int = 10, num_tests: int = 1000) -> float:
    for _ in range(num_warmups):
        fn()
    start_ns = time.perf_counter_ns()
    for _ in range(num_tests):
        fn()
    return (time.perf_counter_ns() - start_ns) / num_tests / 1000


def make_layout_case(case):
    gemm_type, dtype, major, mn, k, gran_mn, num_groups, alignment, psum_cpu = case
    mn_major = major == Major.MN
    psum = None
    if psum_cpu:
        expected_size = mn if gemm_type == GemmType.MGroupedContiguousWithPsumLayout else k
        assert len(psum_cpu) == num_groups and ceil_div(psum_cpu[-1], alignment) * alignment == expected_size
        psum = torch.tensor(psum_cpu, dtype=torch.int32, device='npu')

    sf_k_divisor = 32 if dtype == torch.float32 else 64
    shape = (ceil_div(mn, gran_mn), ceil_div(k, sf_k_divisor))
    if gemm_type == GemmType.Batched:
        shape = (num_groups, *shape)
    sf = with_major(make_sf(shape, dtype), mn_major)
    expected_sf = sf.clone()

    invalid_value = 0 if dtype == torch.int16 else -1
    if gemm_type == GemmType.MGroupedContiguousWithPsumLayout:
        fill_group_padding_(expected_sf.transpose(-2, -1), psum_cpu, alignment, gran_mn, 0)
        fill_group_padding_(sf.transpose(-2, -1), psum_cpu, alignment, gran_mn, invalid_value)
    elif gemm_type == GemmType.KGroupedContiguousWithPsumLayout:
        fill_group_padding_(expected_sf, psum_cpu, alignment, sf_k_divisor, 0)
        fill_group_padding_(sf, psum_cpu, alignment, sf_k_divisor, invalid_value)

    recipe = (gran_mn, gran_mn, 32)
    if gemm_type == GemmType.KGroupedContiguousWithPsumLayout:
        run = partial(deep_gemm.transform_k_grouped_sf_into_required_layout, sf, mn, k, recipe, True, psum)
    else:
        kwargs = {}
        if gemm_type == GemmType.Batched:
            kwargs['num_groups'] = num_groups
        elif gemm_type == GemmType.MGroupedContiguousWithPsumLayout:
            kwargs['psum_layout'] = psum
        run = partial(deep_gemm.transform_sf_into_required_layout, sf, mn, k, recipe, is_sfa=True, **kwargs)
    return sf, run, reference(expected_sf, mn, gran_mn)


def test_sf_layout(skip_prof: bool = False) -> None:
    case_groups = {}
    cases = list(enumerate_sf_layout())
    for mn, _, aligned_ks, psum_layout, num_groups, gran_mn, alignment, dtype, major in enumerate_k_grouped_psum_sf_layout():
        grouped_size = sum(aligned_ks)
        cases.append((GemmType.MGroupedContiguousWithPsumLayout, dtype, major, grouped_size, mn, gran_mn, num_groups, alignment, psum_layout))
        cases.append((GemmType.KGroupedContiguousWithPsumLayout, dtype, major, mn, grouped_size, gran_mn, num_groups, alignment, psum_layout))
    cases.extend(enumerate_sf_layout_performance_tests())
    for case in cases:
        gemm_type, dtype, major, *_ = case
        case_groups.setdefault((dtype, gemm_type, major), []).append(case)

    def run_case(case):
        gemm_type, dtype, major, mn, k, gran_mn, num_groups, _, _ = case
        sf, run, expected = make_layout_case(case)
        output = run()
        torch.npu.synchronize()
        check_layout(output, expected)

        kind = {
            GemmType.Normal: 'normal',
            GemmType.Batched: 'batch',
            GemmType.MGroupedContiguousWithPsumLayout: 'M-group',
            GemmType.KGroupedContiguousWithPsumLayout: 'K-group',
        }[gemm_type]
        dtype = 'FP32' if dtype == torch.float32 else 'short'
        major = 'MN-major' if major == Major.MN else 'K-major'
        tags = [f'mn={mn:7}', f'k={k:6}', f'type={kind}', f'dtype={dtype}', f'major={major}']
        if gemm_type == GemmType.Batched:
            tags.append(f'batches={num_groups:3}')
        elif gemm_type in (GemmType.MGroupedContiguousWithPsumLayout, GemmType.KGroupedContiguousWithPsumLayout):
            tags.append(f'groups={num_groups:3}')
        tags.append(f'gran_mn={gran_mn:3}')
        prefix = f" > Perf ({', '.join(tags)}):"
        prefix = f'{prefix:<100}'
        if skip_prof:
            print(prefix + f'{"OK":>10}')
        elif output.data_ptr() == sf.data_ptr():
            print(prefix + f'{bench_host_us(run):7.1f} us')
        else:
            profile = bench_msprof(
                run, kernel_names='transform_sfI',
                num_warmups=10, num_tests=12, flush_l2=True,
            )
            bandwidth = (
                sf.numel() * sf.element_size() + output.numel() * output.element_size()
            ) / profile.dur_us / 1e6
            print(
                prefix + f'{profile.dur_us:7.1f} us | {bandwidth:7.3f} TB/s | '
                f'vec={profile.aiv_vec * 100:4.1f}% | scalar={profile.aiv_scalar * 100:4.1f}% | '
                f'mte2={profile.aiv_mte2 * 100:4.1f}% | mte3={profile.aiv_mte3 * 100:4.1f}%'
            )

    print('Testing', "transform sf")
    print('Num tests: ', len(cases))
    compile_cases = {}
    for case in cases:
        gemm_type, dtype, major, _, _, gran_mn, _, alignment, _ = case
        compile_cases.setdefault((gemm_type, dtype, major, gran_mn, alignment), case)
    kernels = [make_layout_case(case)[1] for case in compile_cases.values()]
    par_compile(kernels)
    del kernels
    gc.collect()
    torch.npu.empty_cache()

    for cases in case_groups.values():
        for case in cases:
            run_case(case)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('select', nargs='?', default='all', choices=['all', 'correctness', 'performance'])
    parser.add_argument('--skip-prof', '--no-prof', dest='skip_prof', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.manual_seed(0)

    if args.select == 'correctness':
        test_sf_layout(skip_prof=True)
    else:
        test_sf_layout(skip_prof=args.skip_prof)
