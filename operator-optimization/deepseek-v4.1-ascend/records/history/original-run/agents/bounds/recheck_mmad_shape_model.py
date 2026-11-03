#!/usr/bin/env python3
"""Reproduce the vendor single-Mmad shape-model arithmetic; no device calls."""
from pathlib import Path
from hashlib import sha256
import json

base = Path(__file__).resolve().parent
root = base.parent.parent
m, n, k = 3, 25600, 6144
cm, cn, ck, k0 = 16, 16, 16, 16
align = lambda v, a: ((v + a - 1) // a) * a
am, an, ak = align(m, 16), align(n, 16), align(k, k0)
parallel = cm * cn * ck
useful = m * n * k
padded = am * an * ak
assert useful % parallel == 0 and padded % parallel == 0
assert (am, an, ak) == (16, 25600, 6144)
assert (useful // parallel, padded // parallel) == (115200, 614400)
assert useful * 16 == padded * 3

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

files = {
    'official_formula_svg': base / 'cann-mmad-formula.svg',
    'rendered_formula_png': base / 'cann-mmad-formula.png',
    'frozen_metrics': base.parent / 'diagnosis/engram-matmul-metrics.json',
    'metric_semantics_check': base.parent / 'diagnosis/engram-cube-mac-formula-check.json',
    'frozen_device_info': root / 'evidence/profile-device-capacity-info.json',
    'this_script': Path(__file__).resolve(),
}
result = {
    'source_url': 'https://asc.gitcode.com/api/appendix/cube_instruction_theoretical_perf_summary.html',
    'architecture_source_url': 'https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html',
    'page_version': 'unversioned web snapshot; not established as identical to installed CANN',
    'scope': 'symbolic one-Mmad formula arithmetic only, not a feasible instruction or an observed kernel',
    'shape': {'m':m,'n':n,'k':k,'cm':cm,'cn':cn,'ck':ck,'k0':k0},
    'padded_shape': {'m':am,'n':an,'k':ak},
    'useful_mac_logical': useful,
    'padded_mac_model': padded,
    'formula_numerator_cycles': useful // parallel,
    'formula_padded_cycles': padded // parallel,
    'utilization_formula': '115200 / (delta_t_cycles + 614400)',
    'delta_t_zero_utilization_pct': 100 * useful / padded,
    'one_instruction_l0_bytes': {'A_bf16': am*ak*2, 'B_bf16': ak*an*2, 'C_fp32': am*an*4},
    '2201_per_core_buffer_bytes': {'A': 64*1024, 'B':64*1024,'C':128*1024},
    'limits': ['whole-shape one Mmad exceeds all three per-core L0 buffers; must tile', 'ratio is not actual utilization or universal bound under fusion, batching or alternative algorithms', 'profiler aic_mac_ratio and cube_utilization use different definitions', 'page math SVG preserved; raw full-page exact release and original build correspondence unknown'],
    'sha256': {name:digest(path) for name,path in files.items()},
    'no_npu_calls': True,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
