"""Build and resolve the standalone op without any NPU device or workload."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

assert not list(Path('/dev').glob('davinci*'))
import torch
import torch_npu
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent.parent
build = root / 'build'
build.mkdir(exist_ok=True)
npu = Path(torch_npu.__file__).resolve().parent
cann = Path('/usr/local/Ascend/cann-9.2.0')
provider = root.parent / 'continuation_G55_fused_moe_20261114/operator_sources/custom_transformer/op_api/lib/libcust_opapi.so'
assert hashlib.sha256(provider.read_bytes()).hexdigest() == (
    '5e7735b6b84626596a5eb7ac6e4b2a028fa227ee8f267f435ca6cd5f4928fbea'
)
os.environ['SGLANG_DSV41_FUSED_MOE_OPAPI'] = str(provider)
os.environ['MAX_JOBS'] = '1'
library = load(
    name='sglang_dsv41_fused_hidden',
    sources=[str(root / 'candidate/fused_moe.cpp'),
             str(root / 'candidate/aclnn_torch_adapter/NPUBridge.cpp'),
             str(root / 'candidate/aclnn_torch_adapter/NPUStorageImpl.cpp')],
    extra_include_paths=[str(npu / 'include'),
                         str(root / 'candidate'),
                         str(cann / 'include')],
    extra_cflags=['-O2', '-std=c++20', '-fvisibility=hidden', '-fvisibility-inlines-hidden', '-fno-gnu-unique'],
    extra_ldflags=[f'-L{npu / "lib"}', '-ltorch_npu', '-ldl'],
    build_directory=str(build),
    is_python_module=False,
    verbose=True,
)
resolved = torch.ops.sglang_dsv41_fused.provider_path()
assert Path(resolved).resolve() == provider.resolve()
schema = str(torch.ops.sglang_dsv41_fused.dispatch.default._schema)
assert 'float swiglu_limit' in schema and 'Tensor? active_mask' in schema
symbols = subprocess.check_output(['readelf', '--dyn-syms', '--wide', str(library)], text=True)
exported_helpers = [line for line in symbols.splitlines() if any(n in line for n in ['GetOpApiFuncAddr', 'ConvertType']) and ('GLOBAL DEFAULT' in line or 'WEAK   DEFAULT' in line or 'UNIQUE DEFAULT' in line)]
assert not exported_helpers, exported_helpers
result = {
    'exported_generic_helpers': exported_helpers,
    'status': 'CPU_build_and_explicit_provider_resolution_passed',
    'library': str(library),
    'library_sha256': hashlib.sha256(Path(library).read_bytes()).hexdigest(),
    'provider': resolved,
    'schema': schema,
    'torch': torch.__version__,
    'torch_npu': torch_npu.__version__,
    'NPU_devices': 0,
    'device_execution_verified': False,
    'numerical_correctness_verified': False,
}
(build / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
