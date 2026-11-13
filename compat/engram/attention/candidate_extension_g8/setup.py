"""Build on the TARGET CPU with its Torch2.13; never on the client VM.
SGLANG_V41_CANN_ROOT must identify current CANN9.2. Does not install system-wide.
"""
import os
from pathlib import Path
from setuptools import setup
import torch
import torch_npu
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

if not torch.__version__.startswith('2.13') or not torch_npu.__version__.startswith('2.13'):
    raise RuntimeError('This isolated adapter must be built with target torch/torch_npu2.13')
root=Path(__file__).resolve().parent
cann=Path(os.environ['SGLANG_V41_CANN_ROOT']).resolve()
if not (cann/'include/acl/acl.h').is_file():raise ValueError('Invalid explicit current CANN root')
setup(name='sglang_v41_a3_candidate_adapter',
      ext_modules=[NpuExtension(name='_sglang_v41_a3_g4',sources=[str(root/'single_op.cpp')],
          include_dirs=[str(root),str(cann/'include')],library_dirs=[str(cann/'lib64')],
          libraries=['ascendcl','dl'],extra_compile_args=['-O2','-g0','-std=c++17'])],
      cmdclass={'build_ext':BuildExtension})
