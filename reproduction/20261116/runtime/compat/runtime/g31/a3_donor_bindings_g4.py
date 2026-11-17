"""Full actual-donor QLI+SMLA four-op binding, rebuilt for target Torch2.13."""
import os
from pathlib import Path
from a3_backend_candidate import A3Bindings


def capabilities():
 return {'profile':'actual donor four-op A3 INT8/BF16 vendor, target Torch2.13 adapter',
         'candidate_source':True,'candidate_consumer':True,'target_index_heads':32,
         'ratios':[1,2],'validation':'static contract only until target compile/operator/graph evidence'}


def make_bindings():
 import torch
 if not hasattr(torch,'npu') or not torch.npu.is_initialized():raise RuntimeError('Initialize target NPU context first')
 vendor=Path(os.environ['SGLANG_V41_CANDIDATE_VENDOR_ROOT']).resolve()
 roots=[str(Path(x).resolve()) for x in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if x]
 if str(vendor) not in roots:raise RuntimeError('Set task-private vendor ASCEND_CUSTOM_OPP_PATH BEFORE starting Python')
 opapi=Path(os.environ['SGLANG_V41_A3_OPAPI_LIBRARY'])
 if not opapi.is_absolute() or not opapi.is_file() or opapi.name!='libsglang_v41_a3_opapi.so':raise ValueError('Expected absolute renamed isolated opapi')
 if (vendor/'op_api/lib/libcust_opapi.so').exists():raise ValueError('Filtered vendor must not expose generic libcust_opapi.so')
 shared=Path(os.environ['SGLANG_V41_CANDIDATE_EXTENSION']).resolve()
 if not shared.is_file():raise ValueError('Expected newly compiled target Torch2.13 g4 extension')
 torch.ops.load_library(str(shared))
 ops=torch.ops.sglang_v41_a3_g4
 return A3Bindings(qli=ops.qli,qli_metadata=ops.qli_metadata,sparse_metadata=ops.sparse_metadata,sparse=ops.sparse)
