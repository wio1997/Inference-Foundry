import importlib, json, runpy, sys, types
from pathlib import Path
import torch
import torch_npu
r=Path('/work/src/DeepInfer/vendor/FlashMLA/tests')
sys.path.insert(0,str(r))
# Load the unmodified numerical/reference helpers without importing their
# CUDA-only Triton benchmark dependency. No benchmark is executed here.
kk=types.ModuleType("kernelkit")
kk.__path__=[str(r/'kernelkit')]
sys.modules['kernelkit']=kk
for name in ('platform','compare','generate','precision','utils'):
    helper=importlib.import_module('kernelkit.'+name)
    for key,value in vars(helper).items():
        if not key.startswith('_'):
            setattr(kk,key,value)
module=runpy.run_path(str(r/'test-sparse-prefill.py'))
from lib import TestParam
cases=[TestParam(1,128,128,h_q=64,num_runs=0,seed=41),
       TestParam(62,592,128,h_q=64,num_runs=0,seed=42),
       TestParam(213,1840,256,h_q=64,num_runs=0,seed=43),
       TestParam(213,153,256,h_q=64,num_runs=0,seed=44,have_attn_sink=True,have_topk_length=True)]
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device('npu:0')
results=[]
for case in cases:
    passed=bool(module['run_test'](case))
    results.append({'s_q':case.s_q,'s_kv':case.s_kv,'topk':case.topk,'passed':passed})
    print(f'FLASHMLA case: {results[-1]}',flush=True)
    assert passed, f'FlashMLA numerical reference mismatch: {case}'
Path('/work/runtime-20261021/flashmla-correctness.json').write_text(json.dumps(results,indent=2))
print('FLASHMLA_SPARSE_PREFILL_REFERENCE_PASS',flush=True)
