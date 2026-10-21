import json
from pathlib import Path
import torch,torch_npu,deep_gemm
results=[]
for groups,gran in [(None,32),(4,32),(None,1)]:
    mn,k=128,512
    shape=(*(() if groups is None else (groups,)),mn//gran,k//32)
    exponents=(torch.arange(torch.tensor(shape).prod().item()).reshape(shape)%6+123).to(torch.uint8)
    sf=torch.pow(2.,exponents.float()-127).npu()
    actual=deep_gemm.transform_sf_into_required_layout(sf,mn,k,recipe=(1,gran,32),num_groups=groups,is_sfa=False)
    torch.npu.synchronize()
    expected=exponents.repeat_interleave(gran,dim=-2).contiguous().view(torch.int16)
    assert torch.equal(actual.cpu(),expected),(groups,gran,actual.shape,expected.shape)
    results.append(dict(groups=groups,granularity=gran,passed=True))
    print('DG_SCALE_LAYOUT_PASS',results[-1],flush=True)
Path('/work/runtime-20261021/dg-layout-correctness.json').write_text(json.dumps(results,indent=2))
