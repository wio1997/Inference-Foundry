from functools import cache
import torch
import triton

@cache
def get_vectorcore_num():
    props=triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
    count=props.get('num_vectorcore',-1)
    if count<=0: raise RuntimeError('Triton Ascend did not report vectorcore count')
    return count
