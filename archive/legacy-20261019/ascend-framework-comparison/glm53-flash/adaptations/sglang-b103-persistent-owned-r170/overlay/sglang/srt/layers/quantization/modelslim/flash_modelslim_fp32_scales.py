# SPDX-License-Identifier: Apache-2.0
"""Preserve Flash W8A8_DYNAMIC checkpoint FP32 scale/offset values.

Native ModelSlim keeps INT8 weights but allocates these two parameters with
params_dtype. Passing FP32 to the native dynamic scheme changes only their
storage: weight is explicitly INT8 and activation/output dtypes stay native.
"""
import torch
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w8a8_int8 import ModelSlimW8A8Int8


class FlashModelSlimW8A8Int8FP32Scales(ModelSlimW8A8Int8):
    def create_weights(self,layer,input_size_per_partition,output_partition_sizes,
                       input_size,output_size,params_dtype,**extra_weight_attrs):
        if not self.is_dynamic:
            raise ValueError('Flash scale-storage override requires W8A8_DYNAMIC')
        super().create_weights(layer,input_size_per_partition,output_partition_sizes,
                               input_size,output_size,torch.float32,**extra_weight_attrs)
        if layer.weight.dtype!=torch.int8:
            raise TypeError('Flash checkpoint weight must stay INT8')
        if layer.weight_scale.dtype!=torch.float32 or layer.weight_offset.dtype!=torch.float32:
            raise TypeError('Flash checkpoint scale/offset must preserve FP32')
