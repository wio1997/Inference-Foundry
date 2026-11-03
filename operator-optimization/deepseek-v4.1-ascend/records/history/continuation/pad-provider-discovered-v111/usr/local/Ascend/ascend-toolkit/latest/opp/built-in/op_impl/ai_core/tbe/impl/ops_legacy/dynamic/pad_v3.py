#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
pad_v3
"""
import functools
from impl.ops_legacy.util.platform_adapter import para_check
from impl.ops_legacy.util.platform_adapter import tik
from impl.ops_legacy.util.platform_adapter import tbe_platform
from impl.ops_legacy.util.platform_adapter import tbe_context
from impl.ops_legacy.util.platform_adapter import register_operator
from impl.ops_legacy import constant_util as constant
from impl.ops_legacy.dynamic.reflection_pad_v3 import reflection_pad_v3
from impl.ops_legacy.dynamic.replication_pad_v3 import replication_pad_v3
from impl.ops_legacy.dynamic.pad_v3_5hd import pad_v3_5hd
from impl.ops_legacy.util import util_select_op_base
from tbe.common.platform import get_bit_len
from tbe.dsl.unify_schedule.util import is_nano, is_tiny


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    CONSTANT_PAD_MAX_SIZE = 8
    MAX_INT64 = 2**64 - 1
    MAX_INT32 = 2**31 - 1
    TILING_NUMS = 28
    EIGHT_BIT = 8
    BLOCK_BYTES = 32
    TWO_BYTES = 2
    TWO_PIECES = 2
    FOUR_PIECES = 4
    ONE_BYTES = 1
    RESERVED_UB = 1024
    TRANS_MIN_BLKS = 16
    ONE_BYTES_DTYPE_BLOCK_NUM = 32
    TWO_BYTES_DTYPE_BLOCK_NUM = 16
    MODE0 = 0
    MODE1 = 1
    MODE2 = 2
    MODE3 = 3
    MODE4 = 4
    MODE5 = 5
    PADV3_DATA_MOVE_LAST_DIM_CUT = 6
    LAST_DIM_PAD_MODE = 7
    PADV3_INT8_HIGH_PROFROMANCE = 8
    MASK_TWO_BYTES_DTYPE = 128
    THRESHOLD_VALUE = 8192
    # Optimal repeat num to fill output gm
    FILL_NUM = 256
    BLOCK = 32
    RESERVE_FOR_FP32_VNCHWCONV_MODE_2 = 11 * 1024 * 8
    RESERVE_FOR_FP32_VNCHWCONV_MODE_5 = 11 * 1024 * 2
    # max_burst_len  65525//32*32
    MAX_UB_CNT_DATA_MOVE_PAD = 65504


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,too-many-statements,too-many-branches
def check_supported(x, paddings, constant_values, y, mode, paddings_contiguous, kernel_name="pad_v3"):
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    if int(-1) in x_shape or int(-2) in x_shape:
        return "Unknown"
    if len(x_shape) > Constant.CONSTANT_PAD_MAX_SIZE:
        reason = "input shape does not support more than 8 dimensions"
        return False, reason
    if mode == "constant" and x_dtype not in ["int8", "float16", "bfloat16", "float", "int32"]:
        reason = "constant mode, input dtype must in [int8, float16, bfloat16, float, int32]"
        return False, reason
    if mode in ["reflect", "edge"] and x_dtype not in ["float16", "bfloat16", "float", "int32"]:
        reason = "reflect or edge mode, input dtype must in [float16, bfloat16, float, int32]"
        return False, reason
    if mode not in ["constant", "reflect", "edge"]:
        reason = "mode only support [constant, reflect, edge]"
        return False, reason
    return True, ""


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,too-many-statements,too-many-branches
def op_select_format(x, paddings, constant_values, y, mode, paddings_contiguous, kernel_name="pad_v3"):
    """
    when the attr mode is constant, the PadV3 can support HC1HWC0 and ND.
    """
    version_info = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
    if (tbe_platform.api_check_support("tik.vcopy") or version_info == "Ascend310P") and mode == "constant":
        dtype_x = ["int8", "float16", "bfloat16", "float", "int32", 
                   "int8", "float16", "bfloat16", "float", "int32"]
        dtype_paddings = ["int32", "int32", "int32", "int32", "int32", 
                          "int64", "int64", "int64", "int64", "int64"]
    else:
        dtype_x = ["float16", "bfloat16", "float", "int32", 
                   "float16", "bfloat16", "float", "int32"]
        dtype_paddings = ["int32", "int32", "int32", "int32",
                          "int64", "int64", "int64", "int64"]
    format_tensor = ["ND"] * len(dtype_x)
    format_scalar = format_tensor
    if mode == 'constant':
        format_scalar = format_scalar + format_scalar
        format_tensor = format_tensor + ["NC1HWC0"] * len(dtype_x)
        dtype_x = dtype_x + dtype_x
        dtype_paddings = dtype_paddings + dtype_paddings

    dtype_str_x = ','.join(dtype_x)
    dtype_str_paddings = ','.join(dtype_paddings)
    format_str_scalar = ','.join(format_scalar)
    format_str_tensor = ','.join(format_tensor)

    input0 = util_select_op_base.gen_param(
        classify="input0", name="x", datatype=dtype_str_x, format=format_str_tensor,
        unknownshape_format=format_str_tensor)
    input1 = util_select_op_base.gen_param(
        classify="input1", name="paddings", datatype=dtype_str_paddings, format=format_str_scalar,
        unknownshape_format=format_str_scalar)
    input2 = util_select_op_base.gen_param(
        classify="input2", name="constant_values", datatype=dtype_str_x, format=format_str_scalar,
        unknownshape_format=format_str_scalar)
    output0 = util_select_op_base.gen_param(
        classify="output0", name="y", datatype=dtype_str_x, format=format_str_tensor,
        unknownshape_format=format_str_tensor)
    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# 'pylint: disable=too-many-instance-attributes,too-many-statements,too-many-locals,too-many-lines
# 'pylint: disable=too-many-arguments,invalid-name
class PadV3Init:
    """
    Function: class that execute pad_v3
    """

    def __init__(self, x, paddings, constant_values, y, mode='constant', padding_contiguous=True, kernel_name='pad_v3'):
        """
        init the op
        :param
        x: the input tensor
        :param
        paddings: the list of paddings
        :param
        constant_values: the value to fill the tensor
        :param
        y: the output of op
        :param
        mode: the fill mode
        :param
        padding_contiguous: is the memory is contiguous
        :param
        kernel_name: the kernel name of op
        :return
        None
        """

        self.tik_instance = tik.Tik()
        self.unknown_max_shape = (Constant.MAX_INT64,)
        self.tiling_dtype = "int64"
        self.tiling_shape = (Constant.TILING_NUMS,)
        self.one_bytes_dtype = {"int8", "uint8"}
        self.four_bytes_dtype = {"float32", "int32", "uint32"}
        self.two_bytes_dtype = {"bfloat16", "float16", "int16", "uint16"}
        self.x_dtype = x.get("dtype")
        self.scalar_dtype = "int64"
        if self.x_dtype in self.four_bytes_dtype:
            self.scalar_dtype = self.x_dtype
        elif self.x_dtype in self.two_bytes_dtype:
            self.x_dtype = "float16"
        self.paddings_dtype = paddings.get('dtype')
        self.constant_values = constant_values
        self.inner_dtype = "float16"
        # MDC 610 compute grouping avoiding patch DTS:2022022105256
        self.version_info = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
        self.kernel_name = kernel_name
        self.input_gm = None
        self.output_gm = None
        self.tiling_gm = None
        self.sync_workspace = None
        self.special_max_output_size = None
        self.input_gm_list = []
        self.output_gm_list = []
        self.input_bytes_size = 0
        self.inner_bytes_size = get_bit_len(self.inner_dtype) // Constant.EIGHT_BIT
        self.x_bytes_size = get_bit_len(self.x_dtype) // Constant.EIGHT_BIT
        self.block_num = constant.BLOCK_SIZE // self.x_bytes_size
        self.inner_block_num = constant.BLOCK_SIZE // self.inner_bytes_size
        self.dump_mask_max_x = 8 * self.block_num

        self.total_ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.ub_size_bytes = self.total_ub_size - Constant.RESERVED_UB
        self.ub_number = self.ub_size_bytes // self.x_bytes_size
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        # default copy data number in one time
        if is_tiny():
            self.copy_num = (self.ub_size_bytes - Constant.RESERVED_UB) // self.x_bytes_size // 6
        else:
            if self.x_dtype in self.two_bytes_dtype:
                self.copy_num = 6400
            else:
                self.copy_num = 3200
        self.max_numel_vec_dup_one_loop = None
        self.pad_scalar = self.tik_instance.Scalar(dtype=self.x_dtype, name='pad_scalar')
        # since the vnchwconv api does not support processing int8 dtype
        # the padding value also needs to be converted to fp16
        self.pad_scalar_int8 = self.tik_instance.Scalar("float16", name='pad_scalar_int8')
        if self.constant_values:
            self.constant_values_gm = self.tik_instance.Tensor(self.x_dtype, (self.block_num,),
                                                               name='constant_values_gm',
                                                               scope=tik.scope_gm)
        self.mode = mode
        self.padding_contiguous = padding_contiguous
        # tiling scaler init
        self.tiling_key = self.tik_instance.Scalar(self.tiling_dtype, "tiling_key", init_value=0)
        self.tiling_input_dim_0 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_0", init_value=0)
        self.tiling_input_dim_1 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_1", init_value=0)
        self.tiling_input_dim_2 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_2", init_value=0)
        self.tiling_input_dim_3 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_3", init_value=0)
        self.tiling_input_dim_4 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_4", init_value=0)
        self.tiling_input_dim_5 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_5", init_value=0)
        self.tiling_input_dim_6 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_6", init_value=0)
        self.tiling_input_dim_7 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_7", init_value=0)
        self.tiling_pading_00 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_00", init_value=0)
        self.tiling_pading_01 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_01", init_value=0)
        self.tiling_pading_10 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_10", init_value=0)
        self.tiling_pading_11 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_11", init_value=0)
        self.tiling_pading_20 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_20", init_value=0)
        self.tiling_pading_21 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_21", init_value=0)
        self.tiling_pading_30 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_30", init_value=0)
        self.tiling_pading_31 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_31", init_value=0)
        self.tiling_pading_40 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_40", init_value=0)
        self.tiling_pading_41 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_41", init_value=0)
        self.tiling_pading_50 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_50", init_value=0)
        self.tiling_pading_51 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_51", init_value=0)
        self.tiling_pading_60 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_60", init_value=0)
        self.tiling_pading_61 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_61", init_value=0)
        self.tiling_pading_70 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_70", init_value=0)
        self.tiling_pading_71 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_71", init_value=0)
        self.tiling_input_dim_cut_axis = self.tik_instance.Scalar(self.tiling_dtype,
                                                                  "tiling_input_dim_cut_axis",
                                                                  init_value=0)
        self.tiling_output_dim_0 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_0", init_value=0)
        self.tiling_output_dim_1 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_1", init_value=0)
        self.tiling_output_dim_2 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_2", init_value=0)
        self.tiling_output_dim_3 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_3", init_value=0)
        self.tiling_output_dim_4 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_4", init_value=0)
        self.tiling_output_dim_5 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_5", init_value=0)
        self.tiling_output_dim_6 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_6", init_value=0)
        self.tiling_output_dim_7 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_7", init_value=0)
        self.tiling_core_num = self.tik_instance.Scalar(self.tiling_dtype, "tiling_core_num", init_value=self.core_nums)
        
        self.tiling_input_shape = [
            self.tiling_input_dim_0, self.tiling_input_dim_1, self.tiling_input_dim_2, self.tiling_input_dim_3,
            self.tiling_input_dim_4, self.tiling_input_dim_5, self.tiling_input_dim_6, self.tiling_input_dim_7
        ]
        self.tiling_output_shape = [
            self.tiling_output_dim_0, self.tiling_output_dim_1, self.tiling_output_dim_2, self.tiling_output_dim_3,
            self.tiling_output_dim_4, self.tiling_output_dim_5, self.tiling_output_dim_6, self.tiling_output_dim_7
        ]
        self.tiling_pading_value = [[self.tiling_pading_00, self.tiling_pading_01],
                                    [self.tiling_pading_10, self.tiling_pading_11],
                                    [self.tiling_pading_20, self.tiling_pading_21],
                                    [self.tiling_pading_30, self.tiling_pading_31],
                                    [self.tiling_pading_40, self.tiling_pading_41],
                                    [self.tiling_pading_50, self.tiling_pading_51],
                                    [self.tiling_pading_60, self.tiling_pading_61],
                                    [self.tiling_pading_70, self.tiling_pading_71]]
        self.input_offset = []
        self.output_offset = []

        # core scaler init
        self.core_outer_num = self.tik_instance.Scalar(self.tiling_dtype, "core_outer_num", init_value=0)
        self.core_outer_start = self.tik_instance.Scalar(self.tiling_dtype, "core_outer_start", init_value=0)
        self.core_inner_num = self.tik_instance.Scalar(self.tiling_dtype, "core_inner_num", init_value=0)
        self.core_inner_start = self.tik_instance.Scalar(self.tiling_dtype, "core_inner_start", init_value=0)
        self.high_performance_branch = self.tik_instance.Scalar("int8", "high_performance_branch", init_value=0)
        
    def get_pad_scalar(self):
        """
        get_pad_scalar
        """
        constant_values_ub = self.tik_instance.Tensor(self.x_dtype, (self.block_num,),
                                                      name='constant_values_ub',
                                                      scope=tik.scope_ubuf)
        constant_values_ub_fp16 = self.tik_instance.Tensor("float16", (16,),
                                                      name='constant_values_ub',
                                                      scope=tik.scope_ubuf)
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_instance.data_move_pad(constant_values_ub, self.constant_values_gm,
                                            nburst=1, burst=self.x_bytes_size,
                                            dst_gap=0, src_gap=0,
                                            right_padding=0, left_padding=0,
                                            padding_value=None)
        else:
            self.tik_instance.data_move(constant_values_ub, self.constant_values_gm, 0, 1, 1, 0, 0)
        self.pad_scalar.set_as(constant_values_ub[0])
        if self.x_dtype in self.one_bytes_dtype:
            self.tik_instance.vec_conv(16, '', constant_values_ub_fp16, constant_values_ub, 1, 0, 0)
            self.pad_scalar_int8.set_as(constant_values_ub_fp16[0])

    def core_schedule_args(self, core_index):
        """
        core_schedule_args
        """
        if not is_nano():
            with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 0):
                core_outer_all = self.tiling_input_shape[-1]
                self.core_outer_num.set_as((core_outer_all + self.tiling_core_num - 1) // self.tiling_core_num)
                self.core_outer_num.set_as((self.core_outer_num + self.block_num - 1) // self.block_num)
                self.core_outer_num.set_as(self.core_outer_num * self.block_num)
                self.core_outer_start.set_as(core_index * self.core_outer_num)
                with self.tik_instance.if_scope(self.core_outer_start + self.core_outer_num > core_outer_all):
                    self.core_outer_num.set_as(core_outer_all - self.core_outer_start)
                    self.tik_instance.scalar_max(self.core_outer_num, self.core_outer_num, 0)
                    with self.tik_instance.if_scope(self.core_outer_num % self.block_num != 0):
                        self.core_outer_num.set_as((self.core_outer_num + self.block_num - 1) // self.block_num)
                        self.core_outer_num.set_as(self.core_outer_num * self.block_num)
                    self.core_outer_start.set_as(core_outer_all - self.core_outer_num)
                    with self.tik_instance.if_scope(self.core_outer_start < 0):
                        self.core_outer_num.set_as(core_outer_all)
        with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 1):
            core_outer_all = functools.reduce(lambda x, y: x * y, self.tiling_input_shape[:7])
            self.core_outer_num.set_as((core_outer_all + self.tiling_core_num - 1) // self.tiling_core_num)
            self.core_outer_start.set_as(core_index * self.core_outer_num)
            with self.tik_instance.if_scope(core_outer_all % self.tiling_core_num != 0):
                with self.tik_instance.if_scope(core_index >= core_outer_all % self.tiling_core_num):
                    self.core_outer_num.set_as(self.core_outer_num - 1)
                    self.core_outer_start.set_as(core_index * self.core_outer_num + \
                        core_outer_all % self.tiling_core_num)
        with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 2):
            core_outer_all = functools.reduce(lambda x, y: x * y, self.tiling_input_shape[:6])
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.core_outer_num.set_as((core_outer_all + self.tiling_core_num - 1) // self.tiling_core_num)
                self.core_outer_start.set_as(core_index * self.core_outer_num)
                with self.tik_instance.if_scope(core_outer_all % self.tiling_core_num != 0):
                    with self.tik_instance.if_scope(core_index >= core_outer_all % self.tiling_core_num):
                        self.core_outer_num.set_as(self.core_outer_num - 1)
                        self.core_outer_start.set_as(core_index * self.core_outer_num + \
                            core_outer_all % self.tiling_core_num)
            else:
                with self.tik_instance.if_scope(self.tiling_output_dim_7 * self.tiling_input_dim_6 < self.block_num):
                    # the last two is less one block, only can process use one core
                    self.core_outer_num.set_as(0)
                    self.core_outer_start.set_as(0)
                    with self.tik_instance.if_scope(core_index == 0):
                        self.core_outer_num.set_as(core_outer_all)
                with self.tik_instance.else_scope():
                    self.core_outer_num.set_as((core_outer_all + self.tiling_core_num - 1) // self.tiling_core_num)
                    self.core_outer_start.set_as(core_index * self.core_outer_num)
                    with self.tik_instance.if_scope(core_outer_all % self.tiling_core_num != 0):
                        with self.tik_instance.if_scope(core_index >= core_outer_all % self.tiling_core_num):
                            self.core_outer_num.set_as(self.core_outer_num - 1)
                            self.core_outer_start.set_as(core_index * self.core_outer_num + \
                                core_outer_all % self.tiling_core_num)
        with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 3):
            core_outer_all = functools.reduce(lambda x, y: x * y, self.tiling_input_shape[:5])
            self.core_outer_num.set_as((core_outer_all + self.tiling_core_num - 1) // self.tiling_core_num)
            self.core_outer_start.set_as(core_index * self.core_outer_num)
            with self.tik_instance.if_scope(core_outer_all % self.tiling_core_num != 0):
                with self.tik_instance.if_scope(core_index >= core_outer_all % self.tiling_core_num):
                    self.core_outer_num.set_as(self.core_outer_num - 1)
                    self.core_outer_start.set_as(core_index * self.core_outer_num + \
                        core_outer_all % self.tiling_core_num)
        for i, _ in enumerate(self.tiling_input_shape):
            scalar = self.tik_instance.Scalar(self.tiling_dtype, "input_offset_" + str(i), init_value=0)
            scalar.set_as(functools.reduce(lambda x, y: x * y, self.tiling_input_shape[i:]))
            self.input_offset.append(scalar)
        for i, _ in enumerate(self.tiling_output_shape):
            scalar = self.tik_instance.Scalar(self.tiling_dtype, "output_offset_" + str(i), init_value=0)
            scalar.set_as(functools.reduce(lambda x, y: x * y, self.tiling_output_shape[i:]))
            self.output_offset.append(scalar)

    def tiling_args(self):
        """
        when input shape is less 6, will. expand to 6
        tiling info:
            tiling_key:
            tiling_input_dim_0
            tiling_input_dim_1
            tiling_input_dim_2
            tiling_input_dim_3
            tiling_input_dim_4
            tiling_input_dim_5
            tiling_input_dim_6
            tiling_input_dim_7
            tiling_pading_00
            tiling_pading_01
            tiling_pading_10
            tiling_pading_11
            tiling_pading_20
            tiling_pading_21
            tiling_pading_30
            tiling_pading_31
            tiling_pading_40
            tiling_pading_41
            tiling_pading_50
            tiling_pading_51
            tiling_pading_60
            tiling_pading_61
            tiling_pading_70
            tiling_pading_71
            tiling_input_dim_cut_axis: which dim will be cut
            tiling_core_num
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_NUMS,),
                                                    name="tiling_ub",
                                                    scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.TILING_NUMS // 4, 0, 0)
            self.tiling_key.set_as(tiling_ub[0])
            self.tiling_input_dim_0.set_as(tiling_ub[1])
            self.tiling_input_dim_1.set_as(tiling_ub[2])
            self.tiling_input_dim_2.set_as(tiling_ub[3])
            self.tiling_input_dim_3.set_as(tiling_ub[4])
            self.tiling_input_dim_4.set_as(tiling_ub[5])
            self.tiling_input_dim_5.set_as(tiling_ub[6])
            self.tiling_input_dim_6.set_as(tiling_ub[7])
            self.tiling_input_dim_7.set_as(tiling_ub[8])
            self.tiling_pading_00.set_as(tiling_ub[9])
            self.tiling_pading_01.set_as(tiling_ub[10])
            self.tiling_pading_10.set_as(tiling_ub[11])
            self.tiling_pading_11.set_as(tiling_ub[12])
            self.tiling_pading_20.set_as(tiling_ub[13])
            self.tiling_pading_21.set_as(tiling_ub[14])
            self.tiling_pading_30.set_as(tiling_ub[15])
            self.tiling_pading_31.set_as(tiling_ub[16])
            self.tiling_pading_40.set_as(tiling_ub[17])
            self.tiling_pading_41.set_as(tiling_ub[18])
            self.tiling_pading_50.set_as(tiling_ub[19])
            self.tiling_pading_51.set_as(tiling_ub[20])
            self.tiling_pading_60.set_as(tiling_ub[21])
            self.tiling_pading_61.set_as(tiling_ub[22])
            self.tiling_pading_70.set_as(tiling_ub[23])
            self.tiling_pading_71.set_as(tiling_ub[24])
            self.tiling_input_dim_cut_axis.set_as(tiling_ub[25])
            self.tiling_core_num.set_as(tiling_ub[26])

            # calcu output_dim
            for i, _ in enumerate(self.tiling_input_shape):
                input_dims = self.tiling_input_shape[i]
                pad_left = self.tiling_pading_value[i][0]
                pad_right = self.tiling_pading_value[i][1]
                output_dims = self.tiling_output_shape[i]
                output_dims.set_as(input_dims + pad_left + pad_right)

        if self.core_nums > 1 and self.version_info != "Ascend610" and self.version_info != "BS9SX1A":
            self.sync_workspace = self.tik_instance.Tensor('int64', (self.core_nums * 4,),
                                                            tik.scope_gm,
                                                            'sync_workspace',
                                                            is_workspace=True,
                                                            is_atomic_add=True)
        if not self.constant_values:
            self.pad_scalar.set_as(0)
        else:
            self.get_pad_scalar()
            
        if (self.version_info == "Ascend910" or self.version_info == "Ascend310") \
            and self.x_dtype in self.four_bytes_dtype:
            tail_reserve_ub = 16 * 16
            output_ub_cut_num = 6
            max_line_in_ub = 16
            ori_max_output_size = (self.ub_size_bytes - self.x_bytes_size * tail_reserve_ub * 2) // \
                                (output_ub_cut_num * max_line_in_ub * self.x_bytes_size)
            self.special_max_output_size = (ori_max_output_size // max_line_in_ub) * max_line_in_ub
            with self.tik_instance.if_scope(tik.all(self.pad_scalar == 0, 
                                                    self.special_max_output_size >= self.tiling_output_shape[-1],
                                                    self.tiling_key != Constant.MODE4)):
                self.high_performance_branch.set_as(1)
                self.tiling_input_dim_cut_axis.set_as(2)

    def init_src_dst_gm(self, input_dict_list, pad_input_idx=0, pad_outnput_idx=0):
        """
        init gm tensor set tiling, input, paddings output tensor(gm)
        :param
        input_dict_list: the dict of input_dict
        :param
        pad_input_idx: pad_input_idx
        :param
        pad_outnput_idx: pad_outnput_idx
        :return:
        None
        """
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype,
                                                  self.tiling_shape,
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        x_dtype = input_dict_list[0].get("dtype")
        paddings_dtype = input_dict_list[1].get("dtype")
        x_gm = self.tik_instance.Tensor(self.x_dtype, self.unknown_max_shape, name="x", scope=tik.scope_gm)
        paddings_gm = self.tik_instance.Tensor(paddings_dtype,
                                               self.unknown_max_shape,
                                               name="paddings",
                                               scope=tik.scope_gm)
        self.input_gm_list.append(x_gm)
        self.input_gm_list.append(paddings_gm)
        if self.constant_values is not None:
            self.input_gm_list.append(self.constant_values_gm)

        atomic_flag = False
        if self.version_info == "Ascend610" or self.version_info == "BS9SX1A":
            atomic_flag = True
        y_gm = self.tik_instance.Tensor(self.x_dtype, self.unknown_max_shape, name="y",
                                        scope=tik.scope_gm, is_atomic_add = atomic_flag)
        self.input_bytes_size = get_bit_len(x_dtype) // Constant.EIGHT_BIT
        self.output_gm_list.append(y_gm)

        self.input_gm = self.input_gm_list[pad_input_idx]
        self.output_gm = self.output_gm_list[pad_outnput_idx]

    def calc_core_args(self, core_index, output_shape):
        """
        calculate args for each core
        """
        core_offset = self.tik_instance.Scalar(dtype="int64", name="core_offset", init_value=0)
        offset_index = self.tik_instance.Scalar(dtype="int64", name="offset_index", init_value=0)
        with self.tik_instance.if_scope(self.tiling_core_num > 1):
            core_offset.set_as((output_shape + self.tiling_core_num - 1) // self.tiling_core_num)
            core_offset.set_as((core_offset + self.block_num - 1) // self.block_num)
            core_offset.set_as(core_offset * self.block_num)
            offset_index.set_as(core_index * core_offset)
            with self.tik_instance.if_scope((offset_index + core_offset) > output_shape):
                core_offset.set_as(0)
                with self.tik_instance.if_scope(offset_index < output_shape):
                    core_offset.set_as(output_shape - offset_index)
                    core_offset.set_as((core_offset + self.block_num - 1) // self.block_num)
                    core_offset.set_as(core_offset * self.block_num)
                    offset_index.set_as(output_shape - core_offset)
                    with self.tik_instance.if_scope(offset_index < 0):
                        offset_index.set_as(0)
        with self.tik_instance.else_scope():
            offset_index.set_as(0)
            core_offset.set_as(output_shape)
        return offset_index, core_offset

    def fill_gm_output_tensor(self, core_index):
        with self.tik_instance.new_stmt_scope():
            fill_gm_ub_shape = Constant.THRESHOLD_VALUE // self.input_bytes_size
            output_shape = self.tik_instance.Scalar(dtype="int64", name="output_shape", init_value=0)
            output_shape.set_as(functools.reduce(lambda x, y: x * y, self.tiling_output_shape))
            fill_gm_ub = self.tik_instance.Tensor(self.x_dtype, (fill_gm_ub_shape,),
                                                  name="fill_gm_ub",
                                                  scope=tik.scope_ubuf)

            if self.input_bytes_size == Constant.ONE_BYTES:
                fill_gm_ub_fp16 = self.tik_instance.Tensor("float16", (fill_gm_ub_shape,),
                                                           name="fill_gm_ub_fp16", scope=tik.scope_ubuf)
                pad_scalar_fp16 = self.tik_instance.Scalar("float16")
                pad_scalar_fp16.set_as(self.pad_scalar)
                self.tik_instance.vector_dup(Constant.FILL_NUM // Constant.TWO_BYTES, fill_gm_ub_fp16,
                                             pad_scalar_fp16, Constant.BLOCK_BYTES * 2, 1, 8)
                self.tik_instance.vec_conv(Constant.FILL_NUM // Constant.TWO_BYTES, '',
                                           fill_gm_ub, fill_gm_ub_fp16,
                                           Constant.THRESHOLD_VALUE // Constant.FILL_NUM * Constant.TWO_BYTES,
                                           4, 8)
            else:
                self.tik_instance.vector_dup(Constant.FILL_NUM // self.input_bytes_size, fill_gm_ub,
                                             self.pad_scalar, Constant.BLOCK_BYTES, 1, 8)
            offset_index, core_offset = self.calc_core_args(core_index, output_shape)
            core_offset_block_num = self.tik_instance.Scalar(dtype="int64", name="core_offset_block_num", init_value=0)
            data_move_loop_num = self.tik_instance.Scalar(dtype="int64", name="data_move_loop_num", init_value=0)
            data_move_loop_left = self.tik_instance.Scalar(dtype="int64", name="data_move_loop_left", init_value=0)

            core_offset_block_num.set_as((core_offset + self.block_num - 1) // self.block_num)
            data_move_loop_num.set_as(core_offset_block_num // Constant.FILL_NUM)
            data_move_loop_left.set_as(core_offset_block_num % Constant.FILL_NUM)
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                with self.tik_instance.for_range(0, data_move_loop_num) as loop_index:
                    core_offset_loop = loop_index * Constant.FILL_NUM * self.block_num + offset_index
                    self.tik_instance.data_move(self.output_gm[core_offset_loop], fill_gm_ub, 0, 1, Constant.FILL_NUM,
                                                0, 0)
                with self.tik_instance.if_scope(data_move_loop_left > 0):
                    core_offset_loop = core_offset + offset_index - data_move_loop_left * self.block_num
                    with self.tik_instance.if_scope(core_offset_loop > 0):
                        self.tik_instance.data_move(self.output_gm[core_offset_loop], fill_gm_ub, 0, 1,
                                                    data_move_loop_left, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(self.output_gm, fill_gm_ub, 0, 1,
                                                    data_move_loop_left, 0, 0)   

    def pad_v3_d_compute_tiling(self):
        """
        pad_v3 operation
        """
        self.tiling_args()
        with self.tik_instance.for_range(0, self.tiling_core_num, block_num=self.tiling_core_num) as core_index:
            self.core_schedule_args(core_index)
            with self.tik_instance.if_scope(self.tiling_key != Constant.MODE4):
                if self.version_info != "Ascend610" and self.version_info != "BS9SX1A":
                    self.fill_gm_output_tensor(core_index)
                    if self.core_nums > 1:
                        self.tik_instance.block_barrier(self.sync_workspace)
            self.do_pad(core_index)

    def get_output_outer_idx(self, in_idx, outer_num=7):
        """
        get_output_outer_idx use in_idx
        """
        input_dim_0 = in_idx // self.input_offset[1]
        input_dim_1 = (in_idx % self.input_offset[1]) // self.input_offset[2]
        input_dim_2 = (in_idx % self.input_offset[2]) // self.input_offset[3]
        input_dim_3 = (in_idx % self.input_offset[3]) // self.input_offset[4]
        input_dim_4 = (in_idx % self.input_offset[4]) // self.input_offset[5]
        input_dim_5 = (in_idx % self.input_offset[5]) // self.input_offset[6]
        input_dim_6 = (in_idx % self.input_offset[6]) // self.input_offset[7]
        input_dim_7 = in_idx % self.input_offset[7]


        input_list = [input_dim_0, input_dim_1, input_dim_2, input_dim_3, 
                      input_dim_4, input_dim_5, input_dim_6, input_dim_7]
        output_list = []
        for i, _ in enumerate(self.tiling_input_shape):
            input_dims = input_list[i]
            pad_left = self.tiling_pading_value[i][0]
            output_dims = input_dims + pad_left
            output_list.append(output_dims)

        output_idx = 0
        if (outer_num <= 7):
            for i in range(outer_num):
                output_idx = output_idx + output_list[i] * self.output_offset[i + 1]
        else:
            for i in range(7):
                output_idx = output_idx + output_list[i] * self.output_offset[i + 1]
            output_idx += output_list[7]
        return output_idx

    def data_move(self, gm_src_info, gm_dst_info, copy_len, used_ub):
        """
        func for data_move
        :param
        gm_src_info:gm_src_info
        :param
        gm_dst_info:gm_dst_info
        :param
        copy_len:copy_len
        :param
        used_ub:used_ub
        :return:
        None
        """
        input_gm, input_offset = gm_src_info
        output_gm, output_offset = gm_dst_info
        bursn_len = (copy_len + self.block_num - 1) // self.block_num
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_instance.data_move_pad(used_ub, input_gm[input_offset], 1, copy_len * self.x_bytes_size, 0, 0)
            self.tik_instance.data_move_pad(output_gm[output_offset], used_ub, 1, copy_len * self.x_bytes_size, 0, 0)
        else:
            self.tik_instance.data_move(used_ub, input_gm[input_offset], 0, 1, bursn_len, 0, 0)
            self.tik_instance.data_move(output_gm[output_offset], used_ub, 0, 1, bursn_len, 0, 0)

    def data_move_with_mask_less_block(self, gm_src_info, gm_dst_info, copy_len, used_ub, ub_one_block):
        """
        func for data_move_with_mask
        """
        input_gm, input_offset = gm_src_info
        output_gm, output_offset = gm_dst_info
        if tbe_platform.api_check_support("tik.data_move_pad"):
            self.tik_instance.data_move_pad(used_ub, input_gm[input_offset], 1, copy_len * self.x_bytes_size, 0, 0)
            self.tik_instance.data_move_pad(output_gm[output_offset], used_ub, 1, copy_len * self.x_bytes_size, 0, 0)
        else:
            bursn_len = (copy_len + self.block_num - 1) // self.block_num
            self.tik_instance.data_move(ub_one_block, input_gm[input_offset], 0, 1, bursn_len, 0, 0)
            vnchw_src_list = [ub_one_block] * Constant.TRANS_MIN_BLKS
            vnchw_dst_list = [used_ub[i * self.block_num] for i in range(Constant.TRANS_MIN_BLKS)]
            self.tik_instance.vnchwconv(False, False, vnchw_dst_list, vnchw_src_list, 1, 0, 0)
            burst_num = 1
            burst_len = self.block_num - copy_len + 1
            pad_dim = 16
            self.tik_instance.data_move(used_ub[copy_len * pad_dim:], ub_one_block[self.block_num:], 0, burst_num,
                                        burst_len, 0, 0)
            vnchw_src_list = [used_ub[i * Constant.TRANS_MIN_BLKS] for i in range(Constant.TRANS_MIN_BLKS)]
            vnchw_dst_list = [used_ub[i * Constant.TRANS_MIN_BLKS + Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS] 
                            for i in range(Constant.TRANS_MIN_BLKS)]
            self.tik_instance.vnchwconv(False, False, vnchw_dst_list, vnchw_src_list, 1, 0, 0)
            self.tik_instance.data_move(output_gm[output_offset],
                                    used_ub[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS], 0, 1, bursn_len, 0, 0)

    def do_tiling_key_mode_0(self):
        """
        do_tiling_key_mode_0: in this tiling self.core_outer_num mean the last dim num for each core
        """
        outer_all_dim_num = self.tik_instance.Scalar(dtype="int64", name="outer_all_dim_num")
        outer_all_dim_num.set_as(functools.reduce(lambda x, y: x * y, self.tiling_input_shape[:7]))
        copy_num = self.copy_num
        scalar_copy_num = self.tik_instance.Scalar(dtype="int32", name="scalar_copy_num")
        scalar_copy_num.set_as(copy_num)
        with self.tik_instance.if_scope(scalar_copy_num > self.core_outer_num):
            scalar_copy_num.set_as(self.core_outer_num)
        copy_loop_ceil = self.tik_instance.Scalar(dtype="int64", name="copy_loop_ceil")
        copy_loop_floor = self.tik_instance.Scalar(dtype="int64", name="copy_loop_floor")
        copy_loop_ceil.set_as((self.core_outer_num + scalar_copy_num - 1) // scalar_copy_num)
        copy_loop_floor.set_as(self.core_outer_num // scalar_copy_num)
        copy_tail = self.core_outer_num % scalar_copy_num
        process_num_ub = self.tik_instance.Tensor("int32", (self.block_num,),
                                                  name="process_num_ub",
                                                  scope=tik.scope_ubuf)
        process_num_ub[0].set_as(scalar_copy_num)
        process_num_ub[1].set_as(copy_tail)

        def _run_one_dim(input_outer_idx, input_ub_list):
            """
            run one dim
            :param
            input_outer_idx: the input index of outer data
            :param
            input_ub_list: the list of input ub
            :return:
            None
            """
            data_ub_ping, data_ub_pang, _ = input_ub_list
            output_outer_offset = self.tik_instance.Scalar(dtype="int64", name="output_outer_offset")
            input_gm_offset = input_outer_idx * self.tiling_input_shape[-1]
            output_outer_offset.set_as(self.get_output_outer_idx(input_gm_offset) + self.tiling_pading_value[-1][0])
            # When scalar_copy_num % self.block_num != 0
            # it means that the first core only needs to move less than one block.
            # Refer to the cron_schedule args function, what happens when self.tiling_input_dim_cut axis == 0
            with self.tik_instance.if_scope(self.core_outer_start < 0):
                one_block_ub = self.tik_instance.Tensor(self.x_dtype, (self.block_num,),
                                                    name="process_num_ub",
                                                    scope=tik.scope_ubuf)
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    self.tik_instance.data_move(one_block_ub, self.input_gm[input_gm_offset], 0, 1, 1, 0, 0)
                    self.tik_instance.data_move_pad(self.output_gm[output_outer_offset],
                                                    one_block_ub,
                                                    nburst=1,
                                                    burst=scalar_copy_num * self.x_bytes_size,
                                                    dst_gap=0,
                                                    src_gap=0,
                                                    right_padding=0,
                                                    left_padding=0,
                                                    padding_value=None)
                else:
                    self.tik_instance.data_move(one_block_ub, self.input_gm[input_gm_offset], 0, 1, 1, 0, 0)
                    with self.tik_instance.for_range(self.core_outer_num - 1, self.block_num) as idx:
                        one_block_ub[idx].set_as(self.pad_scalar)
                    self.tik_instance.data_move(self.output_gm[output_outer_offset], one_block_ub, 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, copy_loop_ceil // 2) as copy_idx:
                    ping_idx = copy_idx * 2
                    idx_scalar = self.tik_instance.Scalar(dtype="int32", name="idx_scalar")
                    idx_scalar.set_as(process_num_ub[ping_idx // copy_loop_floor])
                    self.data_move(
                        [self.input_gm, input_gm_offset + ping_idx * scalar_copy_num + self.core_outer_start],
                        [self.output_gm, output_outer_offset + ping_idx * scalar_copy_num + self.core_outer_start],
                        idx_scalar, data_ub_ping)
                    pang_idx = copy_idx * 2 + 1
                    idx_scalar1 = self.tik_instance.Scalar(dtype="int32", name="idx_scalar1")
                    idx_scalar1.set_as(process_num_ub[pang_idx // copy_loop_floor])
                    self.data_move(
                        [self.input_gm, input_gm_offset + pang_idx * scalar_copy_num + self.core_outer_start],
                        [self.output_gm, output_outer_offset + pang_idx * scalar_copy_num + self.core_outer_start],
                        idx_scalar1, data_ub_pang)
                with self.tik_instance.if_scope(copy_loop_ceil % 2 != 0):
                    ping_idx = copy_loop_ceil - 1
                    idx_scalar2 = self.tik_instance.Scalar(dtype="int32", name="idx_scalar2")
                    idx_scalar2.set_as(process_num_ub[ping_idx // copy_loop_floor])
                    self.data_move(
                        [self.input_gm, input_gm_offset + ping_idx * scalar_copy_num + self.core_outer_start],
                        [self.output_gm, output_outer_offset + ping_idx * scalar_copy_num + self.core_outer_start],
                        idx_scalar2, data_ub_ping)

        ping_data_ub_ping = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="ping_data_ub_ping",
                                                     scope=tik.scope_ubuf)
        ping_data_ub_pang = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="ping_data_ub_pang",
                                                     scope=tik.scope_ubuf)
        ping_data_ub_tail = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="ping_data_ub_tail",
                                                     scope=tik.scope_ubuf)

        pang_data_ub_ping = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="pang_data_ub_ping",
                                                     scope=tik.scope_ubuf)
        pang_data_ub_pang = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="pang_data_ub_pang",
                                                     scope=tik.scope_ubuf)
        pang_data_ub_tail = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="pang_data_ub_tail",
                                                     scope=tik.scope_ubuf)

        ping_ub_list = [ping_data_ub_ping, ping_data_ub_pang, ping_data_ub_tail]
        pang_ub_list = [pang_data_ub_ping, pang_data_ub_pang, pang_data_ub_tail]

        with self.tik_instance.for_range(0, outer_all_dim_num // 2) as _outer_num_idx:
            _outer_idx = _outer_num_idx * 2
            _run_one_dim(_outer_idx, ping_ub_list)
            _outer_idx = _outer_num_idx * 2 + 1
            _run_one_dim(_outer_idx, pang_ub_list)
        with self.tik_instance.if_scope(outer_all_dim_num % 2 != 0):
            _outer_idx = outer_all_dim_num - 1
            _run_one_dim(_outer_idx, ping_ub_list)

    def do_tiling_key_mode_1(self):
        """
        do_tiling_key_mode_1  when tiling key = 1
        """
        copy_num = self.copy_num
        scalar_copy_num = self.tik_instance.Scalar(dtype="int32", name="scalar_copy_num")
        scalar_copy_num.set_as(copy_num)
        with self.tik_instance.if_scope(scalar_copy_num > self.tiling_input_shape[-1]):
            scalar_copy_num.set_as(self.tiling_input_shape[-1])

        block_copy_num = self.tik_instance.Scalar(dtype="int32", name="block_copy_num", init_value=self.block_num)
        with self.tik_instance.if_scope(scalar_copy_num < self.block_num):
            block_copy_num.set_as(self.tiling_input_shape[-1])
        copy_tail = self.tik_instance.Scalar(dtype="int32", name="copy_tail")
        copy_tail.set_as(self.tiling_input_shape[-1] % self.block_num)
        tail_copy_offset = self.tik_instance.Scalar(dtype="int64", name="tail_copy_offset")
        tail_copy_offset.set_as(copy_tail)
        with self.tik_instance.if_scope(copy_tail == 0):
            tail_copy_offset.set_as(self.block_num)

        copy_new_num = self.tiling_input_shape[-1] - tail_copy_offset
        with self.tik_instance.if_scope(scalar_copy_num > copy_new_num):
            scalar_copy_num.set_as(copy_new_num)
        copy_loop_ceil = self.tik_instance.Scalar(dtype="int64", name="copy_loop_ceil", init_value=0)
        copy_loop_floor = self.tik_instance.Scalar(dtype="int64", name="copy_loop_floor", init_value=0)
        with self.tik_instance.if_scope(scalar_copy_num != 0):
            copy_loop_ceil.set_as((copy_new_num + scalar_copy_num - 1) // scalar_copy_num)
            copy_loop_floor.set_as(copy_new_num // scalar_copy_num)
            copy_tail.set_as(copy_new_num % scalar_copy_num)
        process_num_ub = self.tik_instance.Tensor("int32", (self.block_num,),
                                                  name="process_num_ub",
                                                  scope=tik.scope_ubuf)
        process_num_ub[0].set_as(scalar_copy_num)
        process_num_ub[1].set_as(copy_tail)
        process_num_ub[2].set_as(copy_new_num - scalar_copy_num * (copy_loop_ceil - 1))

        def _run_one_dim(input_outer_idx, input_ub_list):
            """
            run one dim
            :param
            input_outer_idx: the input index of outer data
            :param
            input_ub_list: the list of input ub
            :return:
            None
            """
            ub_one_block, data_ub_ping, data_ub_pang, _ = input_ub_list
            output_outer_offset = self.tik_instance.Scalar(dtype="int64", name="output_outer_offset")
            input_gm_offset = input_outer_idx * self.tiling_input_shape[-1]
            output_outer_offset.set_as(self.get_output_outer_idx(input_gm_offset) + self.tiling_pading_value[-1][0])
            # copy one block first
            self.data_move_with_mask_less_block([self.input_gm, input_gm_offset], [self.output_gm, output_outer_offset],
                                                block_copy_num, data_ub_pang, ub_one_block)

            with self.tik_instance.for_range(0, copy_loop_ceil // 2) as copy_idx:
                ping_idx = copy_idx * 2
                idx_scalar = self.tik_instance.Scalar(dtype="int32", name="idx_scalar")
                idx_scalar.set_as(process_num_ub[ping_idx // copy_loop_floor])
                self.data_move([self.input_gm, input_gm_offset + tail_copy_offset + ping_idx * scalar_copy_num],
                               [self.output_gm, output_outer_offset + tail_copy_offset + ping_idx * scalar_copy_num],
                               idx_scalar, data_ub_ping)
                pang_idx = copy_idx * 2 + 1
                idx_scalar1 = self.tik_instance.Scalar(dtype="int32", name="idx_scalar1")
                idx_scalar1.set_as(process_num_ub[pang_idx // copy_loop_floor])
                self.data_move([self.input_gm, input_gm_offset + tail_copy_offset + pang_idx * scalar_copy_num],
                               [self.output_gm, output_outer_offset + tail_copy_offset + pang_idx * scalar_copy_num],
                               idx_scalar1, data_ub_pang)
            with self.tik_instance.if_scope(copy_loop_ceil % 2 != 0):
                ping_idx = copy_loop_ceil - 1
                idx_scalar2 = self.tik_instance.Scalar(dtype="int32", name="idx_scalar2")
                idx_scalar2.set_as(process_num_ub[ping_idx // copy_loop_floor])
                self.data_move([self.input_gm, input_gm_offset + tail_copy_offset + ping_idx * scalar_copy_num],
                               [self.output_gm, output_outer_offset + tail_copy_offset + ping_idx * scalar_copy_num],
                               idx_scalar2, data_ub_ping)

        ping_data_ub_one_block = self.tik_instance.Tensor(self.x_dtype,
                                                          (self.block_num * self.block_num + self.block_num,),
                                                          name="ping_data_ub_one_block",
                                                          scope=tik.scope_ubuf)
        ping_data_ub_ping = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="ping_data_ub_ping",
                                                     scope=tik.scope_ubuf)
        ping_data_ub_pang = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="ping_data_ub_pang",
                                                     scope=tik.scope_ubuf)
        ping_data_ub_tail = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="ping_data_ub_tail",
                                                     scope=tik.scope_ubuf)

        pang_data_ub_one_block = self.tik_instance.Tensor(self.x_dtype,
                                                          (self.block_num * self.block_num + self.block_num,),
                                                          name="pang_data_ub_one_block",
                                                          scope=tik.scope_ubuf)
        pang_data_ub_ping = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="pang_data_ub_ping",
                                                     scope=tik.scope_ubuf)
        pang_data_ub_pang = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="pang_data_ub_pang",
                                                     scope=tik.scope_ubuf)
        pang_data_ub_tail = self.tik_instance.Tensor(self.x_dtype, (copy_num,),
                                                     name="pang_data_ub_tail",
                                                     scope=tik.scope_ubuf)

        if self.x_dtype in self.four_bytes_dtype:
            self.tik_instance.vector_dup(self.block_num * 4, ping_data_ub_one_block[8:], self.pad_scalar, 1, 1, 8)
            self.tik_instance.vector_dup(self.block_num * 4, pang_data_ub_one_block[8:], self.pad_scalar, 1, 1, 8)
        else:
            self.tik_instance.vector_dup(self.block_num * 8, ping_data_ub_one_block[16:], self.pad_scalar, 2, 1, 8)
            self.tik_instance.vector_dup(self.block_num * 8, pang_data_ub_one_block[16:], self.pad_scalar, 2, 1, 8)

        ping_ub_list = [ping_data_ub_one_block, ping_data_ub_ping, ping_data_ub_pang, ping_data_ub_tail]
        pang_ub_list = [pang_data_ub_one_block, pang_data_ub_ping, pang_data_ub_pang, pang_data_ub_tail]

        with self.tik_instance.for_range(0, self.core_outer_num // 2) as _outer_num_idx:
            _outer_idx = _outer_num_idx * 2 + self.core_outer_start
            _run_one_dim(_outer_idx, ping_ub_list)
            _outer_idx = _outer_num_idx * 2 + 1 + self.core_outer_start
            _run_one_dim(_outer_idx, pang_ub_list)
        with self.tik_instance.if_scope(self.core_outer_num % 2 != 0):
            _outer_idx = self.core_outer_num - 1 + self.core_outer_start
            _run_one_dim(_outer_idx, ping_ub_list)

    def cal_max_output_ub(self, max_line_in_ub):
        """
        calculate max output ub size for mode 2 and 3
        :param
        max_line_in_ub: max unit length for each line
        :return:
        max_output_size: max output ub size to calculate
        """
        tail_reserve_ub = 16 * 16
        output_ub_cut_num = 4
        max_output_size = 0
        if self.x_dtype in self.one_bytes_dtype:
            output_ub_cut_num = 5
            ori_max_output_size = (self.ub_size_bytes - Constant.TWO_BYTES * tail_reserve_ub * \
                Constant.FOUR_PIECES) // (output_ub_cut_num * max_line_in_ub * Constant.TWO_BYTES)
            max_output_size = (ori_max_output_size // Constant.ONE_BYTES_DTYPE_BLOCK_NUM) * \
                Constant.ONE_BYTES_DTYPE_BLOCK_NUM
        elif (self.version_info == "Ascend910" or self.version_info == "Ascend310") \
            and self.x_dtype in self.four_bytes_dtype:
            ori_max_output_size = (self.ub_size_bytes - self.x_bytes_size * tail_reserve_ub * Constant.TWO_PIECES - \
                                Constant.RESERVE_FOR_FP32_VNCHWCONV_MODE_2) // \
                                (output_ub_cut_num * max_line_in_ub * self.x_bytes_size)
            max_output_size = (ori_max_output_size // max_line_in_ub) * max_line_in_ub
        else:
            ori_max_output_size = (self.ub_size_bytes - self.x_bytes_size * tail_reserve_ub * Constant.TWO_PIECES) // \
                                (output_ub_cut_num * max_line_in_ub * self.x_bytes_size)
            max_output_size = (ori_max_output_size // max_line_in_ub) * max_line_in_ub
        return max_output_size

    def do_tiling_key_mode_2(self):
        """
        do_tiling_key_mode_2 when tiling key = 2
        """
        max_line_in_ub = 16
        max_output_size = self.cal_max_output_ub(max_line_in_ub)
        second_dim_input_num = self.tiling_input_shape[-2]
        third_dim_input_num = self.tiling_input_shape[-1]
        third_dim_output_num = self.tiling_output_shape[-1]

        first_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="first_dim_cut_num")
        second_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_cut_num")

        second_dim_cut_num.set_as(max_output_size // third_dim_output_num)
        with self.tik_instance.if_scope(second_dim_cut_num > second_dim_input_num):
            second_dim_cut_num.set_as(second_dim_input_num)

        first_dim_cut_num.set_as(max_line_in_ub * second_dim_cut_num)
        # cut inner first dim and second dim info
        second_dim_total_loop_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_total_loop_num")
        second_dim_total_loop_tail = self.tik_instance.Scalar(dtype="int64", name="second_dim_total_loop_tail")
        second_dim_total_loop_num.set_as(second_dim_input_num // second_dim_cut_num)
        second_dim_total_loop_tail.set_as(second_dim_input_num % second_dim_cut_num)

        second_dim_outer_cut_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_cut_num")
        second_dim_outer_cut_num.set_as(max_line_in_ub)
        with self.tik_instance.if_scope(second_dim_total_loop_num < max_line_in_ub):
            second_dim_outer_cut_num.set_as(second_dim_total_loop_num)

        second_dim_outer_loop_num_ceil = \
            (second_dim_total_loop_num + second_dim_outer_cut_num - 1) // second_dim_outer_cut_num
        second_dim_outer_loop_num_floor = second_dim_total_loop_num // second_dim_outer_cut_num

        second_dim_outer_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                               name="second_dim_outer_sigment_ub",
                                                               scope=tik.scope_ubuf)
        second_dim_outer_sigment_ub[0].set_as(second_dim_outer_cut_num)
        second_dim_outer_sigment_ub[1].set_as(second_dim_total_loop_num % second_dim_outer_cut_num)

        second_dim_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                         name="second_dim_sigment_ub",
                                                         scope=tik.scope_ubuf)
        second_dim_sigment_ub[0].set_as(second_dim_cut_num)
        second_dim_sigment_ub[1].set_as(second_dim_input_num % second_dim_cut_num)

        loop_align_tail = self.tik_instance.Scalar(dtype="int64", name="loop_align_tail")
        tail_align_tail = self.tik_instance.Scalar(dtype="int64", name="tail_align_tail")
        one_core_flag = self.tik_instance.Scalar(dtype="int64", name="one_core_flag", init_value=0)
        loop_align_tail.set_as((second_dim_cut_num * third_dim_output_num) % max_line_in_ub)
        tail_align_tail.set_as((second_dim_total_loop_tail * third_dim_output_num) % max_line_in_ub)
        with self.tik_instance.if_scope(self.tiling_output_shape[-1] * self.tiling_input_shape[-2] <= max_line_in_ub):
            loop_align_tail.set_as(0)
            tail_align_tail.set_as(0)
            one_core_flag.set_as(self.block_num - 1)

        vnchw_src_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride0", init_value=1)
        vnchw_dst_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride0", init_value=16)
        vnchw_src_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride1", init_value=16)
        vnchw_dst_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride1", init_value=1)
        if self.x_dtype in self.four_bytes_dtype:
            vnchw_src_stride1.set_as(32)
            vnchw_dst_stride1.set_as(2)
        vnchw_repeat0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat0", init_value=1)
        vnchw_repeat1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat1", init_value=1)
        input_algin_output_num = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat1", init_value=1)
        input_algin_output_num.set_as((second_dim_cut_num * third_dim_output_num - self.tiling_pading_value[-1][1] + \
                                       self.block_num - 1) // self.block_num * self.block_num - second_dim_cut_num * \
                                       third_dim_output_num - self.tiling_pading_value[-1][0])
        with self.tik_instance.if_scope(input_algin_output_num < 0):
            input_algin_output_num.set_as(0)
        with self.tik_instance.if_scope(input_algin_output_num > third_dim_input_num):
            input_algin_output_num.set_as(third_dim_input_num)
        vnchw_repeat0.set_as(((second_dim_cut_num * third_dim_input_num) + \
                               input_algin_output_num + self.block_num - 1) // self.block_num)
        vnchw_repeat1.set_as(((second_dim_cut_num * third_dim_output_num) + max_line_in_ub - 1) // max_line_in_ub)
        with self.tik_instance.if_scope(vnchw_repeat0 == 1):
            vnchw_src_stride0.set_as(0)
            vnchw_dst_stride0.set_as(0)
        with self.tik_instance.if_scope(vnchw_repeat1 == 1):
            vnchw_src_stride1.set_as(0)
            vnchw_dst_stride1.set_as(0)

        def run_outer_by_outer(second_dim_start, do_inner_num, do_outer_num, align_tail, disable_sync_mte3=False):
            """
            run_outer_by_outer
            :param
            second_dim_start:the second dim start of input
            :param
            do_inner_num: the number which do inner
            :param
            do_outer_num: the number which do outer
            :param
            align_tail: the tail of align
            :param
            disable_sync_mte3: disable_sync_mte3
            :return:
            None
            """

            def _run_one_outer(_outer_num_idx, ub_list):
                """
                _run_one_outer
                :param
                _outer_num_idx:
                :param
                ub_list:
                :return:
                none
                """
                origin_data_ub, vnchw_data_ub, origin_output_tail_data_ub = ub_list
                input_outer_idx = _outer_num_idx + self.core_outer_start
                input_gm_offset = input_outer_idx * self.input_offset[-2]
                output_outer_offset = self.tik_instance.Scalar(dtype="int64", name="output_outer_offset")
                output_outer_offset.set_as(self.get_output_outer_idx(input_gm_offset, 6))
                pad_left = self.tiling_pading_value[-1][0]
                pad_right = self.tiling_pading_value[-1][1]
                # step1. copy 16 dims in origin_data_ub
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                            burst_len = do_inner_num * third_dim_input_num * self.x_bytes_size
                            src_offset = (second_dim_start + _copy_idx * do_inner_num) * third_dim_input_num
                            self.tik_instance.data_move_pad(origin_data_ub[_copy_idx * max_output_size],
                                                        self.input_gm[input_gm_offset + src_offset],
                                                        1, burst_len, 0, 0, 0, 0)
                else:
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        with self.tik_instance.for_range(0, do_outer_num - 1) as _copy_idx:
                            burst_len = (do_inner_num * third_dim_input_num + \
                                        input_algin_output_num + self.block_num - 1) // self.block_num
                            src_offset = (second_dim_start + _copy_idx * do_inner_num) * third_dim_input_num
                            self.tik_instance.data_move(origin_data_ub[_copy_idx * max_output_size],
                                                        self.input_gm[input_gm_offset + src_offset],
                                                        0, 1, burst_len, 0, 0)
                        src_offset_last = (second_dim_start + (do_outer_num - 1) * do_inner_num) * third_dim_input_num
                        burst_len_last = (do_inner_num * third_dim_input_num + self.block_num - 1) // self.block_num
                        self.tik_instance.data_move(origin_data_ub[(do_outer_num - 1) * max_output_size],
                                                    self.input_gm[input_gm_offset + src_offset_last],
                                                    0, 1, burst_len_last, 0, 0)
                        
                # step2. vnchw 16 dims origin_data_ub to vnchw_data_ub
                origin_data_ub_list = [origin_data_ub[i * max_output_size] for i in range(0, Constant.TRANS_MIN_BLKS)]
                vnchw_data_ub_list = [vnchw_data_ub[i * self.block_num] for i in range(0, Constant.TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False, vnchw_data_ub_list, origin_data_ub_list, vnchw_repeat0,
                                            vnchw_dst_stride0, vnchw_src_stride0)
                # step2.1 Reuse origin_data_ub as vnchw_output_data_ub，need to fill origin_data_ub with pad_scalar
                self.tik_instance.vector_dup(self.block_num * 8, origin_data_ub, self.pad_scalar,
                                             max_line_in_ub * max_output_size // self.block_num // 8, 1, 8)
                
                # step3. rearange vnchw_data_ub to vnchw_output_data_ub
                # step3.0 copy input data to vnchw_output_data_ub with datamove
                burst_num = do_inner_num
                burst_len = third_dim_input_num * (max_line_in_ub // self.block_num)
                src_offset = 0
                dst_offset = pad_left * self.block_num * (max_line_in_ub // self.block_num)
                src_stride = 0
                dst_stride = (pad_left + pad_right) * (max_line_in_ub // self.block_num)
                self.tik_instance.data_move(origin_data_ub[dst_offset], vnchw_data_ub[src_offset], 0, burst_num,
                                            burst_len, src_stride, dst_stride)
                if not tbe_platform.api_check_support("tik.data_move_pad"):
                    with self.tik_instance.if_scope(tik.all(input_algin_output_num > 0, do_outer_num > 1)):
                        dst_offset = third_dim_output_num * max_line_in_ub * do_inner_num + pad_left * max_line_in_ub
                        src_offset = third_dim_input_num * max_line_in_ub * do_inner_num
                        self.tik_instance.data_move(origin_data_ub[dst_offset], vnchw_data_ub[src_offset], 0, 1,
                                                    input_algin_output_num * (max_line_in_ub // self.block_num), 0, 0)
                # step4. vnchw vnchw_output_data_ub to 16 dims origin_output_data_ub
                origin_output_data_ub_list = []
                if self.x_dtype in self.four_bytes_dtype:
                    for i in range(0, Constant.TRANS_MIN_BLKS):
                        if i % 2 == 0:
                            tmp = vnchw_data_ub[i // 2 * max_output_size]
                        else:
                            tmp = vnchw_data_ub[i // 2 * max_output_size + self.block_num]
                        origin_output_data_ub_list.append(tmp)
                else:
                    origin_output_data_ub_list = [
                        vnchw_data_ub[i * max_output_size]
                        for i in range(0, Constant.TRANS_MIN_BLKS)
                        ]
                vnchw_output_data_ub_list = \
                    [origin_data_ub[i * 16] for i in range(0, Constant.TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False, origin_output_data_ub_list, vnchw_output_data_ub_list,
                                            vnchw_repeat1 , vnchw_dst_stride1, vnchw_src_stride1)

                if self.x_dtype in self.four_bytes_dtype:
                    ub_offset = max_output_size * Constant.TRANS_MIN_BLKS // 2
                    origin_output_data_ub_list = []
                    for i in range(0, Constant.TRANS_MIN_BLKS):
                        if i % 2 == 0:
                            tmp = vnchw_data_ub[ub_offset + i // 2 * max_output_size]
                        else:
                            tmp = vnchw_data_ub[ub_offset + i // 2 * max_output_size + self.block_num]
                        origin_output_data_ub_list.append(tmp)
                    vnchw_output_data_ub_list = \
                        [origin_data_ub[self.block_num + i * 16] for i in range(0, Constant.TRANS_MIN_BLKS)]
                    self.tik_instance.vnchwconv(False, False, origin_output_data_ub_list, vnchw_output_data_ub_list,
                                                vnchw_repeat1 , vnchw_dst_stride1, vnchw_src_stride1)

                # step5. copy 16 dims to output
                # step5.1 copy do_outer_num - 1 lines to output use ceil_div block
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                            dst_offset = \
                                output_outer_offset + \
                                (self.tiling_pading_value[-2][0] + second_dim_start + _copy_idx * do_inner_num) \
                                * self.output_offset[-1]
                            tail_num = (do_inner_num * third_dim_output_num - pad_right) * self.x_bytes_size
                            self.tik_instance.data_move_pad(self.output_gm[dst_offset],
                                                            vnchw_data_ub[_copy_idx * max_output_size],
                                                            nburst=1,
                                                            burst=tail_num,
                                                            dst_gap=0,
                                                            src_gap=0,
                                                            right_padding=0,
                                                            left_padding=0,
                                                            padding_value=None)
                else:
                    with self.tik_instance.if_scope(do_inner_num * third_dim_output_num % self.block_num != 0):
                        with self.tik_instance.new_stmt_scope(disable_sync=True):
                            with self.tik_instance.for_range(0, do_outer_num - 1) as _copy_idx:
                                dst_offset = \
                                    output_outer_offset + \
                                    (self.tiling_pading_value[-2][0] + second_dim_start + _copy_idx * do_inner_num) \
                                    * self.output_offset[-1]  
                                burst_len = (do_inner_num * third_dim_output_num - \
                                             pad_right + self.block_num - 1) // self.block_num
                                self.tik_instance.data_move(self.output_gm[dst_offset],
                                                            vnchw_data_ub[_copy_idx * max_output_size], 0, 1,
                                                            burst_len, 0, 0)
                        # step5.1 copy the last do_outer_num lines to output use floor_div block
                        dst_offset = \
                            output_outer_offset + \
                            (self.tiling_pading_value[-2][0] + \
                            second_dim_start + (do_outer_num - 1) * do_inner_num) \
                            * self.output_offset[-1]
                        burst_len = (do_inner_num * third_dim_output_num - self.tiling_pading_value[-1][1] +\
                                     one_core_flag) // self.block_num
                        with self.tik_instance.if_scope(burst_len <= 0):
                            burst_len = (do_inner_num * third_dim_output_num + one_core_flag) // self.block_num
                        loop_num = burst_len * self.block_num - do_inner_num * \
                            third_dim_output_num - self.tiling_pading_value[-1][0]
                        with self.tik_instance.if_scope(tik.all(loop_num > 0, align_tail == 0, do_inner_num == 1,
                                                                self.tiling_pading_value[-2][0] + \
                                                                self.tiling_pading_value[-2][1] == 0)):
                            src_offset_gm = input_gm_offset + (second_dim_start + do_outer_num * do_inner_num) *\
                                            third_dim_input_num
                            src_offset_ub = do_inner_num * third_dim_output_num
                            self.tik_instance.data_move(origin_output_tail_data_ub, 
                                                        self.input_gm[src_offset_gm], 0, 1, 1, 0, 0)
                            with self.tik_instance.for_range(0, loop_num - 1) as loop_idx:
                                vnchw_data_ub[src_offset_ub + self.tiling_pading_value[-1][0] +\
                                    loop_idx].set_as(origin_output_tail_data_ub[loop_idx])
                        self.tik_instance.data_move(self.output_gm[dst_offset],
                                                    vnchw_data_ub[(do_outer_num - 1) * max_output_size],
                                                    0, 1, burst_len, 0, 0)
                        # step6. process tail for the last line
                        with self.tik_instance.if_scope(align_tail != 0):
                            if self.x_dtype in self.four_bytes_dtype:
                                with self.tik_instance.if_scope(do_inner_num * third_dim_output_num >= 16):
                                    burst_len = 2
                                    origin_output_data_ub_list = \
                                        [origin_output_tail_data_ub[i * self.block_num]
                                        for i in range(0, Constant.TRANS_MIN_BLKS)]
                                    vnchw_output_data_ub_list = \
                                        [origin_data_ub[i * 16 + (do_inner_num * third_dim_output_num - 16) * 16]
                                        for i in range(0, Constant.TRANS_MIN_BLKS)]
                                    self.tik_instance.vnchwconv(False, False, origin_output_data_ub_list,
                                                            vnchw_output_data_ub_list, 1, 0, 0)
                                    half_origin_output_tail_data_ub_offset = 8 * 16
                                    origin_output_data_ub_list = \
                                        [origin_output_tail_data_ub[half_origin_output_tail_data_ub_offset + \
                                                            i * self.block_num]
                                        for i in range(0, Constant.TRANS_MIN_BLKS)]
                                    vnchw_output_data_ub_list = \
                                        [origin_data_ub[i * 16 + self.block_num + \
                                                            (do_inner_num * third_dim_output_num - 16) * 16]
                                        for i in range(0, Constant.TRANS_MIN_BLKS)]
                                    self.tik_instance.vnchwconv(False, False, origin_output_data_ub_list,
                                                            vnchw_output_data_ub_list, 1, 0, 0)
                                    
                                    dst_offset = \
                                        output_outer_offset \
                                        + (self.tiling_pading_value[-2][0] + second_dim_start \
                                        + do_outer_num * do_inner_num) \
                                        * self.output_offset[-1] - max_line_in_ub
                                    self.tik_instance.data_move(self.output_gm[dst_offset],
                                                                origin_output_tail_data_ub[(do_outer_num - 1) * 16],
                                                                0, 1, burst_len, 0, 0)
                                    
                                with self.tik_instance.elif_scope(do_inner_num * third_dim_output_num < 16):
                                    burst_len = 2
                                    origin_output_data_ub_list = \
                                        [origin_output_tail_data_ub[i * self.block_num]
                                        for i in range(0, Constant.TRANS_MIN_BLKS)]
                                    vnchw_output_data_ub_list = \
                                        [origin_data_ub[i * 16]
                                        for i in range(0, Constant.TRANS_MIN_BLKS)]
                                    self.tik_instance.vnchwconv(False, False, origin_output_data_ub_list,
                                                            vnchw_output_data_ub_list, 1, 0, 0)
                                    half_origin_output_tail_data_ub_offset = 8 * 16
                                    origin_output_data_ub_list = \
                                        [origin_output_tail_data_ub[half_origin_output_tail_data_ub_offset + \
                                            i * self.block_num]
                                        for i in range(0, Constant.TRANS_MIN_BLKS)]
                                    vnchw_output_data_ub_list = \
                                        [origin_data_ub[i * 16 + self.block_num]
                                        for i in range(0, Constant.TRANS_MIN_BLKS)]
                                    self.tik_instance.vnchwconv(False, False, origin_output_data_ub_list,
                                                            vnchw_output_data_ub_list, 1, 0, 0)
                                    
                                    dst_offset = \
                                        output_outer_offset \
                                        + (self.tiling_pading_value[-2][0] + second_dim_start + \
                                            do_outer_num * do_inner_num) \
                                        * self.output_offset[-1] \
                                        - do_inner_num * third_dim_output_num
                                    self.tik_instance.data_move(self.output_gm[dst_offset],
                                                                origin_output_tail_data_ub,
                                                                0, 1, burst_len, 0, 0)
                            else:
                                burst_len = 1
                                origin_output_data_ub_list = \
                                    [origin_output_tail_data_ub[i * self.block_num]
                                    for i in range(0, Constant.TRANS_MIN_BLKS)]
                                vnchw_output_data_ub_list = \
                                    [origin_data_ub[i * 16 + (do_inner_num * third_dim_output_num - 16) * 16]
                                    for i in range(0, Constant.TRANS_MIN_BLKS)]
                                self.tik_instance.vnchwconv(False, False, origin_output_data_ub_list,
                                                            vnchw_output_data_ub_list, 1, 0, 0)
                                
                                dst_offset = \
                                    output_outer_offset \
                                    + (self.tiling_pading_value[-2][0] \
                                    + second_dim_start + do_outer_num * do_inner_num) \
                                    * self.output_offset[-1] \
                                    - max_line_in_ub
                                self.tik_instance.data_move(self.output_gm[dst_offset],
                                                            origin_output_tail_data_ub[(do_outer_num - 1) * 16],
                                                            0, 1, burst_len, 0, 0)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.new_stmt_scope(disable_sync=True):
                            with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                                burst_len = (do_inner_num * third_dim_output_num + self.block_num - 1) // self.block_num
                                dst_offset = \
                                    output_outer_offset + \
                                    (self.tiling_pading_value[-2][0] + second_dim_start + _copy_idx * do_inner_num) \
                                    * self.output_offset[-1]
                                self.tik_instance.data_move(self.output_gm[dst_offset],
                                                            vnchw_data_ub[_copy_idx * max_output_size], 0, 1,
                                                            burst_len, 0, 0)

            origin_data_ub_ping = self.tik_instance.Tensor(self.x_dtype, (max_line_in_ub, max_output_size,),
                                                           name="origin_data_ub_ping",
                                                           scope=tik.scope_ubuf)
            vnchw_data_ub_ping = self.tik_instance.Tensor(self.x_dtype, (max_line_in_ub, max_output_size,),
                                                          name="vnchw_data_ub_ping",
                                                          scope=tik.scope_ubuf)
            origin_output_tail_data_ub_ping = self.tik_instance.Tensor(self.x_dtype, (16 * 16,),
                                                                       name="origin_output_tail_data_ub_ping",
                                                                       scope=tik.scope_ubuf)

            origin_data_ub_pang = self.tik_instance.Tensor(self.x_dtype, (max_line_in_ub , max_output_size),
                                                           name="origin_data_ub_pang",
                                                           scope=tik.scope_ubuf)
            vnchw_data_ub_pang = self.tik_instance.Tensor(self.x_dtype, (max_line_in_ub , max_output_size,),
                                                          name="vnchw_data_ub_ping",
                                                          scope=tik.scope_ubuf)
            origin_output_tail_data_ub_pang = self.tik_instance.Tensor(self.x_dtype, (16 * 16,),
                                                                       name="origin_output_tail_data_ub_ping",
                                                                       scope=tik.scope_ubuf)

            ping_ub_list = [origin_data_ub_ping, vnchw_data_ub_ping, origin_output_tail_data_ub_ping]
            pang_ub_list = [origin_data_ub_pang, vnchw_data_ub_pang, origin_output_tail_data_ub_pang]
            with self.tik_instance.for_range(0, self.core_outer_num // 2) as _outer_idx:
                _run_one_outer(_outer_idx * 2, ping_ub_list)
                _run_one_outer(_outer_idx * 2 + 1, pang_ub_list)
            with self.tik_instance.if_scope(self.core_outer_num % 2 == 1):
                _run_one_outer(self.core_outer_num - 1, ping_ub_list)

        with self.tik_instance.for_range(0, second_dim_outer_loop_num_ceil) as second_dim_outer_idx:
            second_dim_outer_start = second_dim_outer_idx * second_dim_outer_cut_num * second_dim_cut_num
            second_dim_outer_process_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_process_num")
            second_dim_outer_process_num.set_as(second_dim_outer_sigment_ub[second_dim_outer_idx //
                                                                            second_dim_outer_loop_num_floor])
            run_outer_by_outer(second_dim_outer_start, second_dim_cut_num, second_dim_outer_process_num,
                               loop_align_tail)

        with self.tik_instance.if_scope(second_dim_total_loop_tail != 0):
            second_dim_outer_tail_start = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_tail_start")
            second_dim_outer_tail_start.set_as((second_dim_input_num // second_dim_cut_num) * second_dim_cut_num)
            with self.tik_instance.if_scope(second_dim_total_loop_tail * third_dim_output_num < self.block_num):
                new_tail_num = (self.block_num + third_dim_output_num - 1) // third_dim_output_num
                second_dim_outer_tail_start.set_as(second_dim_outer_tail_start - new_tail_num +
                                                   second_dim_total_loop_tail)
                second_dim_total_loop_tail.set_as(new_tail_num)

            run_outer_by_outer(second_dim_outer_tail_start, second_dim_total_loop_tail, 1, tail_align_tail) 

    def do_padv3_int8_high_profromance(self):
        """
        do_padv3_int8_high_profromance when tiling key = 8
        """
        max_line_in_ub = 16
        max_output_size = self.cal_max_output_ub(max_line_in_ub)
        penultimate_dim_input_num = self.tiling_input_shape[-2]
        last_dim_input_num = self.tiling_input_shape[-1]
        last_dim_output_num = self.tiling_output_shape[-1]

        first_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="first_dim_cut_num")
        penultimate_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="penultimate_dim_cut_num")

        penultimate_dim_cut_num.set_as(max_output_size // last_dim_output_num)
        with self.tik_instance.if_scope(penultimate_dim_cut_num > penultimate_dim_input_num):
            penultimate_dim_cut_num.set_as(penultimate_dim_input_num)

        first_dim_cut_num.set_as(max_line_in_ub * penultimate_dim_cut_num)

        # cut inner first dim and second dim info
        second_dim_total_loop_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_total_loop_num")
        second_dim_total_loop_tail = self.tik_instance.Scalar(dtype="int64", name="second_dim_total_loop_tail")
        second_dim_total_loop_num.set_as(penultimate_dim_input_num // penultimate_dim_cut_num)
        second_dim_total_loop_tail.set_as(penultimate_dim_input_num % penultimate_dim_cut_num)

        second_dim_outer_cut_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_cut_num")
        second_dim_outer_cut_num.set_as(max_line_in_ub)
        with self.tik_instance.if_scope(second_dim_total_loop_num < max_line_in_ub): 
            second_dim_outer_cut_num.set_as(second_dim_total_loop_num)

        second_dim_outer_loop_num_ceil = \
            (second_dim_total_loop_num + second_dim_outer_cut_num - 1) // second_dim_outer_cut_num
        second_dim_outer_loop_num_floor = second_dim_total_loop_num // second_dim_outer_cut_num

        second_dim_outer_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                               name="second_dim_outer_sigment_ub",
                                                               scope=tik.scope_ubuf)
        second_dim_outer_sigment_ub[0].set_as(second_dim_outer_cut_num)
        second_dim_outer_sigment_ub[1].set_as(second_dim_total_loop_num % second_dim_outer_cut_num)

        second_dim_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                         name="second_dim_sigment_ub",
                                                         scope=tik.scope_ubuf)
        second_dim_sigment_ub[0].set_as(penultimate_dim_cut_num)
        second_dim_sigment_ub[1].set_as(second_dim_total_loop_tail) 

        loop_align_tail = self.tik_instance.Scalar(dtype="int64", name="loop_align_tail")
        tail_align_tail = self.tik_instance.Scalar(dtype="int64", name="tail_align_tail")
        one_core_flag = self.tik_instance.Scalar(dtype="int64", name="one_core_flag", init_value=0)
        loop_align_tail.set_as((penultimate_dim_cut_num * last_dim_output_num) % self.block_num)
        tail_align_tail.set_as((second_dim_total_loop_tail * last_dim_output_num) % self.block_num)
        with self.tik_instance.if_scope(self.tiling_output_shape[-1] * self.tiling_input_shape[-2] <= self.block_num):
            loop_align_tail.set_as(0)
            tail_align_tail.set_as(0)
            one_core_flag.set_as(Constant.TWO_BYTES_DTYPE_BLOCK_NUM - 1)

        vnchw_src_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride0", init_value=1)
        vnchw_dst_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride0", init_value=16)
        vnchw_src_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride1", init_value=16)
        vnchw_dst_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride1", init_value=1)
        vnchw_repeat0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat0", init_value=1)
        vnchw_repeat1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat1", init_value=1)
        vnchw_repeat0.set_as(((penultimate_dim_cut_num * last_dim_input_num) + \
                               Constant.TWO_BYTES_DTYPE_BLOCK_NUM - 1) // Constant.TWO_BYTES_DTYPE_BLOCK_NUM)
        vnchw_repeat1.set_as((((penultimate_dim_cut_num * last_dim_output_num) + self.block_num - 1) // \
                                self.block_num * self.block_num + max_line_in_ub - 1) // max_line_in_ub)
        with self.tik_instance.if_scope(vnchw_repeat0 == 1):
            vnchw_src_stride0.set_as(0)
            vnchw_dst_stride0.set_as(0)
        with self.tik_instance.if_scope(vnchw_repeat1 == 1):
            vnchw_src_stride1.set_as(0)
            vnchw_dst_stride1.set_as(0)

        def run_outer_by_outer(second_dim_start, do_inner_num, do_outer_num, align_tail, disable_sync_mte3=False):
            """
            run_outer_by_outer
            :param
            second_dim_start:the second dim start of input
            :param
            do_inner_num: the number which do inner
            :param
            do_outer_num: the number which do outer
            :param
            align_tail: the tail of align
            :param
            disable_sync_mte3: disable_sync_mte3
            :return:
            None
            """

            def _run_one_outer(_outer_num_idx, ub_list):
                """
                _run_one_outer
                :param
                _outer_num_idx:
                :param
                ub_list:
                :return:
                none
                """
                origin_data_ub, vnchw_data_ub_a, vnchw_data_ub_b, tail_data_ub = ub_list
                input_outer_idx = _outer_num_idx + self.core_outer_start
                input_gm_offset = input_outer_idx * self.input_offset[-2]
                output_outer_offset = self.tik_instance.Scalar(dtype="int64", name="output_outer_offset")
                output_outer_offset.set_as(self.get_output_outer_idx(input_gm_offset, 6))
                # step1. copy 16 dims in origin_data_ub
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                            burst_len = do_inner_num * last_dim_input_num * self.x_bytes_size
                            src_offset = (second_dim_start + _copy_idx * do_inner_num) * last_dim_input_num
                            self.tik_instance.data_move_pad(origin_data_ub[_copy_idx * max_output_size],
                                                        self.input_gm[input_gm_offset + src_offset],
                                                        1, burst_len, 0, 0, 0, 0)
                else:
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                            burst_len = ((do_inner_num * last_dim_input_num) + self.block_num - 1) // self.block_num
                            src_offset = (second_dim_start + _copy_idx * do_inner_num) * last_dim_input_num
                            self.tik_instance.data_move(origin_data_ub[_copy_idx * max_output_size],
                                                        self.input_gm[input_gm_offset + src_offset],
                                                        0, 1, burst_len, 0, 0)
                # step2. cast to float16
                self.conv_one_byte_dtype_to_fp16(origin_data_ub,
                                                vnchw_data_ub_a, 
                                                do_outer_num * max_output_size)
                
                # step3. vnchw 16 dims origin_data_ub to vnchw_data_ub
                origin_data_ub_list = \
                    [vnchw_data_ub_a[i * max_output_size] for i in range(0, Constant.TRANS_MIN_BLKS)]
                vnchw_data_ub_list = \
                    [vnchw_data_ub_b[i * max_line_in_ub] for i in range(0, Constant.TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False, vnchw_data_ub_list, origin_data_ub_list, vnchw_repeat0,
                                            vnchw_dst_stride0, vnchw_src_stride0)
                
                # step4. Reuse origin_data_ub as vnchw_output_data_ub，need to fill origin_data_ub with pad_scalar
                self.tik_instance.vector_dup(Constant.MASK_TWO_BYTES_DTYPE, vnchw_data_ub_a, self.pad_scalar_int8,
                                             max_line_in_ub * max_output_size // Constant.MASK_TWO_BYTES_DTYPE, 1, 8)
                # step5. rearange vnchw_data_ub to vnchw_output_data_ub
                # step5.0 copy input data to vnchw_output_data_ub with datamove
                pad_left = self.tiling_pading_value[-1][0]
                pad_right = self.tiling_pading_value[-1][1]
                burst_num = do_inner_num
                burst_len = last_dim_input_num 
                src_offset = 0
                dst_offset = pad_left * Constant.TWO_BYTES_DTYPE_BLOCK_NUM
                src_stride = 0
                dst_stride = (pad_left + pad_right)
                self.tik_instance.data_move(vnchw_data_ub_a[dst_offset], vnchw_data_ub_b[src_offset], 0, burst_num,
                                            burst_len, src_stride, dst_stride)
                # step6. vnchw origin_output_data_ub to 16 dims vnchw_output_data_ub 
                origin_output_data_ub_list = \
                    [vnchw_data_ub_b[i * max_output_size] for i in range(0, Constant.TRANS_MIN_BLKS)]
                vnchw_output_data_ub_list = \
                    [vnchw_data_ub_a[i * max_line_in_ub] for i in range(0, Constant.TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False, origin_output_data_ub_list, vnchw_output_data_ub_list, 
                                            vnchw_repeat1 , vnchw_dst_stride1, vnchw_src_stride1)
                
                # step7. cast to int8
                self.conv_fp16_to_one_byte_dtype(vnchw_data_ub_b, origin_data_ub, (do_outer_num * max_output_size))
                # Support data_move_pad branch
                if tbe_platform.api_check_support("tik.data_move_pad"):
                    # step8. data_move_to_gm
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                            dst_offset = \
                                output_outer_offset + \
                                (self.tiling_pading_value[-2][0] + second_dim_start + _copy_idx * do_inner_num) \
                                * last_dim_output_num
                            tail_num = (do_inner_num * last_dim_output_num) * self.x_bytes_size
                            self.tik_instance.data_move_pad(self.output_gm[dst_offset],
                                                            origin_data_ub[_copy_idx * max_output_size],
                                                            nburst=1,
                                                            burst=tail_num,
                                                            dst_gap=0,
                                                            src_gap=0,
                                                            right_padding=0,
                                                            left_padding=0,
                                                            padding_value=None)
                # not Support data_move_pad branch
                else:
                    # step8. data_move_to_gm
                    with self.tik_instance.if_scope(do_inner_num * last_dim_output_num % self.block_num != 0):
                        with self.tik_instance.new_stmt_scope(disable_sync=disable_sync_mte3):
                            with self.tik_instance.for_range(0, do_outer_num - 1) as _copy_idx:
                                dst_offset = \
                                    output_outer_offset + \
                                    (self.tiling_pading_value[-2][0] + second_dim_start + _copy_idx * do_inner_num) \
                                    * last_dim_output_num
                                burst_len = (do_inner_num * last_dim_output_num + self.block_num - 1) // self.block_num
                                self.tik_instance.data_move(self.output_gm[dst_offset],
                                                            origin_data_ub[_copy_idx * max_output_size], 0, 1,
                                                            burst_len, 0, 0)
                        with self.tik_instance.if_scope(align_tail == 0):
                            burst_len = (do_inner_num * last_dim_output_num + self.block_num - 1) // self.block_num
                            dst_offset = \
                                output_outer_offset + \
                                (self.tiling_pading_value[-2][0] + second_dim_start + \
                                 (do_outer_num - 1) * do_inner_num) \
                                * last_dim_output_num
                            self.tik_instance.data_move(self.output_gm[dst_offset],
                                                        origin_data_ub[(do_outer_num - 1) * max_output_size], 0, 1,
                                                        burst_len, 0, 0)
                        with self.tik_instance.elif_scope(align_tail != 0):
                            burst_len = (do_inner_num * last_dim_output_num + self.block_num - 1) // self.block_num
                            dst_offset = \
                                output_outer_offset + \
                                (self.tiling_pading_value[-2][0] + second_dim_start + \
                                 (do_outer_num - 1) * do_inner_num) \
                                * last_dim_output_num
                            self.tik_instance.data_move(self.output_gm[dst_offset],
                                                        origin_data_ub[(do_outer_num - 1) * max_output_size], 0, 1,
                                                        burst_len - 1, 0, 0)
                            # step9. process tail vnchwconv for the last line
                            origin_output_data_ub_list = \
                                [tail_data_ub[i * self.block_num]
                                for i in range(0, Constant.TRANS_MIN_BLKS)]
                            vnchw_output_data_ub_list = \
                                [vnchw_data_ub_a[i * 16 + (do_inner_num * last_dim_output_num - self.block_num) * 16]
                                for i in range(0, Constant.TRANS_MIN_BLKS)]
                            self.tik_instance.vnchwconv(False, False, 
                                                        origin_output_data_ub_list, vnchw_output_data_ub_list, 
                                                        2, vnchw_dst_stride1, vnchw_src_stride1)
                            # step10. conv tail fp16 to int8
                            self.conv_fp16_to_one_byte_dtype(tail_data_ub[(do_outer_num - 1) * self.block_num],
                                                            origin_data_ub, self.block_num)
                            dst_offset = \
                                output_outer_offset + \
                                (self.tiling_pading_value[-2][0] + second_dim_start + do_outer_num * do_inner_num) \
                                * last_dim_output_num - self.block_num
                            # step11. data_move tail to gm
                            self.tik_instance.data_move(self.output_gm[dst_offset],
                                                        origin_data_ub,
                                                        0, 1, 1, 0, 0)
                    # step8. data_move_to_gm
                    with self.tik_instance.else_scope():
                        with self.tik_instance.new_stmt_scope(disable_sync=True):
                            with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                                burst_len = (do_inner_num * last_dim_output_num + self.block_num - 1) // self.block_num
                                dst_offset = \
                                    output_outer_offset + \
                                    (self.tiling_pading_value[-2][0] + second_dim_start + _copy_idx * do_inner_num) \
                                    * self.output_offset[-1]
                                self.tik_instance.data_move(self.output_gm[dst_offset],
                                                            origin_data_ub[_copy_idx * max_output_size], 0, 1,
                                                            burst_len, 0, 0)

            origin_data_ub_ping = self.tik_instance.Tensor(self.x_dtype, (max_line_in_ub * max_output_size,),
                                                           name="origin_data_ub_ping",
                                                           scope=tik.scope_ubuf)
            vnchw_data_ub_ping = self.tik_instance.Tensor("float16", (max_line_in_ub * max_output_size,),
                                                          name="vnchw_data_ub_ping",
                                                          scope=tik.scope_ubuf)

            origin_output_data_ub_ping = self.tik_instance.Tensor("float16", (max_line_in_ub * max_output_size,),
                                                                  name="origin_output_data_ub_ping",
                                                                  scope=tik.scope_ubuf)
            tail_data_ub_ping = self.tik_instance.Tensor("float16", (16 * 16 * Constant.TWO_PIECES,),
                                                        name="tail_data_ub_ping",
                                                        scope=tik.scope_ubuf)

            origin_data_ub_pang = self.tik_instance.Tensor(self.x_dtype, (max_line_in_ub * max_output_size,),
                                                           name="origin_data_ub_pang",
                                                           scope=tik.scope_ubuf)
            vnchw_data_ub_pang = self.tik_instance.Tensor("float16", (max_line_in_ub * max_output_size,),
                                                          name="vnchw_data_ub_ping",
                                                          scope=tik.scope_ubuf)

            origin_output_data_ub_pang = self.tik_instance.Tensor("float16", (max_line_in_ub * max_output_size,),
                                                                  name="origin_output_data_ub_ping",
                                                                  scope=tik.scope_ubuf)
            tail_data_ub_pang = self.tik_instance.Tensor("float16", (16 * 16 * Constant.TWO_PIECES,),
                                                        name="tail_data_ub_pang",
                                                        scope=tik.scope_ubuf)

            ping_ub_list = [
                origin_data_ub_ping, vnchw_data_ub_ping, origin_output_data_ub_ping, tail_data_ub_ping
            ]
            pang_ub_list = [
                origin_data_ub_pang, vnchw_data_ub_pang, origin_output_data_ub_pang, tail_data_ub_pang
            ]
            with self.tik_instance.for_range(0, self.core_outer_num // 2) as _outer_idx:
                _run_one_outer(_outer_idx * 2, ping_ub_list)
                _run_one_outer(_outer_idx * 2 + 1, pang_ub_list)
            with self.tik_instance.if_scope(self.core_outer_num % 2 == 1):
                _run_one_outer(self.core_outer_num - 1, ping_ub_list)

        with self.tik_instance.for_range(0, second_dim_outer_loop_num_ceil) as second_dim_outer_idx:
            second_dim_outer_start = second_dim_outer_idx * second_dim_outer_cut_num * penultimate_dim_cut_num
            second_dim_outer_process_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_process_num")
            second_dim_outer_process_num.set_as(second_dim_outer_sigment_ub[second_dim_outer_idx //
                                                                            second_dim_outer_loop_num_floor])
            run_outer_by_outer(second_dim_outer_start, penultimate_dim_cut_num, second_dim_outer_process_num,
                               loop_align_tail)

        with self.tik_instance.if_scope(second_dim_total_loop_tail != 0):
            second_dim_outer_tail_start = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_tail_start")
            second_dim_outer_tail_start.set_as((penultimate_dim_input_num // penultimate_dim_cut_num) * \
                                                penultimate_dim_cut_num)
            with self.tik_instance.if_scope(second_dim_total_loop_tail * last_dim_output_num < self.block_num):
                new_tail_num = (self.block_num + last_dim_output_num - 1) // last_dim_output_num
                second_dim_outer_tail_start.set_as(second_dim_outer_tail_start - new_tail_num +
                                                   second_dim_total_loop_tail)
                second_dim_total_loop_tail.set_as(new_tail_num)
            run_outer_by_outer(second_dim_outer_tail_start, second_dim_total_loop_tail, 1, tail_align_tail)
            
    def do_tiling_key_mode_3(self, is_last_output_algin=False):
        """
        do_tiling_key_mode_3 when tiling key = 3
        """
        max_line_in_ub = 16
        max_output_size = self.cal_max_output_ub(max_line_in_ub)
        first_dim_input_num = self.tiling_input_shape[-3]
        second_dim_input_num = self.tiling_input_shape[-2]
        third_dim_input_num = self.tiling_input_shape[-1]
        third_dim_output_num = self.tiling_output_shape[-1]

        first_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="first_dim_cut_num")
        first_dim_cut_num.set_as(max_line_in_ub)
        second_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_cut_num")
        second_dim_cut_num.set_as(max_output_size // third_dim_output_num)
        with self.tik_instance.if_scope(first_dim_cut_num > first_dim_input_num):
            first_dim_cut_num.set_as(first_dim_input_num)
        with self.tik_instance.if_scope(second_dim_cut_num > second_dim_input_num):
            second_dim_cut_num.set_as(second_dim_input_num)

        # cut inner first dim and second dim info
        first_dim_loop_num_ceil = (first_dim_input_num + first_dim_cut_num - 1) // first_dim_cut_num
        first_dim_loop_num_floor = first_dim_input_num // first_dim_cut_num
        first_dim_tail_num = first_dim_input_num % first_dim_cut_num
        second_dim_loop_num_ceil = (second_dim_input_num + second_dim_cut_num - 1) // second_dim_cut_num
        second_dim_loop_num_floor = second_dim_input_num // second_dim_cut_num
        second_dim_tail_num = second_dim_input_num % second_dim_cut_num
        first_dim_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                        name="first_dim_sigment_ub",
                                                        scope=tik.scope_ubuf)
        second_dim_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                         name="second_dim_sigment_ub",
                                                         scope=tik.scope_ubuf)
        first_dim_sigment_ub[0].set_as(first_dim_cut_num)
        first_dim_sigment_ub[1].set_as(first_dim_tail_num)
        second_dim_sigment_ub[0].set_as(second_dim_cut_num)
        second_dim_sigment_ub[1].set_as(second_dim_tail_num)

        with self.tik_instance.for_range(0, first_dim_loop_num_ceil) as first_dim_idx:
            first_dim_start = first_dim_idx * first_dim_cut_num
            first_dim_process_num = self.tik_instance.Scalar(dtype="int64", name="first_dim_process_num")
            first_dim_process_num.set_as(first_dim_sigment_ub[first_dim_idx // first_dim_loop_num_floor])
            with self.tik_instance.for_range(0, second_dim_loop_num_ceil) as second_dim_idx:
                second_dim_process_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_process_num")
                second_dim_process_num.set_as(second_dim_sigment_ub[second_dim_idx // second_dim_loop_num_floor])
                second_dim_start = second_dim_idx * second_dim_cut_num

                vnchw_src_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride0", init_value=1)
                vnchw_dst_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride0", init_value=16)
                vnchw_src_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride1", init_value=16)
                vnchw_dst_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride1", init_value=1)
                vnchw_repeat0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat0", init_value=1)
                vnchw_repeat1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat1", init_value=1)
                vnchw_repeat0.set_as(
                    ((second_dim_process_num * third_dim_input_num) + self.block_num - 1) // self.block_num)
                vnchw_repeat1.set_as(
                    ((second_dim_process_num * third_dim_output_num) + self.block_num - 1) // self.block_num)
                with self.tik_instance.if_scope(vnchw_repeat0 == 1):
                    vnchw_src_stride0.set_as(0)
                    vnchw_dst_stride0.set_as(0)
                with self.tik_instance.if_scope(vnchw_repeat1 == 1):
                    vnchw_src_stride1.set_as(0)
                    vnchw_dst_stride1.set_as(0)

                def _run_one_outer(_outer_num_idx, ub_list):
                    """
                    run_one_outer
                    :param _outer_num_idx:
                    :param ub_list:
                    :return:
                    None
                    """
                    origin_data_ub, vnchw_data_ub, origin_output_tail_data_ub = ub_list
                    input_outer_idx = _outer_num_idx + self.core_outer_start
                    input_gm_offset = input_outer_idx * self.input_offset[3]
                    output_outer_offset = self.tik_instance.Scalar(dtype="int64", name="output_outer_offset")
                    output_outer_offset.set_as(
                        self.get_output_outer_idx(input_gm_offset, 5) +
                        self.tiling_pading_value[4][0] * self.output_offset[5])

                    # step1. copy 16 dims in origin_data_ub
                    with self.tik_instance.for_range(0, first_dim_process_num) as _copy_idx:
                        burst_len = \
                            ((second_dim_process_num * third_dim_input_num) + self.block_num - 1) // self.block_num
                        src_offset = \
                            (first_dim_start + _copy_idx) * self.input_offset[6] \
                            + second_dim_start * self.input_offset[7]
                        self.tik_instance.data_move(origin_data_ub[_copy_idx * max_output_size],
                                                    self.input_gm[input_gm_offset + src_offset], 0, 1, burst_len, 0, 0)

                    # step2. vnchw 16 dims origin_data_ub to vnchw_data_ub
                    origin_data_ub_list = [origin_data_ub[i * max_output_size] \
                                           for i in range(0, Constant.TRANS_MIN_BLKS)]
                    vnchw_data_ub_list = [vnchw_data_ub[i * 16] for i in range(0, Constant.TRANS_MIN_BLKS)]
                    self.tik_instance.vnchwconv(False, False, vnchw_data_ub_list, origin_data_ub_list, vnchw_repeat0,
                                                vnchw_dst_stride0, vnchw_src_stride0)
                    # step2.1 Reuse origin_data_ub as vnchw_output_data_ub，need to fill origin_data_ub with pad_scalar
                    self.tik_instance.vector_dup(self.block_num * 8, origin_data_ub, self.pad_scalar,
                                                max_line_in_ub * max_output_size // self.block_num // 8, 1, 8)
                    pad_left = self.tiling_pading_value[-1][0]
                    pad_right = self.tiling_pading_value[-1][1]
                    # step3. rearange vnchw_data_ub to vnchw_output_data_ub
                    # step3.0 copy input data to vnchw_output_data_ub with datamove

                    burst_num = second_dim_process_num
                    burst_len = third_dim_input_num
                    src_offset = 0
                    dst_offset = pad_left * self.block_num
                    src_stride = 0
                    dst_stride = pad_left + pad_right
                    self.tik_instance.data_move(origin_data_ub[dst_offset], vnchw_data_ub[src_offset], 0,
                                                burst_num, burst_len, src_stride, dst_stride)

                    # step4. vnchw vnchw_output_data_ub to 16 dims origin_output_data_ub
                    origin_output_data_ub_list = \
                        [vnchw_data_ub[i * max_output_size] for i in range(0, Constant.TRANS_MIN_BLKS)]
                    vnchw_output_data_ub_list = \
                        [origin_data_ub[i * 16] for i in range(0, Constant.TRANS_MIN_BLKS)]
                    self.tik_instance.vnchwconv(False, False, origin_output_data_ub_list, vnchw_output_data_ub_list,
                                                vnchw_repeat1, vnchw_dst_stride1, vnchw_src_stride1)

                    # step5. copy 16 dims to output
                    with self.tik_instance.for_range(0, first_dim_process_num) as _copy_idx:
                        burst_len = (second_dim_process_num * third_dim_output_num) // self.block_num
                        dst_offset = \
                            output_outer_offset \
                            + (self.tiling_pading_value[5][0] + (first_dim_start + _copy_idx)) \
                            * self.output_offset[6] \
                            + (self.tiling_pading_value[6][0] + second_dim_start) * self.output_offset[7]
                        self.tik_instance.data_move(self.output_gm[dst_offset],
                                                    vnchw_data_ub[_copy_idx * max_output_size], 0, 1, burst_len,
                                                    0, 0)
                    # is_last_output_algin is True
                    if not is_last_output_algin:
                        copy_tail_offset = self.tik_instance.Scalar(dtype="int64", name="copy_tail_offset")
                        copy_tail_offset.set_as(third_dim_output_num % 16)
                        with self.tik_instance.if_scope(copy_tail_offset == 0):
                            copy_tail_offset.set_as(16)
                        with self.tik_instance.else_scope():
                            copy_tail_offset.set_as(16 - copy_tail_offset)
                        vnchw_repeat = 1
                        origin_output_tail_data_ub_list = \
                            [origin_output_tail_data_ub[i * 16] for i in range(0, Constant.TRANS_MIN_BLKS)]
                        vnchw_output_data_ub_list = \
                            [origin_data_ub[((third_dim_output_num * second_dim_process_num - 16) + i) * 16]
                             for i in range(0, Constant.TRANS_MIN_BLKS)]
                        self.tik_instance.vnchwconv(False, False, origin_output_tail_data_ub_list,
                                                    vnchw_output_data_ub_list, vnchw_repeat, 0, 0)

                        with self.tik_instance.for_range(0, first_dim_process_num) as _copy_idx:
                            dst_offset = \
                                output_outer_offset \
                                + (self.tiling_pading_value[5][0] + (first_dim_start + _copy_idx)) \
                                * self.output_offset[6] \
                                + (self.tiling_pading_value[6][0] + second_dim_start) * self.output_offset[7] \
                                + second_dim_process_num * third_dim_output_num - 16
                            self.tik_instance.data_move(self.output_gm[dst_offset],
                                                        origin_output_tail_data_ub[_copy_idx * 16], 0, 1,
                                                        16 // self.block_num, 0, 0)

                origin_data_ub_ping = self.tik_instance.Tensor(self.x_dtype, (max_line_in_ub * max_output_size,),
                                                               name="origin_data_ub_ping",
                                                               scope=tik.scope_ubuf)
                vnchw_data_ub_ping = self.tik_instance.Tensor(self.x_dtype, (max_line_in_ub * max_output_size,),
                                                              name="vnchw_data_ub_ping",
                                                              scope=tik.scope_ubuf)
                origin_output_tail_data_ub_ping = \
                    self.tik_instance.Tensor(self.x_dtype, (16 * 16,),
                                             name="origin_output_tail_data_ub_ping", scope=tik.scope_ubuf)

                origin_data_ub_pang = self.tik_instance.Tensor(self.x_dtype, (max_line_in_ub * max_output_size,),
                                                               name="origin_data_ub_pang",
                                                               scope=tik.scope_ubuf)
                vnchw_data_ub_pang = self.tik_instance.Tensor(self.x_dtype, (max_line_in_ub * max_output_size,),
                                                              name="vnchw_data_ub_ping",
                                                              scope=tik.scope_ubuf)
                origin_output_tail_data_ub_pang = \
                    self.tik_instance.Tensor(self.x_dtype, (16 * 16,),
                                             name="origin_output_tail_data_ub_ping", scope=tik.scope_ubuf)

                ping_ub_list = [origin_data_ub_ping, vnchw_data_ub_ping, origin_output_tail_data_ub_ping]
                pang_ub_list = [origin_data_ub_pang, vnchw_data_ub_pang, origin_output_tail_data_ub_pang]
                with self.tik_instance.for_range(0, self.core_outer_num // 2) as _outer_idx:
                    _run_one_outer(_outer_idx * 2, ping_ub_list)
                    _run_one_outer(_outer_idx * 2 + 1, pang_ub_list)
                with self.tik_instance.if_scope(self.core_outer_num % 2 == 1):
                    _run_one_outer(self.core_outer_num - 1, ping_ub_list)

    def do_tiling_key_mode_4(self, core_index):
        """
        do tiling key mode 4
        """
        def _data_move_in_out(temp_ub, gm_offset, block_num, ele_num):
            self.tik_instance.data_move(temp_ub, self.input_gm[gm_offset], 0, 1, block_num, 0, 0)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                if not is_nano():
                    self.tik_instance.data_move_pad(self.output_gm[gm_offset],
                                                    temp_ub, nburst=1, burst=ele_num * self.x_bytes_size,
                                                    dst_gap=0, src_gap=0, right_padding=0, left_padding=0,
                                                    padding_value=None)
                else:
                    burst_num = ele_num * self.x_bytes_size // Constant.MAX_UB_CNT_DATA_MOVE_PAD
                    ele_num_tail = ele_num * self.x_bytes_size % Constant.MAX_UB_CNT_DATA_MOVE_PAD
                    with self.tik_instance.if_scope(burst_num != 0):
                        self.tik_instance.data_move_pad(self.output_gm[gm_offset],
                                                    temp_ub, nburst=burst_num, burst=Constant.MAX_UB_CNT_DATA_MOVE_PAD,
                                                    dst_gap=0, src_gap=0, right_padding=0, left_padding=0,
                                                    padding_value=None)
                    self.tik_instance.data_move_pad(self.output_gm[gm_offset + 
                                        burst_num * Constant.MAX_UB_CNT_DATA_MOVE_PAD // self.x_bytes_size],
                                        temp_ub[burst_num * Constant.MAX_UB_CNT_DATA_MOVE_PAD // self.x_bytes_size],
                                            nburst=1, burst=ele_num_tail, dst_gap=0, src_gap=0,
                                            right_padding=0, left_padding=0, padding_value=None)
            else:
                self.tik_instance.data_move(self.output_gm[gm_offset], temp_ub, 0, 1, block_num, 0, 0)
        offset_gm = self.tik_instance.Scalar(dtype='int32', name='offset_gm')
        align_burst = self.tik_instance.Scalar(dtype='int32', name='align_burst')
        total_output_tensor_each_core = self.tik_instance.Scalar(dtype='int32', name='total_output_tensor_each_core')
        total_output_tensor = self.tik_instance.Scalar(dtype='int32', name='total_output_tensor', init_value=1)
        for ele in self.tiling_output_shape:
            total_output_tensor.set_as(total_output_tensor * ele)
        num_each_block = self.tik_instance.Scalar(dtype='int32')
        num_each_block.set_as(Constant.BLOCK // self.x_bytes_size)
        core_nums = self.tik_instance.Scalar(dtype='int32')
        core_nums.set_as(self.tiling_core_num)
        total_output_tensor_each_core.set_as((total_output_tensor - 1) // core_nums + 1)
        with self.tik_instance.for_range(0, Constant.BLOCK) as i:
            with self.tik_instance.if_scope(total_output_tensor_each_core < num_each_block):
                core_nums.set_as(core_nums - 1)
                total_output_tensor_each_core.set_as((total_output_tensor - 1) // core_nums + 1)
        total_output_tensor_each_core.set_as(((total_output_tensor_each_core - 1) // num_each_block + 1) \
                                             * num_each_block)
        core_nums.set_as((total_output_tensor - 1) // total_output_tensor_each_core + 1)
        with self.tik_instance.if_scope(core_index < core_nums - 1):
            with self.tik_instance.new_stmt_scope():
                move_ub = self.tik_instance.Tensor(dtype=self.x_dtype,
                                    shape=(self.ub_number,),
                                    name='move_ub',
                                    scope=tik.scope_ubuf)
                offset_gm.set_as(core_index * total_output_tensor_each_core)
                with self.tik_instance.if_scope(total_output_tensor_each_core // self.ub_number > 0):
                    with self.tik_instance.for_range(0, total_output_tensor_each_core // self.ub_number) as i:
                        _data_move_in_out(temp_ub=move_ub, 
                                        gm_offset=offset_gm + i * self.ub_number,
                                        block_num=self.ub_number // num_each_block, 
                                        ele_num=self.ub_number)
                with self.tik_instance.if_scope(total_output_tensor_each_core % self.ub_number > 0):
                    align_burst.set_as(((total_output_tensor_each_core % self.ub_number) - 1) // num_each_block + 1)
                    _data_move_in_out(temp_ub=move_ub, 
                                gm_offset=offset_gm + total_output_tensor_each_core // self.ub_number * self.ub_number, 
                                block_num=align_burst, 
                                ele_num=total_output_tensor_each_core % self.ub_number)
        with self.tik_instance.elif_scope(core_index == core_nums - 1):
            with self.tik_instance.new_stmt_scope():
                move_ub = self.tik_instance.Tensor(dtype=self.x_dtype,
                                                   shape=(self.ub_number,),
                                                   name='move_ub',
                                                   scope=tik.scope_ubuf)
                offset_gm.set_as(core_index * total_output_tensor_each_core)
                with self.tik_instance.if_scope((total_output_tensor - offset_gm) // self.ub_number > 0):
                    with self.tik_instance.for_range(0, (total_output_tensor - offset_gm) // self.ub_number) as i:
                        _data_move_in_out(temp_ub=move_ub, 
                                    gm_offset=offset_gm + i * self.ub_number, 
                                    block_num=self.ub_number // num_each_block, 
                                    ele_num=self.ub_number)
                with self.tik_instance.if_scope((total_output_tensor - offset_gm) % self.ub_number > 0):
                    align_burst.set_as((((total_output_tensor - offset_gm) % self.ub_number) - 1) // num_each_block + 1)
                    _data_move_in_out(temp_ub=move_ub, 
                                gm_offset=offset_gm + total_output_tensor_each_core // self.ub_number * self.ub_number, 
                                block_num=align_burst, 
                                ele_num=(total_output_tensor - offset_gm) % self.ub_number)

    def do_tiling_key_mode_pure_datamove(self, core_index):
        """
        tiling mode for int8, pure data_move solution, alse for other dtypes in 310B and 910B
        last dim less than ub_number
        """
        second_dim_input_num = self.tik_instance.Scalar(dtype='int32', init_value=1)
        for ele in self.tiling_input_shape[:-2]:
            second_dim_input_num.set_as(second_dim_input_num * ele)     
        
        last_dim = self.tiling_input_shape[-1]
        last_dim_block_ceil = (last_dim - 1) // self.block_num + 1
        last_dim_ceil = last_dim_block_ceil * self.block_num
        last_dim_block_floor = last_dim // self.block_num
        last_dim_tail = last_dim % self.block_num
        last_dim_floor = last_dim_block_floor * self.block_num
        penultimate_dim = self.tiling_input_shape[-2]

        core_nums = self.tik_instance.Scalar(dtype='int32')
        core_nums.set_as(self.tiling_core_num)
        move_ub = self.tik_instance.Tensor(dtype=self.x_dtype,
                                           shape=(self.ub_number,),
                                           name='move_ub',
                                           scope=tik.scope_ubuf)
        core_slices = self.tik_instance.Scalar(dtype='int64') # cut data into slices according to core_nums
        core_slices.set_as((core_nums - 1) // second_dim_input_num + 1)
        ub_slices = self.tik_instance.Scalar(dtype='int64') # cut data into slices according to data_size
        data_move_in = self.tik_instance.Scalar(dtype='int32')
        slice_num = self.tik_instance.Scalar(dtype='int64', init_value=1)
    
        def _run_one_outer():
            """
            data_move process when penultimate dim is cut
            """
            with self.tik_instance.for_range(0, each_core_repeat_times) as each_core_repeat_id:
                with self.tik_instance.new_stmt_scope():
                    input_line = self.tik_instance.Scalar(dtype="int32")
                    with self.tik_instance.if_scope(core_index < core_pre_num):
                        input_line.set_as(loop_num_each_core_pre * core_index + each_core_repeat_id)
                    with self.tik_instance.else_scope():
                        input_line.set_as(loop_num_each_core_tail * core_index + core_pre_num + each_core_repeat_id)
                    index_0 = input_line // slice_num
                    pre_and_tail = input_line % slice_num
                    index_1 = self.tik_instance.Scalar(dtype="int32")
                    with self.tik_instance.if_scope(pre_and_tail < slice_length_num):
                        index_1.set_as(pre_and_tail * slice_length)
                    with self.tik_instance.else_scope():
                        index_1.set_as(slice_length_num * slice_length + (pre_and_tail - slice_length_num) * slice_tail)
                    input_offset = index_0 * last_dim * penultimate_dim + index_1 * last_dim 
                    output_offset = self.get_output_outer_idx(input_offset, outer_num=7)

                    data_move_in.set_as(slice_length)
                    with self.tik_instance.if_scope(pre_and_tail >= slice_length_num):
                        data_move_in.set_as(slice_tail)
                    
                    with self.tik_instance.for_range(0, data_move_in) as line_id:
                        if tbe_platform.api_check_support("tik.data_move_pad"):
                            self.tik_instance.data_move_pad(move_ub[line_id * last_dim_ceil],
                                                            self.input_gm[input_offset + line_id * last_dim],
                                                            1, last_dim * self.x_bytes_size, 0, 0)
                            self.tik_instance.data_move_pad(self.output_gm[output_offset +
                                                                            line_id * last_dim_out +
                                                                            pad_left],
                                                            move_ub[line_id * last_dim_ceil],
                                                            1, last_dim * self.x_bytes_size, 0, 0)
                        else:
                            self.tik_instance.data_move(move_ub[line_id * last_dim_ceil], 
                                                        self.input_gm[input_offset + line_id * last_dim],
                                                        0, 1, last_dim_block_ceil, 0, 0)
                            self.tik_instance.data_move(self.output_gm[output_offset + 
                                                                       line_id * last_dim_out + pad_left],
                                                        move_ub[line_id * last_dim_ceil], 
                                                        0, 1, last_dim_block_floor, 0, 0)
                            self.tik_instance.data_move(move_ub[line_id * last_dim_ceil], 
                                                        self.input_gm[input_offset + 
                                                                      (line_id + 1) * last_dim - self.block_num],
                                                        0, 1, 1, 0, 0)
                            self.tik_instance.data_move(self.output_gm[output_offset +
                                                                       line_id * last_dim_out +
                                                                       pad_left + last_dim - self.block_num],
                                                        move_ub[line_id * last_dim_ceil],
                                                        0, 1, 1, 0, 0)
        # cut penultimate dim into slices
        penu_threshold = self.ub_number // last_dim_ceil
        ub_slices.set_as((penultimate_dim - 1) // penu_threshold + 1)
        if not is_nano():
            self.tik_instance.scalar_max(slice_num, core_slices, ub_slices)
            self.tik_instance.scalar_min(slice_num, slice_num, penultimate_dim)
        slice_length = (penultimate_dim - 1) // slice_num + 1
        slice_tail = slice_length - 1
        slice_length_num = penultimate_dim - slice_tail * slice_num

        pad_left = self.tiling_pading_value[-1][0]
        pad_right = self.tiling_pading_value[-1][1]
        last_dim_out = pad_left + last_dim + pad_right
        loop_num_each_core_pre = (second_dim_input_num * slice_num - 1) // core_nums + 1
        loop_num_each_core_tail = loop_num_each_core_pre - 1
        core_pre_num = second_dim_input_num * slice_num - loop_num_each_core_tail * core_nums
        each_core_repeat_times = self.tik_instance.Scalar(dtype='int32')
        with self.tik_instance.if_scope(core_index < core_pre_num):
            each_core_repeat_times.set_as(loop_num_each_core_pre)
        with self.tik_instance.else_scope():
            each_core_repeat_times.set_as(loop_num_each_core_tail)    
        with self.tik_instance.if_scope(core_index < core_nums):
            _run_one_outer()

    def do_tiling_key_mode_last_dim_cut(self, core_index):
        """
        tiling mode for int8, pure data_move solution, else for other dtypes in 310B and 910B
        last dim larger than ub_number
        """
        second_dim_input_num = self.tik_instance.Scalar(dtype='int32', init_value=1)
        for ele in self.tiling_input_shape[:-1]:
            second_dim_input_num.set_as(second_dim_input_num * ele)     
        
        last_dim = self.tiling_input_shape[-1]
        core_nums = self.tik_instance.Scalar(dtype='int32')
        core_nums.set_as(self.tiling_core_num)
        move_ub = self.tik_instance.Tensor(dtype=self.x_dtype,
                                           shape=(self.ub_number,),
                                           name='move_ub',
                                           scope=tik.scope_ubuf)
        core_slices = self.tik_instance.Scalar(dtype='int64') # cut data into slices according to core_nums
        core_slices.set_as((core_nums - 1) // second_dim_input_num + 1)
        ub_slices = self.tik_instance.Scalar(dtype='int64') # cut data into slices according to data_size
        slice_num = self.tik_instance.Scalar(dtype='int64', init_value=1)
        
        def _data_move_in_out(temp_ub, input_offset, data_move_in_num_ceil,
                              data_move_out_num_floor, process_num):
            output_offset = self.get_output_outer_idx(input_offset, outer_num=8)
            if tbe_platform.api_check_support("tik.data_move_pad"):
                self.tik_instance.data_move_pad(temp_ub,
                                                self.input_gm[input_offset],
                                                1, process_num * self.x_bytes_size, 0, 0)
                self.tik_instance.data_move_pad(self.output_gm[output_offset],
                                                temp_ub,
                                                1, process_num * self.x_bytes_size, 0, 0)
            else:
                self.tik_instance.data_move(temp_ub, 
                                            self.input_gm[input_offset],
                                            0, 1, data_move_in_num_ceil, 0, 0)
                self.tik_instance.data_move(self.output_gm[output_offset],
                                            temp_ub, 0, 1,
                                            data_move_out_num_floor, 0, 0)
                self.tik_instance.data_move(temp_ub, 
                                            self.input_gm[input_offset + process_num - self.block_num],
                                            0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.output_gm[output_offset + process_num - self.block_num],
                                            temp_ub,
                                            0, 1, 1, 0, 0)
                
        def _run_one_outer():
            """
            data_move process when last dim is cut
            """
            with self.tik_instance.for_range(0, each_core_repeat_times) as each_core_repeat_id:
                with self.tik_instance.new_stmt_scope():
                    input_line = self.tik_instance.Scalar(dtype="int32")
                    with self.tik_instance.if_scope(core_index < core_pre_num):
                        input_line.set_as(loop_num_each_core_pre * core_index + each_core_repeat_id)
                    with self.tik_instance.else_scope():
                        input_line.set_as(loop_num_each_core_tail * core_index + core_pre_num + each_core_repeat_id)
                    index_0 = input_line // slice_num
                    pre_and_tail = input_line % slice_num
                    index_1 = self.tik_instance.Scalar(dtype="int32")
                    with self.tik_instance.if_scope(pre_and_tail < slice_length_num):
                        index_1.set_as(pre_and_tail * slice_length)
                    with self.tik_instance.else_scope():
                        index_1.set_as(slice_length_num * slice_length + (pre_and_tail - slice_length_num) * slice_tail)
                    input_offset = index_0 * last_dim + index_1

                    with self.tik_instance.if_scope(pre_and_tail < slice_length_num):
                        _data_move_in_out(move_ub, input_offset, slice_length_block_ceil,
                                          slice_length_block_floor, slice_length)

                    with self.tik_instance.if_scope(pre_and_tail >= slice_length_num):
                        _data_move_in_out(move_ub, input_offset, slice_tail_block_ceil,
                                          slice_tail_block_floor, slice_tail)
                            
        # cut last dim into slices
        ub_slices.set_as((last_dim - 1) // self.ub_number + 1)
        if not is_nano():
            self.tik_instance.scalar_max(slice_num, core_slices, ub_slices)
        slice_length = (last_dim - 1) // slice_num + 1
        slice_tail = slice_length - 1
        slice_length_num = last_dim - slice_tail * slice_num
        slice_length_block_floor = slice_length // self.block_num
        slice_length_floor = slice_length_block_floor * self.block_num
        slice_length_block_ceil = (slice_length - 1) // self.block_num + 1
        slice_length_tail = slice_length - slice_length_floor

        slice_tail_block_floor = slice_tail // self.block_num
        slice_tail_floor = slice_tail_block_floor * self.block_num
        slice_tail_block_ceil = (slice_tail - 1) // self.block_num + 1
        slice_tail_tail = slice_tail - slice_tail_floor

        loop_num_each_core_pre = (second_dim_input_num * slice_num - 1) // core_nums + 1
        loop_num_each_core_tail = loop_num_each_core_pre - 1
        core_pre_num = second_dim_input_num * slice_num - loop_num_each_core_tail * core_nums
        each_core_repeat_times = self.tik_instance.Scalar(dtype='int32')
        with self.tik_instance.if_scope(core_index < core_pre_num):
            each_core_repeat_times.set_as(loop_num_each_core_pre)
        with self.tik_instance.else_scope():
            each_core_repeat_times.set_as(loop_num_each_core_tail)    
        with self.tik_instance.if_scope(core_index < core_nums):
            _run_one_outer()
                    
    def do_high_performance_branch_special_scenarios(self):
        """
        do_high_performance_branch_special_scenarios when high_performance_branch = 1
        """
        self.input_gm = self.input_gm.reinterpret_cast_to(self.inner_dtype)
        self.output_gm = self.output_gm.reinterpret_cast_to(self.inner_dtype)
        max_line_in_ub = 16
        max_output_size = self.special_max_output_size * 2 
        second_dim_input_num = self.tiling_input_shape[-2] 
        third_dim_input_num = self.tiling_input_shape[-1] * 2
        third_dim_output_num = self.tiling_output_shape[-1] * 2 

        first_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="first_dim_cut_num")
        second_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_cut_num")

        second_dim_cut_num.set_as(max_output_size // third_dim_output_num) 
        with self.tik_instance.if_scope(second_dim_cut_num > second_dim_input_num):
            second_dim_cut_num.set_as(second_dim_input_num)

        first_dim_cut_num.set_as(max_line_in_ub * second_dim_cut_num)

        # cut inner first dim and second dim info
        second_dim_total_loop_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_total_loop_num")
        second_dim_total_loop_tail = self.tik_instance.Scalar(dtype="int64", name="second_dim_total_loop_tail")
        second_dim_total_loop_num.set_as(second_dim_input_num // second_dim_cut_num) 
        second_dim_total_loop_tail.set_as(second_dim_input_num % second_dim_cut_num)

        second_dim_outer_cut_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_cut_num")
        second_dim_outer_cut_num.set_as(max_line_in_ub)
        with self.tik_instance.if_scope(second_dim_total_loop_num < max_line_in_ub):
            second_dim_outer_cut_num.set_as(second_dim_total_loop_num)

        second_dim_outer_loop_num_ceil = \
            (second_dim_total_loop_num + second_dim_outer_cut_num - 1) // second_dim_outer_cut_num
        second_dim_outer_loop_num_floor = second_dim_total_loop_num // second_dim_outer_cut_num

        second_dim_outer_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                               name="second_dim_outer_sigment_ub",
                                                               scope=tik.scope_ubuf)
        second_dim_outer_sigment_ub[0].set_as(second_dim_outer_cut_num) 
        second_dim_outer_sigment_ub[1].set_as(second_dim_total_loop_num % second_dim_outer_cut_num) 

        second_dim_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                         name="second_dim_sigment_ub",
                                                         scope=tik.scope_ubuf)
        second_dim_sigment_ub[0].set_as(second_dim_cut_num)
        second_dim_sigment_ub[1].set_as(second_dim_input_num % second_dim_cut_num)

        loop_align_tail = self.tik_instance.Scalar(dtype="int64", name="loop_align_tail")
        tail_align_tail = self.tik_instance.Scalar(dtype="int64", name="tail_align_tail")
        one_core_flag = self.tik_instance.Scalar(dtype="int64", name="one_core_flag", init_value=0)
        loop_align_tail.set_as((second_dim_cut_num * third_dim_output_num) % self.inner_block_num)
        tail_align_tail.set_as((second_dim_total_loop_tail * third_dim_output_num) % self.inner_block_num)
        with self.tik_instance.if_scope(third_dim_output_num * self.tiling_input_shape[-2] <= self.inner_block_num):
            loop_align_tail.set_as(0)
            tail_align_tail.set_as(0)
            one_core_flag.set_as(self.inner_block_num - 1)

        vnchw_src_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride0", init_value=1)
        vnchw_dst_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride0", init_value=16)
        vnchw_src_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride1", init_value=16)
        vnchw_dst_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride1", init_value=1)
        vnchw_repeat0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat0", init_value=1)
        vnchw_repeat1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat1", init_value=1)
        vnchw_repeat0.set_as(((second_dim_cut_num * third_dim_input_num) + \
            self.inner_block_num - 1) // self.inner_block_num)
        vnchw_repeat1.set_as(((second_dim_cut_num * third_dim_output_num) + \
            self.inner_block_num - 1) // self.inner_block_num)
        with self.tik_instance.if_scope(vnchw_repeat0 == 1):
            vnchw_src_stride0.set_as(0)
            vnchw_dst_stride0.set_as(0)
        with self.tik_instance.if_scope(vnchw_repeat1 == 1):
            vnchw_src_stride1.set_as(0)
            vnchw_dst_stride1.set_as(0)

        def run_outer_by_outer(second_dim_start, do_inner_num, do_outer_num, align_tail, disable_sync_mte3=False):
            """
            run_outer_by_outer
            :param
            second_dim_start:the second dim start of input
            :param
            do_inner_num: the number which do inner
            :param
            do_outer_num: the number which do outer
            :param
            align_tail: the tail of align
            :param
            disable_sync_mte3: disable_sync_mte3
            :return:
            None
            """

            def _run_one_outer(_outer_num_idx, ub_list):
                """
                _run_one_outer
                :param
                _outer_num_idx:
                :param
                ub_list:
                :return:
                none
                """
                dtype_rate = 2
                origin_data_ub, vnchw_data_ub, _, _ = ub_list
                _, _, origin_output_data_ub, origin_output_tail_data_ub = ub_list
                input_outer_idx = _outer_num_idx + self.core_outer_start
                input_gm_offset = input_outer_idx * self.input_offset[-2]
                output_outer_offset = self.tik_instance.Scalar(dtype="int64", name="output_outer_offset")
                output_outer_offset.set_as(self.get_output_outer_idx(input_gm_offset, 6) * dtype_rate)
                input_gm_offset *= dtype_rate
                # step1. copy 16 dims in origin_data_ub
                with self.tik_instance.new_stmt_scope(disable_sync=True):
                    with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                        burst_len = ((do_inner_num * third_dim_input_num) + \
                            self.inner_block_num - 1) // self.inner_block_num
                        src_offset = (second_dim_start + _copy_idx * do_inner_num) * third_dim_input_num
                        self.tik_instance.data_move(origin_data_ub[_copy_idx * max_output_size],
                                                    self.input_gm[input_gm_offset + src_offset], 0, 1, burst_len, 0, 0)
                # step2. vnchw 16 dims origin_data_ub to vnchw_data_ub
                origin_data_ub_list = [origin_data_ub[i * max_output_size] for i in range(0, Constant.TRANS_MIN_BLKS)]
                vnchw_data_ub_list = [vnchw_data_ub[i * 16] for i in range(0, Constant.TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False, vnchw_data_ub_list, origin_data_ub_list, vnchw_repeat0,
                                            vnchw_dst_stride0, vnchw_src_stride0)

                pad_left = self.tiling_pading_value[-1][0] * dtype_rate
                pad_right = self.tiling_pading_value[-1][1] * dtype_rate
                # step2.1 Reuse origin_data_ub as vnchw_output_data_ub，need to fill origin_data_ub with pad_scalar
                origin_data_ub = origin_data_ub.reinterpret_cast_to(self.x_dtype)
                self.tik_instance.vector_dup(self.inner_block_num * 4, origin_data_ub, self.pad_scalar,
                                             max_line_in_ub * max_output_size // self.inner_block_num // 8, 1, 8)
                origin_data_ub = origin_data_ub.reinterpret_cast_to('float16')
                # step3. rearange vnchw_data_ub to vnchw_output_data_ub
                # step3.0 copy input data to vnchw_output_data_ub with datamove
                burst_num = do_inner_num
                burst_len = third_dim_input_num
                src_offset = 0
                dst_offset = pad_left * self.inner_block_num
                src_stride = 0
                dst_stride = pad_left + pad_right
                self.tik_instance.data_move(origin_data_ub[dst_offset], vnchw_data_ub[src_offset], 0, burst_num,
                                            burst_len, src_stride, dst_stride)

                # step4. vnchw vnchw_output_data_ub to 16 dims origin_output_data_ub
                origin_output_data_ub_list = \
                    [origin_output_data_ub[i * max_output_size] for i in range(0, Constant.TRANS_MIN_BLKS)]
                vnchw_output_data_ub_list = \
                    [origin_data_ub[i * 16] for i in range(0, Constant.TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False, origin_output_data_ub_list, vnchw_output_data_ub_list,
                                            vnchw_repeat1, vnchw_dst_stride1, vnchw_src_stride1)

                # step5. copy 16 dims to output
                # step5.1 copy do_outer_num - 1 lines to output use ceil_div block
                with self.tik_instance.if_scope(do_inner_num * third_dim_output_num % self.inner_block_num != 0):
                    with self.tik_instance.new_stmt_scope(disable_sync=disable_sync_mte3):
                        with self.tik_instance.for_range(0, do_outer_num - 1) as _copy_idx:
                            burst_len = (do_inner_num * third_dim_output_num + \
                                self.inner_block_num - 1) // self.inner_block_num
                            dst_offset = \
                                output_outer_offset + \
                                (self.tiling_pading_value[-2][0] + second_dim_start + _copy_idx * do_inner_num) \
                                * self.output_offset[-1] * dtype_rate
                            self.tik_instance.data_move(self.output_gm[dst_offset],
                                                        origin_output_data_ub[_copy_idx * max_output_size], 0, 1,
                                                        burst_len, 0, 0)
                        # step5.1 copy the last do_outer_num lines to output use floor_div block
                        burst_len = (do_inner_num * third_dim_output_num + one_core_flag) // self.inner_block_num
                        dst_offset = \
                            output_outer_offset + \
                            (self.tiling_pading_value[-2][0] + second_dim_start + (do_outer_num - 1) * do_inner_num) \
                            * self.output_offset[-1] * dtype_rate
                        self.tik_instance.data_move(self.output_gm[dst_offset],
                                                    origin_output_data_ub[(do_outer_num - 1) * max_output_size], 0, 1,
                                                    burst_len, 0, 0)

                    # step6. process tail for the last line
                    with self.tik_instance.if_scope(align_tail != 0):
                        origin_output_data_ub_list = \
                            [origin_output_tail_data_ub[i * 16] for i in range(0, Constant.TRANS_MIN_BLKS)]
                        vnchw_output_data_ub_list = \
                            [origin_data_ub[i * 16 + (do_inner_num * third_dim_output_num - 16) * 16]
                            for i in range(0, Constant.TRANS_MIN_BLKS)]
                        self.tik_instance.vnchwconv(False, False, 
                                                    origin_output_data_ub_list, vnchw_output_data_ub_list,
                                                    1, 0, 0)
                        burst_len = 1
                        dst_offset = \
                            output_outer_offset \
                            + (self.tiling_pading_value[-2][0] + second_dim_start + do_outer_num * do_inner_num) \
                            * self.output_offset[-1] * dtype_rate \
                            - self.inner_block_num
                        self.tik_instance.data_move(self.output_gm[dst_offset],
                                                    origin_output_tail_data_ub[(do_outer_num - 1) * 16], 0, 1,
                                                    burst_len, 0, 0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                            burst_len = (do_inner_num * third_dim_output_num + \
                                self.inner_block_num - 1) // self.inner_block_num
                            dst_offset = \
                                output_outer_offset + \
                                (self.tiling_pading_value[-2][0] + second_dim_start + _copy_idx * do_inner_num) \
                                * self.output_offset[-1] * dtype_rate
                            self.tik_instance.data_move(self.output_gm[dst_offset],
                                                        origin_output_data_ub[_copy_idx * max_output_size], 0, 1,
                                                        burst_len, 0, 0)

            origin_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                           name="origin_data_ub_ping",
                                                           scope=tik.scope_ubuf)
            vnchw_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                          name="vnchw_data_ub_ping",
                                                          scope=tik.scope_ubuf)

            origin_output_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, 
                                                                  (max_line_in_ub * max_output_size,),
                                                                  name="origin_output_data_ub_ping",
                                                                  scope=tik.scope_ubuf)
            origin_output_tail_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (16 * 16,),
                                                                       name="origin_output_tail_data_ub_ping",
                                                                       scope=tik.scope_ubuf)


            origin_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                           name="origin_data_ub_pang",
                                                           scope=tik.scope_ubuf)
            vnchw_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                          name="vnchw_data_ub_ping",
                                                          scope=tik.scope_ubuf)

            origin_output_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, 
                                                                  (max_line_in_ub * max_output_size,),
                                                                  name="origin_output_data_ub_ping",
                                                                  scope=tik.scope_ubuf)
            origin_output_tail_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (16 * 16,),
                                                                       name="origin_output_tail_data_ub_ping",
                                                                       scope=tik.scope_ubuf)


            ping_ub_list = [
                origin_data_ub_ping, vnchw_data_ub_ping, origin_output_data_ub_ping,
                origin_output_tail_data_ub_ping
            ]
            pang_ub_list = [
                origin_data_ub_pang, vnchw_data_ub_pang, origin_output_data_ub_pang,
                origin_output_tail_data_ub_pang
            ]
            with self.tik_instance.for_range(0, self.core_outer_num // 2) as _outer_idx:
                _run_one_outer(_outer_idx * 2, ping_ub_list)
                _run_one_outer(_outer_idx * 2 + 1, pang_ub_list)
            with self.tik_instance.if_scope(self.core_outer_num % 2 == 1):
                _run_one_outer(self.core_outer_num - 1, ping_ub_list)

        with self.tik_instance.for_range(0, second_dim_outer_loop_num_ceil) as second_dim_outer_idx:
            second_dim_outer_start = second_dim_outer_idx * second_dim_outer_cut_num * second_dim_cut_num
            second_dim_outer_process_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_process_num")
            second_dim_outer_process_num.set_as(second_dim_outer_sigment_ub[second_dim_outer_idx //
                                                                            second_dim_outer_loop_num_floor])
            run_outer_by_outer(second_dim_outer_start, second_dim_cut_num, second_dim_outer_process_num,
                               loop_align_tail)

        with self.tik_instance.if_scope(second_dim_total_loop_tail != 0):
            second_dim_outer_tail_start = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_tail_start")
            second_dim_outer_tail_start.set_as((second_dim_input_num // second_dim_cut_num) * second_dim_cut_num)
            with self.tik_instance.if_scope(second_dim_total_loop_tail * third_dim_output_num < self.inner_block_num):
                new_tail_num = (self.inner_block_num + third_dim_output_num - 1) // third_dim_output_num
                second_dim_outer_tail_start.set_as(second_dim_outer_tail_start - new_tail_num +
                                                   second_dim_total_loop_tail)
                second_dim_total_loop_tail.set_as(new_tail_num)

            run_outer_by_outer(second_dim_outer_tail_start, second_dim_total_loop_tail, 1, tail_align_tail)
        self.input_gm = self.input_gm.reinterpret_cast_to(self.x_dtype)
        self.output_gm = self.output_gm.reinterpret_cast_to(self.x_dtype)
        
    def conv_one_byte_dtype_to_fp16(self, src, dst, num, offset=0):
        """
        conv_one_byte_dtype_to_fp16
        """
        mask = 128
        loop = num // (mask * 255)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_conv(mask,
                                            '',
                                            dst[tmp_offset], 
                                            src[tmp_offset], 
                                            255, 
                                            8, 
                                            4)
            offset += loop * mask * 255
        repeat_time = (num % (mask * 255)) // mask
        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_conv(mask,
                                        '',
                                        dst[offset], 
                                        src[offset], 
                                        repeat_time, 
                                        8, 
                                        4)
            offset += repeat_time * mask
        last_num = num % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_conv(last_num,
                                        '',
                                        dst[offset], 
                                        src[offset], 
                                        1, 
                                        0, 
                                        0)

    def conv_fp16_to_one_byte_dtype(self, src, dst, num, offset=0):
        """
        conv_fp16_to_one_byte_dtype
        """
        mask = 128
        loop = num // (mask * 255)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_conv(mask,
                                            'floor',
                                            dst[tmp_offset], 
                                            src[tmp_offset], 
                                            255, 
                                            4, 
                                            8)
            offset += loop * mask * 255
        repeat_time = (num % (mask * 255)) // mask
        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_conv(mask,
                                        'floor',
                                        dst[offset], 
                                        src[offset], 
                                        repeat_time, 
                                        4, 
                                        8)
            offset += repeat_time * mask
        last_num = num % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_conv(last_num,
                                        'floor',
                                        dst[offset], 
                                        src[offset], 
                                        1, 
                                        0, 
                                        0)

    def do_pad(self, core_index):
        """
        do_pad with different tiling key
        """
        if self.version_info == "Ascend910" or self.version_info == "Ascend310":
            if self.x_dtype in self.four_bytes_dtype:
                with self.tik_instance.if_scope(self.high_performance_branch == 1):
                    with self.tik_instance.new_stmt_scope():
                        self.do_high_performance_branch_special_scenarios()
                with self.tik_instance.elif_scope(self.high_performance_branch == 0):
                    self.routing_by_tiling_key(core_index)   
            elif self.x_dtype in self.two_bytes_dtype:
                self.routing_by_tiling_key(core_index)
        elif self.version_info == "Ascend310B":
            if self.x_dtype in self.four_bytes_dtype or self.x_dtype in self.two_bytes_dtype:
                with self.tik_instance.if_scope(self.tiling_key == Constant.MODE2):
                    with self.tik_instance.new_stmt_scope():
                        self.do_tiling_key_mode_2()
                with self.tik_instance.else_scope():
                    self.routing_by_tiling_key_for_310b(core_index)
            elif self.x_dtype in self.one_bytes_dtype:
                with self.tik_instance.if_scope(self.tiling_key == Constant.PADV3_INT8_HIGH_PROFROMANCE):
                    with self.tik_instance.new_stmt_scope():
                        self.do_padv3_int8_high_profromance()
                with self.tik_instance.else_scope():
                    self.routing_by_tiling_key_for_310b(core_index)
        else:
            # for Ascend910B Ascend310P Ascend610 BS9SX1A AS31XM1X 
            if self.x_dtype in self.four_bytes_dtype or self.x_dtype in self.two_bytes_dtype:
                self.routing_by_tiling_key(core_index)
            elif self.x_dtype in self.one_bytes_dtype:
                with self.tik_instance.if_scope(self.tiling_key == Constant.PADV3_INT8_HIGH_PROFROMANCE):
                    with self.tik_instance.new_stmt_scope():
                        self.do_padv3_int8_high_profromance()
                with self.tik_instance.elif_scope(self.tiling_key == Constant.MODE4):
                    with self.tik_instance.new_stmt_scope():
                        # paddings are all 0
                        self.do_tiling_key_mode_4(core_index)
                with self.tik_instance.elif_scope(self.tiling_key == Constant.LAST_DIM_PAD_MODE):
                    with self.tik_instance.new_stmt_scope():
                        # pure data_move
                        self.do_tiling_key_mode_pure_datamove(core_index)
                with self.tik_instance.elif_scope(self.tiling_key == Constant.PADV3_DATA_MOVE_LAST_DIM_CUT):
                    with self.tik_instance.new_stmt_scope():
                        # Processing the final one-dimensional larger scene
                        self.do_tiling_key_mode_last_dim_cut(core_index)
                             
    def routing_by_tiling_key(self, core_index):
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE0):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_0()
        with self.tik_instance.elif_scope(self.tiling_key == Constant.MODE1):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_1()
        with self.tik_instance.elif_scope(self.tiling_key == Constant.MODE2):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_2()
        with self.tik_instance.elif_scope(self.tiling_key == Constant.MODE3):  
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_3()
        with self.tik_instance.elif_scope(self.tiling_key == Constant.MODE4): 
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_4(core_index)
        with self.tik_instance.elif_scope(self.tiling_key == Constant.LAST_DIM_PAD_MODE):
            with self.tik_instance.new_stmt_scope():
                # pure data_move
                self.do_tiling_key_mode_pure_datamove(core_index)
        with self.tik_instance.elif_scope(self.tiling_key == Constant.PADV3_DATA_MOVE_LAST_DIM_CUT):
            with self.tik_instance.new_stmt_scope():
                # Processing the final one-dimensional larger scene
                self.do_tiling_key_mode_last_dim_cut(core_index)
    
    def routing_by_tiling_key_for_310b(self, core_index):
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE4):
            with self.tik_instance.new_stmt_scope():
                # paddings are all 0
                self.do_tiling_key_mode_4(core_index)
        with self.tik_instance.elif_scope(self.tiling_key == Constant.LAST_DIM_PAD_MODE):
            with self.tik_instance.new_stmt_scope():
                # pure data_move
                self.do_tiling_key_mode_pure_datamove(core_index)
        with self.tik_instance.elif_scope(self.tiling_key == Constant.PADV3_DATA_MOVE_LAST_DIM_CUT):
            with self.tik_instance.new_stmt_scope():
                # Processing the final one-dimensional larger scene
                self.do_tiling_key_mode_last_dim_cut(core_index)
                                  
    def pad_compute(self, outer_compile_info=None):
        """
        pad_compute
        """
        self.pad_v3_d_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True}

        # add compile info
        max_line_in_ub = 16
        dtype_rate = 1
        max_output_size = self.cal_max_output_ub(max_line_in_ub)
        wr_compile_info = {
            "max_output_size": max_output_size,
            "ub_size": self.ub_number,
            "core_num": self.core_nums,
            "dtype_rate": dtype_rate,
            "mode": self.mode,
            "padding_contiguous": self.padding_contiguous,
            "total_ub_size": self.total_ub_size,
            "x_bytes_size": self.x_bytes_size,
            "soc_version": self.version_info
        }
        if outer_compile_info is not None:
            for key in outer_compile_info.keys():
                wr_compile_info[key] = outer_compile_info[key]
        tbe_context.get_context().add_compile_info("vars", wr_compile_info)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_gm_list,
                                   flowtable=[self.tiling_gm],
                                   outputs=self.output_gm_list,
                                   config=opt_config)

        return self.tik_instance


@register_operator("PadV3")
def pad_v3(x, paddings, constant_values, y, mode='constant', padding_contiguous=True, kernel_name="pad_v3"):
    """ calculating pad_v3 tensor by paddings parameters

    Parameters
    ----------
    x : dict
        shape and dtype of input
    paddings: dict
        shape and dtype of output
        For each dimension D of input, paddings[D, 0] indicates how many
        values to add
        before the contents of tensor in that dimension, and paddings[D, 1]
        indicates
        how many values to add after the contents of tensor in that dimension.
    constant_values: dict
        the value to fill the tensor
    y: dict
        shape and dtype of output
    mode:str
        the cal mode of op
    padding_contiguous: bool
        judge whether the memory is contiguous
    kernel_name : str
        cce kernel name, default value is "pad_v3"

    Returns
    -------
    None.
    """
    if mode == 'reflect':
        return reflection_pad_v3(x, paddings, constant_values, y, mode, padding_contiguous, kernel_name)
    if mode == 'edge':
        return replication_pad_v3(x, paddings, constant_values, y, mode, padding_contiguous, kernel_name)

    cur_format = x.get("format")

    # This branch is taken when the mode is constant and the format is 5HD.
    # If not, then go to the ND branch of constant mode
    if cur_format == "NC1HWC0":
        return pad_v3_5hd(x, paddings, constant_values, y, mode, padding_contiguous, kernel_name)

    src_dtype = x.get("dtype").lower()
    paddings_dtype = paddings.get("dtype").lower()
    supported_dtype = ("int8", "uint8", "bfloat16", "float16", "int16", "uint16", "float32", "int32", "uint32")
    para_check.check_dtype(src_dtype, supported_dtype, param_name="x")
    para_check.check_dtype(paddings_dtype, ("int32", "int64"), param_name="paddings")

    obj = PadV3Init(x, paddings, constant_values, y, mode, padding_contiguous, kernel_name)
    obj.init_src_dst_gm((x, paddings), pad_input_idx=0, pad_outnput_idx=0)

    return obj.pad_compute()