# Copyright 2023 Huawei Technologies Co., Ltd
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
mem_set
"""
import math
from impl.ops_legacy.util.platform_adapter import tik
from impl.ops_legacy.util.platform_adapter import tbe_platform
from impl.ops_legacy.util.platform_adapter import para_check
from impl.ops_legacy.util.platform_adapter import register_operator
from impl.ops_legacy.util.platform_adapter import tbe_context
from impl.ops_legacy.util.platform_adapter import tbe_register
from impl.ops_legacy.mem_set import mem_set as static_mem_set
from tbe.common.buildcfg import get_current_build_config


@tbe_register.register_param_generalization("MemSet")
def dynamic_mem_set_generalization(sizes, dtypes=None, values_int=None, values_float=None, generalize_config=None):
    """mem_set_generalization
    """
    if generalize_config["mode"] == "all_shape":
        size_list = [-1 for _ in sizes]
        generalization_res = [size_list, [], [], []]

        return [generalization_res]

    return None


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals,too-many-arguments
# 'pylint: disable=useless-object-inheritance
def _ceil_align(ori_num, divider):
    """
    Parameters
    ----------
    ori_num: original number
    divider: divider

    Returns
    -------
    """
    return (ori_num + divider - 1) // divider * divider


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    # max_int32
    MAX_INT32 = 2 ** 31 - 1
    # full mask for fp32
    MASK_FP32 = 64
    # max repeat time
    MAX_REPEAT_TIME = 255
    # max tiling params num
    TILING_PARAMS_NUM = 24
    CORE_NUM_INDEX = 20
    # int32 byte
    INT32_BYTE = 4
    FP16_BYTE = 2
    # block byte
    BLOCK_BYTE = int(tbe_platform.get_soc_spec("ubblock_size"))
    ZERO_FP32 = 0.0
    RESTRICT = 191
    BITS_32 = 32
    BITS_16 = 16
    BITS_8 = 8
    BITS_5 = 5
    BITS_10 = 10
    BINARY = 2
    SIGN_OF_NEGATIVE = "1"
    MAX_MASK_BYTE = 256


def _tik_get_ub_size(is_double_buffer=True):
    """
    get ub size

    Parameters
    ----------
    is_double_buffer: is_double_buffer flag

    Returns
    -------
    ub_size
    """
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    if is_double_buffer:
        return ub_size // 2
    return ub_size


# 'pylint: disable=too-many-instance-attributes,too-few-public-methods
class DynamicMemSet():
    """
    MemSet
    """

    # 'pylint: disable=too-few-public-methods,too-many-statements
    def __init__(self, size_list, dtypes, values_int, values_float):
        """
        constructor of class MemSet

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.ub_size = Constant.UB_SIZE
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.is_double_buffer = True
        self.workspace_num = len(size_list)
        self.size_list = size_list
        self.size_int64_align = 4
        self.size_int64 = 8
        self.data_type = dtypes
        self.values_int = values_int
        self.values_float = values_float
        self.int_dict = {"int8": 1, "int16": 2, "int32": 4, "uint8": 1, "uint16": 2, "uint32": 4}
        self.float_dict = {"float16": 2, "float32": 4}
        self.is_long_mode = True if not size_list else False
        self.workspace_addrs = []
        if self.workspace_num >= Constant.RESTRICT:
            self.is_long_mode = True
        self.is_dynamic = True if not size_list else False
        self.is_zero = self.check_zero(self.values_float) and self.check_zero(self.values_int)
        for _s in size_list:
            if _s < 0:
                self.is_dynamic = True
        if not size_list:
            self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.MAX_INT32,), tik.scope_gm, "tiling_gm")
            self.workspace_num = self.tik_instance.Scalar(dtype="int32", name="workspace_num")
            with self.tik_instance.new_stmt_scope():
                ub_tensor = self.tik_instance.Tensor("int32", (Constant.BLOCK_BYTE // Constant.INT32_BYTE,),
                                                     tik.scope_ubuf, "ub_tmp")
                self.tik_instance.data_move(ub_tensor, self.tiling_gm[0], 0, 1, 1, 0, 0)
                self.workspace_num.set_as(ub_tensor[0])
        else:
            if self.is_dynamic:
                gm_tiling_size_align = _ceil_align(self.workspace_num * Constant.TILING_PARAMS_NUM *
                                                   Constant.INT32_BYTE, Constant.BLOCK_BYTE)
                self.tiling_gm = self.tik_instance.Tensor("int32",
                                                          (gm_tiling_size_align // Constant.INT32_BYTE,),
                                                          tik.scope_gm, "tiling_gm")
            self.size_params_dtype = []
            self.size_params_int_value = values_int
            self.size_params_float_value = values_float
            self.size_params_float_value_fp16 = [0.0] * self.workspace_num
        self.ub_tiling_size_align = _ceil_align(Constant.TILING_PARAMS_NUM * Constant.INT32_BYTE, Constant.BLOCK_BYTE)
        self.tiling_ub = self.tik_instance.Tensor("int32", (self.ub_tiling_size_align // Constant.INT32_BYTE,),
                                                  tik.scope_ubuf, "tiling_ub")
        self.ub_size -= self.ub_tiling_size_align * 2
        self.core_num_scalar = self.tik_instance.Scalar(dtype="int32", name="core_num_scalar", init_value=self.core_num)
        if self.is_long_mode:
            self.aligned_workspace_num = _ceil_align(self.workspace_num, self.size_int64_align)
            self.list_ub_size = self.aligned_workspace_num * self.size_int64
            self.ub_size -= self.list_ub_size
            self.addr_gm_list = self.tik_instance.TensorAddrList(self.aligned_workspace_num, tik.scope_gm,
                                                                 "addr_gm_list")
            self.addr_ub_list = self.tik_instance.TensorAddrList(self.aligned_workspace_num, tik.scope_ubuf,
                                                                 "addr_ub_list")
        if size_list:
            for idx in range(0, self.workspace_num):
                int_dtype = self.data_type[idx]
                self.size_params_dtype.append(int_dtype)
                str_dtype = self.memset_dst_type_conversion(int_dtype)
                if str_dtype in self.int_dict:
                    val_32_bits = self.data_convert_to_32_bits(self.size_params_int_value[idx], str_dtype)
                    self.size_params_int_value[idx] = val_32_bits
                    self.size_params_float_value.insert(idx, 0.0)
                elif str_dtype == "float16":
                    self.size_params_int_value.insert(idx, 0)
                    self.size_params_float_value_fp16[idx] = self.size_params_float_value[idx]
                else:
                    self.size_params_int_value.insert(idx, 0)
            self.max_repeat_time = min(self.ub_size // Constant.MAX_MASK_BYTE, Constant.MAX_REPEAT_TIME)
            params_gm_shape_align = _ceil_align(self.workspace_num * Constant.INT32_BYTE, Constant.BLOCK_BYTE)
            sizes_len_align_size = params_gm_shape_align // Constant.INT32_BYTE - self.workspace_num
            align_list_int = sizes_len_align_size * [0, ]
            align_list_float = sizes_len_align_size * [0.0, ]
            self.size_params_dtype += align_list_int
            self.size_params_int_value += align_list_int
            self.size_params_float_value += align_list_float
            if not self.is_dynamic:
                self.sizes = size_list + align_list_int

            params_gm_fp16_shape_align = _ceil_align(self.workspace_num * Constant.FP16_BYTE, Constant.BLOCK_BYTE)
            sizes_len_align_size_fp16 = params_gm_fp16_shape_align // Constant.FP16_BYTE - self.workspace_num
            self.size_params_float_value_fp16 += sizes_len_align_size_fp16 * [0.0, ]
            self.workspace_addrs_params_fp16_gm = self.tik_instance.Tensor("float16",
                                                                           (params_gm_fp16_shape_align //
                                                                            Constant.FP16_BYTE,),
                                                                           tik.scope_gm, "values_fp16_gm", init_value=
                                                                           self.size_params_float_value_fp16)
            if not self.is_dynamic:
                self.workspace_addrs_params_sizes_ub = self.tik_instance.Tensor("int64",
                                                                                (params_gm_shape_align //
                                                                                 Constant.INT32_BYTE,), tik.scope_gm,
                                                                                "sizes_gm",
                                                                                init_value=self.sizes)
            self.workspace_addrs_params_dtype_gm = self.tik_instance.Tensor("int32",
                                                                            (params_gm_shape_align //
                                                                             Constant.INT32_BYTE,), tik.scope_gm,
                                                                            "dtypes_gm",
                                                                            init_value=self.size_params_dtype)
            self.workspace_addrs_params_int_gm = self.tik_instance.Tensor("int32",
                                                                          (params_gm_shape_align //
                                                                           Constant.INT32_BYTE,),
                                                                          tik.scope_gm, "values_int_gm",
                                                                          init_value=self.size_params_int_value)
            self.workspace_addrs_params_float_gm = self.tik_instance.Tensor("float32",
                                                                            (params_gm_shape_align //
                                                                             Constant.INT32_BYTE,),
                                                                            tik.scope_gm, "values_float_gm",
                                                                            init_value=self.size_params_float_value)
        self.full_mask_nums = {"float16": 128, "float32": 64, "int16": 128, "int32": 64, "uint16": 128, "uint32": 64}
        self.dtypes_map = {0: ["float32", "float32"], 1: ["float16", "float16"], 2: ["int8", "int32"],
                           3: ["int32", "int32"], 4: ["uint8", "int32"], 6: ["int16", "int32"],
                           7: ["uint16", "int32"], 8: ["uint32", "int32"]
                           }
        self.support_move_align = tbe_platform.api_check_support("tik.data_move_pad")

        # 'pylint: disable=too-few-public-methods
        class CommonInputScalar():
            """
            CommonInputScalar
            """

            def __init__(self, tik_instance):
                """
                constructor of class CommonInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.select_key = tik_instance.Scalar(
                    dtype="int64", name="select_key")
                self.need_core_num = tik_instance.Scalar(
                    dtype="int64", name="need_core_num")
                self.ele_num_full_mask_repeat_time = \
                    tik_instance.Scalar(
                        dtype="int64",
                        name="ele_num_full_mask_repeat_time")
                self.burst_len_full_mask_repeat_time = \
                    tik_instance.Scalar(
                        dtype="int64",
                        name="burst_len_full_mask_repeat_time")

        # 'pylint: disable=too-few-public-methods
        class InitInputScalar():
            """
            InitInputScalar
            """

            def __init__(self, tik_instance):
                """
                constructor of class InitInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                # front core
                self.ele_num_front_core = tik_instance.Scalar(
                    dtype="int64", name="ele_num_front_core")
                # front part full mask full repeat time front core
                self.init_times_full_mask_repeat_time_front_core = \
                    tik_instance.Scalar(
                        dtype="int64",
                        name="init_times_full_mask_repeat_time_front_core")
                self.ele_num_front_part_front_core = tik_instance.Scalar(
                    dtype="int64",
                    name="ele_num_front_part_front_core")
                # last part front
                self.burst_len_last_part_front_core = tik_instance.Scalar(
                    dtype="int64",
                    name="burst_len_last_part_front_core")
                self.repeat_time_last_part_front_core = tik_instance.Scalar(
                    dtype="int64",
                    name="repeat_time_last_part_front_core")

                # last core
                self.ele_num_last_core = tik_instance.Scalar(
                    dtype="int64", name="ele_num_last_core")
                # front part full mask full repeat time last core
                self.init_times_full_mask_repeat_time_last_core = \
                    tik_instance.Scalar(
                        dtype="int64",
                        name="init_times_full_mask_repeat_time_last_core")
                self.ele_num_front_part_last_core = tik_instance.Scalar(
                    dtype="int64",
                    name="ele_num_front_part_last_core")
                # last part last core
                self.burst_len_last_part_last_core = tik_instance.Scalar(
                    dtype="int64",
                    name="burst_len_last_part_last_core")
                self.repeat_time_last_part_last_core = tik_instance.Scalar(
                    dtype="int64",
                    name="repeat_time_last_part_last_core")
                self.core_num = tik_instance.Scalar(dtype="int64", name="core_num")
                self.gap_last_core_last_time = tik_instance.Scalar(dtype="int64", name="gap_last_core_last_time")

                # init dtype and value
                self.init_dtype_tiling = tik_instance.Scalar(dtype="int32", name="init_dtype_tiling")
                self.init_value_tiling = tik_instance.Scalar(dtype="float32", name="init_value_tiling")

        self.obj_common_input_scalar = CommonInputScalar(self.tik_instance)
        self.obj_init_input_scalar = InitInputScalar(self.tik_instance)

    @classmethod
    def memset_dst_type_conversion(self, dst_type):
        if dst_type == 0:
            dst_type = "float32"
        if dst_type == 1:
            dst_type = "float16"
        if dst_type == 2:
            dst_type = "int8"
        if dst_type == 3:
            dst_type = "int32"
        if dst_type == 4:
            dst_type = "uint8"
        if dst_type == 10:
            dst_type = "uint64"
        if dst_type == 9:
            dst_type = "int64"
        if dst_type == 8:
            dst_type = "uint32"
        if dst_type == 6:
            dst_type = "int16"
        if dst_type == 7:
            dst_type = "uint16"
        return dst_type
    
    @staticmethod
    def check_zero(values):
        for num in values:
            if not math.isclose(num, 0.0):
                return False
        return True

    def data_convert_to_32_bits(self, data, dtype):
        if dtype not in ("int32", "uint32") and dtype in self.int_dict:
            if dtype in ("int16", "uint16"):
                bin_data = bin(int(hex(data & 0xFFFF), 16)).replace("0b", "")
                const_len = Constant.BITS_16
                len_to_pad = const_len - len(bin_data)
            else:
                bin_data = bin(int(hex(data & 0xFF), 16)).replace("0b", "")
                const_len = Constant.BITS_8
                len_to_pad = const_len - len(bin_data)
            bin_data_pad = len_to_pad * "0" + bin_data
            bin_32_data = bin_data_pad * (Constant.BITS_32 // const_len)
            data = int(bin_32_data, Constant.BINARY)
        return data

    def data_move_out(self, workspace_addr, burst_len, data_size, ub_tensor):
        if self.is_dynamic and workspace_addr.dtype == "float32" and ub_tensor.dtype != "float32":
            workspace_addr = workspace_addr.reinterpret_cast_to(ub_tensor.dtype)
        if self.support_move_align and data_size is not None and not self.is_long_mode:
            ub_tensor = ub_tensor.reinterpret_cast_to(workspace_addr.dtype)
            self.tik_instance.data_move_pad(workspace_addr, ub_tensor, 1, data_size, 0, 0)
        else:
            self.tik_instance.data_move(workspace_addr, ub_tensor, 0, 1, burst_len, 0, 0)

    def init_value_set(self, data_move_params):
        workspace_addrs, idx, gm_offset, burst_len, gap, ub_tensor = data_move_params
        if self.support_move_align and gap is not None:
            data_size = burst_len * Constant.BLOCK_BYTE - gap
        else:
            data_size = None
            
        if self.is_long_mode:
            workspace_addr = self.addr_ub_list[idx].value
            self.data_move_out(workspace_addr + gm_offset, burst_len, data_size, ub_tensor)
        else:
            self.data_move_out(workspace_addrs[idx][gm_offset], burst_len, data_size, ub_tensor)

    def tiling_in_one_core_front(self, ele_num_one_core, tensor_type_mask_num, type_byte,
                                 ele_num_full_mask_full_repeat_time_input_scalar):
        scalar = ele_num_full_mask_full_repeat_time_input_scalar
        self.obj_init_input_scalar.init_times_full_mask_repeat_time_front_core.set_as(ele_num_one_core // scalar)
        self.obj_init_input_scalar.ele_num_front_part_front_core.set_as(
            self.obj_init_input_scalar.init_times_full_mask_repeat_time_front_core * scalar)

        ele_num_last_part = ele_num_one_core - self.obj_init_input_scalar.ele_num_front_part_front_core
        self.obj_init_input_scalar.burst_len_last_part_front_core.set_as(
            (ele_num_last_part * type_byte + Constant.BLOCK_BYTE - 1) // Constant.BLOCK_BYTE)

        with self.tik_instance.if_scope(ele_num_last_part % tensor_type_mask_num == 0):
            self.obj_init_input_scalar.repeat_time_last_part_front_core.set_as(ele_num_last_part //
                                                                               tensor_type_mask_num)
        with self.tik_instance.else_scope():
            self.obj_init_input_scalar.repeat_time_last_part_front_core.set_as(ele_num_last_part //
                                                                               tensor_type_mask_num + 1)

    def tiling_in_one_core_last(self, ele_num_one_core, tensor_type_mask_num, type_byte,
                                ele_num_full_mask_full_repeat_time_input_scalar):
        scalar = ele_num_full_mask_full_repeat_time_input_scalar
        self.obj_init_input_scalar.init_times_full_mask_repeat_time_last_core.set_as(ele_num_one_core // scalar)
        self.obj_init_input_scalar.ele_num_front_part_last_core.set_as(
            self.obj_init_input_scalar.init_times_full_mask_repeat_time_last_core * scalar)

        ele_num_last_part = ele_num_one_core - self.obj_init_input_scalar.ele_num_front_part_last_core
        self.obj_init_input_scalar.burst_len_last_part_last_core.set_as(
            (ele_num_last_part * type_byte + Constant.BLOCK_BYTE - 1) // Constant.BLOCK_BYTE)

        with self.tik_instance.if_scope(ele_num_last_part % tensor_type_mask_num == 0):
            self.obj_init_input_scalar.repeat_time_last_part_last_core.set_as(ele_num_last_part // tensor_type_mask_num)
        with self.tik_instance.else_scope():
            self.obj_init_input_scalar.repeat_time_last_part_last_core.set_as(ele_num_last_part //
                                                                              tensor_type_mask_num + 1)

    def get_tiling_data_static(self, sizes, idx, dtype):
        tensor_size_ub = self.tik_instance.Tensor("int64", (Constant.BLOCK_BYTE // Constant.INT32_BYTE,),
                                                  tik.scope_ubuf, "ub_tensor_size")
        self.tik_instance.data_move(tensor_size_ub, self.workspace_addrs_params_sizes_ub[idx], 0, 1, 1, 0, 0)
        ori_tensor_size = self.tik_instance.Scalar(dtype="int64", name="ori_tensor_size")
        ori_tensor_size.set_as(tensor_size_ub[0])
        tensor_type_mask_num = 64
        if dtype in self.int_dict:
            type_byte = self.int_dict.get(dtype)
        else:
            type_byte = self.float_dict.get(dtype)
            if dtype == "float16":
                tensor_type_mask_num = 128
        
        tensor_size = _ceil_align(ori_tensor_size, Constant.BLOCK_BYTE)
        self.obj_init_input_scalar.gap_last_core_last_time = tensor_size - ori_tensor_size
        ele_num = tensor_size // type_byte
        with self.tik_instance.if_scope(tensor_size > self.core_num * Constant.MAX_MASK_BYTE):
            self.obj_common_input_scalar.need_core_num.set_as(self.core_num)
        with self.tik_instance.else_scope():
            self.obj_common_input_scalar.need_core_num.set_as(1)
        self.obj_common_input_scalar.ele_num_full_mask_repeat_time.set_as(tensor_type_mask_num * self.max_repeat_time)
        self.obj_common_input_scalar.burst_len_full_mask_repeat_time.set_as(
            self.obj_common_input_scalar.ele_num_full_mask_repeat_time * type_byte // Constant.BLOCK_BYTE)
        with self.tik_instance.if_scope(self.obj_common_input_scalar.need_core_num == 1):
            # use one core
            self.obj_init_input_scalar.ele_num_front_core.set_as(ele_num)
            self.tiling_in_one_core_front(self.obj_init_input_scalar.ele_num_front_core, tensor_type_mask_num,
                                          type_byte, self.obj_common_input_scalar.ele_num_full_mask_repeat_time)
            self.obj_init_input_scalar.ele_num_last_core.set_as(self.obj_init_input_scalar.ele_num_front_core)
            self.tiling_in_one_core_last(self.obj_init_input_scalar.ele_num_last_core, tensor_type_mask_num, type_byte,
                                         self.obj_common_input_scalar.ele_num_full_mask_repeat_time)
        with self.tik_instance.else_scope():
            # use all core
            self.obj_init_input_scalar.ele_num_front_core.set_as(tensor_size //
                                                                 self.obj_common_input_scalar.need_core_num)
            self.obj_init_input_scalar.ele_num_front_core.set_as(_ceil_align(
                self.obj_init_input_scalar.ele_num_front_core, Constant.BLOCK_BYTE))
            self.obj_init_input_scalar.ele_num_front_core.set_as(self.obj_init_input_scalar.ele_num_front_core //
                                                                 type_byte)
            self.obj_common_input_scalar.need_core_num.set_as((ele_num + self.obj_init_input_scalar.ele_num_front_core
                                                               - 1) // self.obj_init_input_scalar.ele_num_front_core)
            self.tiling_in_one_core_front(self.obj_init_input_scalar.ele_num_front_core, tensor_type_mask_num,
                                          type_byte, self.obj_common_input_scalar.ele_num_full_mask_repeat_time)
            self.obj_init_input_scalar.ele_num_last_core.set_as(ele_num - self.obj_init_input_scalar.ele_num_front_core
                                                                * (self.obj_common_input_scalar.need_core_num - 1))
            self.tiling_in_one_core_last(self.obj_init_input_scalar.ele_num_last_core, tensor_type_mask_num, type_byte,
                                         self.obj_common_input_scalar.ele_num_full_mask_repeat_time)
        self.obj_init_input_scalar.core_num.set_as(self.core_num)

    def get_tiling_data(self, tiling_ub):
        input_scalar_index = 0
        input_scalar_index_int64 = 0
        tiling_ub_int64 = tiling_ub.reinterpret_cast_to("int64")
        self.obj_common_input_scalar.select_key.set_as(tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.obj_common_input_scalar.need_core_num.set_as(tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.obj_common_input_scalar.ele_num_full_mask_repeat_time.set_as(tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.obj_common_input_scalar.burst_len_full_mask_repeat_time.set_as(tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        input_scalar_index_int64 = input_scalar_index_int64 + 2
        self.obj_init_input_scalar.ele_num_front_core.set_as(tiling_ub_int64[input_scalar_index_int64])
        input_scalar_index = input_scalar_index + 2
        input_scalar_index_int64 = input_scalar_index_int64 + 1
        self.obj_init_input_scalar.init_times_full_mask_repeat_time_front_core.set_as(
            tiling_ub_int64[input_scalar_index_int64])
        input_scalar_index = input_scalar_index + 2
        input_scalar_index_int64 = input_scalar_index_int64 + 1
        self.obj_init_input_scalar.ele_num_front_part_front_core.set_as(tiling_ub_int64[input_scalar_index_int64])
        input_scalar_index = input_scalar_index + 2
        input_scalar_index_int64 = input_scalar_index_int64 + 1
        self.obj_init_input_scalar.burst_len_last_part_front_core.set_as(tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.obj_init_input_scalar.repeat_time_last_part_front_core.set_as(tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        input_scalar_index_int64 = input_scalar_index_int64 + 1
        self.obj_init_input_scalar.ele_num_last_core.set_as(tiling_ub_int64[input_scalar_index_int64])
        input_scalar_index = input_scalar_index + 2
        input_scalar_index_int64 = input_scalar_index_int64 + 1
        self.obj_init_input_scalar.init_times_full_mask_repeat_time_last_core.set_as(
            tiling_ub_int64[input_scalar_index_int64])
        input_scalar_index = input_scalar_index + 2
        input_scalar_index_int64 = input_scalar_index_int64 + 1
        self.obj_init_input_scalar.ele_num_front_part_last_core.set_as(tiling_ub_int64[input_scalar_index_int64])
        input_scalar_index = input_scalar_index + 2
        self.obj_init_input_scalar.burst_len_last_part_last_core.set_as(tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.obj_init_input_scalar.repeat_time_last_part_last_core.set_as(tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.obj_init_input_scalar.core_num.set_as(tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.obj_init_input_scalar.gap_last_core_last_time.set_as(tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.obj_init_input_scalar.init_dtype_tiling.set_as(tiling_ub[input_scalar_index])
        input_scalar_index = input_scalar_index + 1
        self.obj_init_input_scalar.init_value_tiling.set_as(tiling_ub[input_scalar_index])

    def compute_value_set_ele_size(self, workspace_addrs, core_index, idx, dst_type, input_scalar_index):
        init_value = 0.0
        if not self.size_list:
            init_dtype = "float32"
        else:
            init_dtype = dst_type
            if not self.is_zero:
                ub_tensor = self.tik_instance.Tensor(dst_type, (Constant.BLOCK_BYTE // Constant.INT32_BYTE,),
                                                    tik.scope_ubuf, "ub_tensor_float")
                if dst_type in self.int_dict:
                    self.tik_instance.data_move(ub_tensor, self.workspace_addrs_params_int_gm[idx], 0, 1, 1, 0, 0)
                elif dst_type == "float16":
                    self.tik_instance.data_move(ub_tensor, self.workspace_addrs_params_fp16_gm[idx], 0, 1, 1, 0, 0)
                else:
                    self.tik_instance.data_move(ub_tensor, self.workspace_addrs_params_float_gm[idx], 0, 1, 1, 0, 0)
                value_scalar = self.tik_instance.Scalar(dtype=dst_type, name="value_to_set")
                value_scalar.set_as(ub_tensor[0])
                init_value = value_scalar
            else:
                if dst_type in self.int_dict:
                    init_value = 0
        # mov tiling data from gm to ub, and set_as scalar
        if self.is_dynamic:
            init_value_scalar = self.tik_instance.Scalar(dtype=init_dtype, name="init_value_scalar")
            init_value_scalar.set_as(self.obj_init_input_scalar.init_value_tiling)
            init_value = init_value_scalar
        else:
            self.get_tiling_data_static(self.workspace_addrs_params_sizes_ub, idx, dst_type)

        gm_offset = self.tik_instance.Scalar(dtype="int64", name="offset")
        mask = self.full_mask_nums.get(init_dtype)
        ub_for_dup = self.tik_instance.Tensor(init_dtype, (mask * Constant.MAX_REPEAT_TIME,),
                                              tik.scope_ubuf, "ub_for_dup")
        with self.tik_instance.if_scope(core_index < self.obj_common_input_scalar.need_core_num - 1):
            # front core
            with self.tik_instance.for_range(
                    0, self.obj_init_input_scalar.init_times_full_mask_repeat_time_front_core) \
                    as init_index:
                gm_offset.set_as(core_index * self.obj_init_input_scalar.ele_num_front_core + init_index *
                                 self.obj_common_input_scalar.ele_num_full_mask_repeat_time)
                self.tik_instance.vector_dup(mask, ub_for_dup, init_value, Constant.MAX_REPEAT_TIME, 1, 8)
                self.init_value_set([workspace_addrs, idx, gm_offset,
                                    self.obj_common_input_scalar.burst_len_full_mask_repeat_time, None, ub_for_dup])

            # last part front core
            with self.tik_instance.if_scope(
                    self.obj_init_input_scalar.init_times_full_mask_repeat_time_front_core == 0):
                self.tik_instance.vector_dup(mask, ub_for_dup, init_value,
                                             self.obj_init_input_scalar.repeat_time_last_part_front_core, 1, 8)
            gm_offset.set_as(core_index * self.obj_init_input_scalar.ele_num_front_core +
                             self.obj_init_input_scalar.ele_num_front_part_front_core)
            with self.tik_instance.if_scope(
                    self.obj_init_input_scalar.burst_len_last_part_front_core > 0):
                self.init_value_set([workspace_addrs, idx, gm_offset,
                                    self.obj_init_input_scalar.burst_len_last_part_front_core,
                                    None, ub_for_dup])
        with self.tik_instance.if_scope(core_index == self.obj_common_input_scalar.need_core_num - 1):
            # last core
            with self.tik_instance.for_range(
                    0, self.obj_init_input_scalar.init_times_full_mask_repeat_time_last_core) as init_index:
                gm_offset.set_as(core_index * self.obj_init_input_scalar.ele_num_front_core
                                 + init_index * self.obj_common_input_scalar.ele_num_full_mask_repeat_time)
                # front part last core full mask full repeat time
                self.tik_instance.vector_dup(mask, ub_for_dup, init_value, Constant.MAX_REPEAT_TIME, 1, 8)
                self.init_value_set([workspace_addrs, idx, gm_offset,
                                    self.obj_common_input_scalar.burst_len_full_mask_repeat_time, None, ub_for_dup])
            # last part last core
            with self.tik_instance.if_scope(
                    self.obj_init_input_scalar.init_times_full_mask_repeat_time_last_core == 0):
                self.tik_instance.vector_dup(mask, ub_for_dup, init_value,
                                             self.obj_init_input_scalar.repeat_time_last_part_last_core, 1, 8)
            gm_offset.set_as(core_index * self.obj_init_input_scalar.ele_num_front_core +
                             self.obj_init_input_scalar.ele_num_front_part_last_core)
            with self.tik_instance.if_scope(
                    self.obj_init_input_scalar.burst_len_last_part_last_core > 0):
                self.init_value_set([workspace_addrs, idx, gm_offset,
                                    self.obj_init_input_scalar.burst_len_last_part_last_core,
                                    self.obj_init_input_scalar.gap_last_core_last_time, ub_for_dup])

    def compute_value_set_ele_dtype(self, dtype_scalar, dtype_id, workspace_addrs, core_index, idx, input_scalar_index):
        with self.tik_instance.if_scope(dtype_scalar == dtype_id):
            dst_type = self.dtypes_map.get(dtype_id)[0]
            if dst_type in self.int_dict:
                dst_type = "int32"
            self.compute_value_set_ele_size(workspace_addrs, core_index, idx, dst_type,
                                            input_scalar_index)

    def compute_value_set(self, workspace_addrs, core_index, idx):
        input_scalar_index = 0 + idx * Constant.TILING_PARAMS_NUM
        if self.is_dynamic:
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm[input_scalar_index], 0, 1,
                                        self.ub_tiling_size_align // Constant.BLOCK_BYTE, 0, 0)
            self.get_tiling_data(self.tiling_ub)

        if not self.size_list:
            self.compute_value_set_ele_size(workspace_addrs, core_index, idx, "float32", input_scalar_index)
        else:
            with self.tik_instance.new_stmt_scope():
                ub_tensor_dtype = self.tik_instance.Tensor("int32", (Constant.BLOCK_BYTE // Constant.INT32_BYTE,),
                                                           tik.scope_ubuf, "ub_tensor_dtype")
                self.tik_instance.data_move(ub_tensor_dtype, self.workspace_addrs_params_dtype_gm[idx], 0, 1, 1, 0, 0)
                dtype_scalar = self.tik_instance.Scalar(dtype="int32", name="dtype_to_set")
                dtype_scalar.set_as(ub_tensor_dtype[0])
                if self.is_dynamic:
                    dtype_scalar.set_as(self.obj_init_input_scalar.init_dtype_tiling)
                for dtype_id in self.dtypes_map:
                    self.compute_value_set_ele_dtype(dtype_scalar, dtype_id, workspace_addrs, core_index, idx,
                                                     input_scalar_index)

    # 'pylint: disable=unused-argument
    def addr_clean(self, workspace_addrs):
        """
        addr_clean
        :param workspace_addrs:
        :return:
        """
        core_num = self.core_num
        if self.is_dynamic:
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm[0], 0, 1,
                                        self.ub_tiling_size_align // Constant.BLOCK_BYTE, 0, 0)
            self.core_num_scalar.set_as(self.tiling_ub[Constant.CORE_NUM_INDEX])
            core_num = self.core_num_scalar
        with self.tik_instance.for_range(0, core_num, block_num=core_num) as core_index:
            if self.is_long_mode:
                with self.tik_instance.for_range(0, self.workspace_num) as idx:
                    self.compute_value_set(workspace_addrs, core_index, idx)
            else:
                for idx in range(0, self.workspace_num):
                    self.compute_value_set(workspace_addrs, core_index, idx)

    @staticmethod
    def get_debug_config(is_dynamic):
        """
        get_debug_config
        :param is_dynamic: enable_const_fold if dynamic mode
        """
        op_debug_config = get_current_build_config("op_debug_config")
        op_debug_config = op_debug_config if op_debug_config else ""
        # memset do not support oom check
        op_debug_config = op_debug_config.replace("oom,", "").replace("oom", "")
        if is_dynamic:
            opt_config = {"out_of_bound_sync_check": True,
                          "enable_const_fold": True,
                          "op_debug_config": op_debug_config}
        else:
            opt_config = {"op_debug_config": op_debug_config}
        return opt_config

    def build(self, kernel_name):
        """
        tik_instance_fun
        """
        opt_config = self.get_debug_config(self.is_dynamic)
        if self.is_long_mode:
            self.tik_instance.data_move(self.addr_ub_list, self.addr_gm_list, 0,
                                        1, self.aligned_workspace_num // self.size_int64_align, 0, 0)
            tbe_context.get_context().add_build_json_result("wspMode", True)
            self.addr_clean(self.workspace_addrs)
            if self.is_dynamic:
                self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                           inputs=self.addr_gm_list,
                                           outputs=[],
                                           flowtable=[self.tiling_gm],
                                           config=opt_config)
            else:
                self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                           inputs=self.addr_gm_list,
                                           outputs=(), enable_l2=False,
                                           config=opt_config)
        else:
            for idx in range(self.workspace_num):
                dtype_to_set = self.memset_dst_type_conversion(self.data_type[idx])
                addr_gm = self.tik_instance.Tensor(dtype_to_set, (Constant.MAX_INT32,),
                                                   tik.scope_gm, "".join(["gm_tensor", str(idx)]))
                self.workspace_addrs.append(addr_gm)
            self.addr_clean(self.workspace_addrs)
            if self.is_dynamic:
                self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                           inputs=self.workspace_addrs,
                                           outputs=[],
                                           flowtable=[self.tiling_gm],
                                           config=opt_config)
            else:
                self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                           inputs=self.workspace_addrs,
                                           outputs=(), enable_l2=False,
                                           config=opt_config)
        return self.tik_instance


# 'pylint: disable=unused-argument,dangerous-default-value
@register_operator("MemSet")
@para_check.check_op_params(para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_FLOAT,
                            para_check.KERNEL_NAME)
def mem_set(sizes, dtypes=[], values_int=[], values_float=[], kernel_name="MemSet"):
    """
    clean memory of workspace list
    Parameters
    ----------
    sizes :  list
        sizes of workspaces
    dtypes : list
        dtypes of initial values
    values_int : list
        integer values to be set
    values_float : list
        float values to be set
    kernel_name : str
        kernel name, default value is "MemSet"

    Returns
    -------
    compile info
    """
    is_dynamic = True if not sizes else False
    is_clean_mode = True if not sizes else False
    if sum(values_float) + sum(values_int) == 0:
        is_clean_mode = True
    for _s in sizes:
        if _s < 0:
            is_dynamic = True
    if not is_dynamic and is_clean_mode and len(sizes) < Constant.RESTRICT:
        static_mem_set(sizes, dtypes, values_int, values_float, kernel_name)
    else:
        if not dtypes:
            dtypes = [0, ] if not sizes else [0, ] * len(sizes)
            if not values_float:
                values_float = [0.0, ] * len(sizes)

        integers = [] if not values_int else values_int
        floats = [] if not values_float else values_float

        dtypes_full_mask = []
        byte_of_dtypes = []

        obj_dynamic_mem_set = DynamicMemSet(list(sizes), list(dtypes), list(integers), list(floats))
        # add compile info
        for _type in dtypes:
            dst_type = obj_dynamic_mem_set.memset_dst_type_conversion(_type)
            if dst_type in ("int8", "uint8"):
                dst_type = "int32"
            dtypes_full_mask.append(obj_dynamic_mem_set.full_mask_nums.get(dst_type))
            if dst_type in obj_dynamic_mem_set.int_dict:
                byte_of_dtypes.append(obj_dynamic_mem_set.int_dict.get(dst_type))
            else:
                byte_of_dtypes.append(obj_dynamic_mem_set.float_dict.get(dst_type))

        workspace_num = -1 if not sizes else obj_dynamic_mem_set.workspace_num
        max_repeat_time = -1 if not sizes else obj_dynamic_mem_set.max_repeat_time
        ub_size = Constant.UB_SIZE if not sizes else obj_dynamic_mem_set.ub_size
        tbe_context.get_context().add_compile_info("vars",
                                                   {"ub_size": ub_size,
                                                    "core_num": obj_dynamic_mem_set.core_num,
                                                    "workspace_num": workspace_num,
                                                    "max_repeat_time": max_repeat_time,
                                                    "mask_nums": dtypes_full_mask,
                                                    "byte_list": byte_of_dtypes,
                                                    "_workspace_index_list": sizes,
                                                    "is_dynamic": is_dynamic})
        # build cce
        obj_dynamic_mem_set.build(kernel_name)

