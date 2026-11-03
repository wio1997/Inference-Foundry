# Copyright 2020 Huawei Technologies Co., Ltd
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
from impl.ops_legacy.util.platform_adapter import tik
from impl.ops_legacy.util.platform_adapter import tbe_platform
from impl.ops_legacy.util.platform_adapter import para_check
from impl.ops_legacy.util.platform_adapter import error_manager_vector
from impl.ops_legacy.util.platform_adapter import tbe_context


# 'pylint: disable=too-few-public-methods,old-style-class,no-init
class Constant:
    """
    the class for constant.
    """
    RESTRICT = 191
    UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    MEMORY_CLEAR_BLOCK_SIZE = int(tbe_platform.get_soc_spec("ubblock_size"))
    MEMORY_CLEAR_VECTOR_SIZE = 8 * MEMORY_CLEAR_BLOCK_SIZE
    MEMORY_CLEAR_KB_NUM = 48
    MEMORY_CLEAR_ONE_KB = 1024
    MEMORY_CLEAR_UB_ZERO_BUFF = MEMORY_CLEAR_KB_NUM * MEMORY_CLEAR_ONE_KB
    if UB_SIZE < MEMORY_CLEAR_UB_ZERO_BUFF:
        MEMORY_CLEAR_UB_ZERO_BUFF = UB_SIZE


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


def _ceil_divide(ori_num, divider):
    """
    Parameters
    ----------
    ori_num
    divider

    Returns
    -------
    """
    return (ori_num + divider - 1) // divider


def _memset_dst_type_conversion(dst_type):
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
    if not isinstance(dst_type, str):
        dst_type = "error"
    return dst_type


# 'pylint: disable=too-many-instance-attributes
class MemSet(object):
    """
    MemSet Class implementing the function of operator MemSet
    """

    def __init__(self, sizes, dtypes, values_int, values_float):
        """
        :param size_list: size list of workspaces
        :param addr_list: address list of workspaces
        """
        self.int_dict = {"int8": 1, "int16": 2, "int32": 4, "int64": 8, "uint8": 1, "uint16": 2,
                         "uint32": 4, "uint64": 8}
        self.float_dict = {"float16": 2, "float32": 4}
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.workspace_sizes = sizes
        self.data_type = len(sizes) * ["float16", ]
        self.values_int = values_int
        self.values_float = len(sizes) * [0.0, ]
        self.workspace_addrs = []
        self.workspace_addrs_params = []
        self.workspace_num = len(sizes)
        self.total_size = sum(sizes)
        self.support_move_align = tbe_platform.api_check_support("tik.data_move_pad")

        self.is_long = False
        self.size_int64_align = 4
        self.size_int64 = 8

        self.memory_clear_ub_zero_buff = Constant.MEMORY_CLEAR_UB_ZERO_BUFF
        if self.workspace_num >= Constant.RESTRICT:
            self.aligned_workspace_num = _ceil_align(self.workspace_num, self.size_int64_align)
            self.list_ub_size = self.aligned_workspace_num * self.size_int64
            self.memory_clear_ub_zero_buff -= self.list_ub_size
        self.total_blocks = self.total_size // Constant.MEMORY_CLEAR_BLOCK_SIZE
        self.each_core_blocks = _ceil_divide(self.total_blocks, self.core_num)
        self.each_core_size = self.each_core_blocks * Constant.MEMORY_CLEAR_BLOCK_SIZE
        self.each_core_size_aligned = _ceil_align(self.each_core_size, self.memory_clear_ub_zero_buff)


    def gen_value_ub(self, ub_size, str_dtype, value):
        """
        generate an ub tensor of which size is ub_size and set it to be zero

        Parameters
        ----------
        ub_size: size of ub tensor in byte
        str_dtype: dtype

        Returns:
        value_ub: ub value tensor
        """

        if str_dtype in self.int_dict:
            div = self.int_dict.get(str_dtype)
        else:
            div = self.float_dict.get(str_dtype)

        repeat_times = _ceil_divide(ub_size, Constant.MEMORY_CLEAR_VECTOR_SIZE)
        common_mask = Constant.MEMORY_CLEAR_VECTOR_SIZE // div
        value_ub = self.tik_instance.Tensor(str_dtype, (common_mask * repeat_times,), tik.scope_ubuf, "value_ub")
        self.tik_instance.vector_dup(common_mask, value_ub, value, repeat_times, 1, 8)
        return value_ub

    def data_move_out(self, workspace_addr, burst_len, data_size, ub_tensor):
        if self.support_move_align and data_size:
            self.tik_instance.data_move_pad(workspace_addr, ub_tensor, 1, data_size, 0, 0)
        else:
            self.tik_instance.data_move(workspace_addr, ub_tensor, 0, 1, burst_len, 0, 0)

    def set_workspace_single_core(self, data_size, data_addr, value_ub, addr_index):
        """
        clean the memory of a workspace

        Parameters
        ----------
        data_size: memory size of the workspace in byte
        data_addr: memory address of workspace

        Returns
        -------
        None
        """
        if self.data_type[addr_index] in self.int_dict:
            div = self.int_dict.get(self.data_type[addr_index])
        else:
            div = self.float_dict.get(self.data_type[addr_index])

        loop_num = _ceil_divide(data_size, self.memory_clear_ub_zero_buff)
        loop_offset = self.memory_clear_ub_zero_buff // div
        remain_load_size = data_size % self.memory_clear_ub_zero_buff
        burst_len = _ceil_divide(self.memory_clear_ub_zero_buff, Constant.MEMORY_CLEAR_BLOCK_SIZE)
        if remain_load_size > 0:
            burst_len_last = _ceil_divide(remain_load_size, Constant.MEMORY_CLEAR_BLOCK_SIZE)
            with self.tik_instance.for_range(0, loop_num) as idx:
                with self.tik_instance.if_scope(idx + 1 == loop_num):
                    with self.tik_instance.if_scope(idx == 0):
                        self.data_move_out(data_addr, burst_len_last, remain_load_size, value_ub)
                    with self.tik_instance.else_scope():
                        self.data_move_out(data_addr[idx * loop_offset], burst_len_last, remain_load_size, value_ub)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(idx == 0):
                        self.tik_instance.data_move(data_addr, value_ub, 0, 1, burst_len, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(data_addr[idx * loop_offset], value_ub, 0, 1, burst_len, 0, 0)
        else:
            with self.tik_instance.for_range(0, loop_num) as idx:
                self.data_move_out(data_addr[idx * loop_offset], burst_len, self.memory_clear_ub_zero_buff, value_ub)

    def set_workspace_single_core_long(self, data_size, data_addr, value_ub, addr_index):
        """
        clean the memory of a workspace

        Parameters
        ----------
        data_size: memory size of the workspace in byte
        data_addr: memory address of workspace

        Returns
        -------
        None
        """
        if self.data_type[addr_index] in self.int_dict:
            div = self.int_dict.get(self.data_type[addr_index])
        else:
            div = self.float_dict.get(self.data_type[addr_index])

        loop_num = _ceil_divide(data_size, self.memory_clear_ub_zero_buff)
        loop_offset = self.memory_clear_ub_zero_buff // div
        remain_load_size = data_size % self.memory_clear_ub_zero_buff
        burst_len = _ceil_divide(self.memory_clear_ub_zero_buff, Constant.MEMORY_CLEAR_BLOCK_SIZE)
        if remain_load_size > 0:
            burst_len_last = _ceil_divide(remain_load_size, Constant.MEMORY_CLEAR_BLOCK_SIZE)
            with self.tik_instance.for_range(0, loop_num) as idx:
                with self.tik_instance.if_scope(idx + 1 == loop_num):
                    with self.tik_instance.if_scope(idx == 0):
                        self.data_move_out(data_addr, burst_len_last, remain_load_size, value_ub)
                    with self.tik_instance.else_scope():
                        self.data_move_out(data_addr + idx * loop_offset, burst_len_last, remain_load_size, value_ub)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(idx == 0):
                        self.tik_instance.data_move(data_addr, value_ub, 0, 1, burst_len, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(data_addr + idx * loop_offset, value_ub, 0, 1, burst_len, 0, 0)
        else:
            with self.tik_instance.for_range(0, loop_num) as idx:
                self.data_move_out(data_addr + idx * loop_offset, burst_len, self.memory_clear_ub_zero_buff, value_ub)

    def workspaces_sorted_by_core_size(self, data_size_list):
        """
        distribute the workspaces to all cores

        Parameters
        ----------
        data_size_list: list of data size

        Returns
        -------
        core_segs_list: workspace segments list of all cores
        """

        # list of workspace segment, cell format of which is [data_addr_index, offset, seg_len]
        workspace_seg_list = []
        for ix, ds in enumerate(data_size_list):
            num = _ceil_divide(ds, self.each_core_size_aligned)
            rear_size = ds % self.each_core_size_aligned
            if num == 1:
                workspace_seg_list.append([ix, 0, ds])
                continue
            for sn in range(num):
                # data_addr offset in count of fp16 or fp32
                if self.data_type[ix] in self.int_dict:
                    div = self.int_dict.get(self.data_type[ix])
                else:
                    div = self.float_dict.get(self.data_type[ix])
                offset = sn * self.each_core_size_aligned // div
                seg_len = self.each_core_size_aligned
                if rear_size > 0 and sn == num - 1:
                    seg_len = rear_size
                workspace_seg_list.append([ix, offset, seg_len])

        # sort the workspace segs by length in descending order
        def take_third(elem_list):
            return elem_list[2]

        workspace_seg_list.sort(key=take_third, reverse=True)
        return workspace_seg_list

    def workspaces_distribute_to_cores(self, workspace_seg_list):
        core_segs_list = []
        for cidx in range(self.core_num):
            if not workspace_seg_list:
                break

            # if this is the last core, all the rest segs consign to this core
            if cidx == self.core_num - 1:
                core_segs_list.append(workspace_seg_list)
                break

            # for each core, construct a list segs, consign the first seg of seglist to it
            # until the total length of segs is greater than each_core_size_aligned
            segs = []
            segs.append(workspace_seg_list[0])
            workspace_seg_list.pop(0)
            core_seg_len = segs[-1][2]
            for ix in range(len(workspace_seg_list) - 1, -1, -1):
                if core_seg_len + workspace_seg_list[ix][2] <= self.each_core_size_aligned:
                    core_seg_len = core_seg_len + workspace_seg_list[ix][2]
                    segs.append(workspace_seg_list[ix])
                    workspace_seg_list.pop(ix)
                else:
                    break

            core_segs_list.append(segs)

        return core_segs_list

    def workspaces_set_multi_core_integrally(self, data_size_list, data_addr_list, workspace_addrs_params):
        """
        there is a restriction that in one operator we can apply core loop only one time
        it means that if we want to process more than one workspace
        we should take the core loop as the outer one
        ATTENSION:
        in this function, each workspace is cleaned in one core as possible

        Parameters
        ----------
        data_size_list: list of data size
        data_addr_list: list of address size

        Returns
        -------
        tik_instance
        """
        workspace_seg_list = self.workspaces_sorted_by_core_size(data_size_list)
        core_segs = self.workspaces_distribute_to_cores(workspace_seg_list)
        core_used = len(core_segs)

        with self.tik_instance.for_range(0, core_used, block_num=core_used) as block_idx:
            values_set_ub = self.gen_value_ub(self.memory_clear_ub_zero_buff, "float16", 0)
            for core_index in range(core_used):
                with self.tik_instance.if_scope(core_index == block_idx):
                    current_segs = core_segs[core_index]
                    for addr_index, offset, data_size in current_segs:
                        data_addr = data_addr_list[addr_index]
                        self.set_workspace_single_core(data_size, data_addr[offset], values_set_ub, addr_index)

    def workspaces_set_multi_core_integrally_long(self, data_size_list, addr_ub_list, data_addr_list):
        """
        there is a restriction that in one operator we can apply core loop only one time
        it means that if we want to process more than one workspace
        we should take the core loop as the outer one
        ATTENSION:
        in this function, each workspace is cleaned in one core as possible

        Parameters
        ----------
        data_size_list: list of data size
        data_addr_list: list of address size

        Returns
        -------
        tik_instance
        """
        
        workspace_seg_list_zero = self.workspaces_sorted_by_core_size(data_size_list)
        core_segs = self.workspaces_distribute_to_cores(workspace_seg_list_zero)
        core_used = len(core_segs)

        with self.tik_instance.for_range(0, core_used, block_num=core_used) as block_idx:
            values_set_ub = self.gen_value_ub(self.memory_clear_ub_zero_buff, "float16", 0)
            for core_index in range(core_used):
                with self.tik_instance.if_scope(core_index == block_idx):
                    current_segs = core_segs[core_index]
                    for addr_index, offset, data_size in current_segs:
                        data_addr = addr_ub_list[addr_index].value
                        self.set_workspace_single_core_long(data_size, data_addr + offset, values_set_ub,
                                                            addr_index)

    def build(self, kernel_name):
        """
        tik_instance_fun
        """
        if self.workspace_num >= Constant.RESTRICT:
            self.is_long = True
            tbe_context.get_context().add_build_json_result("wspMode", True)
            int_idx = 0
            float_idx = 0
            for idx in range(0, self.workspace_num):
                m_dtype = self.data_type[idx]
                if m_dtype in self.int_dict:
                    m_value = self.values_int[int_idx]
                    int_idx += 1
                elif m_dtype in self.float_dict:
                    m_value = self.values_float[float_idx]
                    float_idx += 1
                self.workspace_addrs_params.append([m_value, m_dtype])
            addr_gm_list = self.tik_instance.TensorAddrList(self.aligned_workspace_num, tik.scope_gm, "addr_gm_list")
            addr_ub_list = self.tik_instance.TensorAddrList(self.aligned_workspace_num, tik.scope_ubuf, "addr_ub_list")
            self.tik_instance.data_move(addr_ub_list, addr_gm_list, 0,
                                        1, self.aligned_workspace_num // self.size_int64_align, 0, 0)
            self.workspaces_set_multi_core_integrally_long(self.workspace_sizes, addr_ub_list,
                                                           self.workspace_addrs_params)
            self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                       inputs=addr_gm_list,
                                       outputs=(), enable_l2=False)
        else:
            int_idx = 0
            float_idx = 0
            for idx in range(0, self.workspace_num):
                m_dtype = self.data_type[idx]
                if m_dtype in self.int_dict:
                    m_value = self.values_int[int_idx]
                    int_idx += 1
                    addr_gm = self.tik_instance.Tensor(m_dtype,
                                                      (self.workspace_sizes[idx] / self.int_dict.get(m_dtype),),
                                                       tik.scope_gm, "".join(["gm", str(idx)]))
                elif m_dtype in self.float_dict:
                    m_value = self.values_float[float_idx]
                    float_idx += 1
                    addr_gm = self.tik_instance.Tensor(m_dtype,
                                                      (self.workspace_sizes[idx] / self.float_dict.get(m_dtype),),
                                                       tik.scope_gm, "".join(["gm", str(idx)]))
                self.workspace_addrs.append(addr_gm)
                self.workspace_addrs_params.append([m_value, m_dtype])
            # set workspaces using multi-core, each workspace is processed integrally as possible
            self.workspaces_set_multi_core_integrally(self.workspace_sizes, self.workspace_addrs,
                                                      self.workspace_addrs_params)
            self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                       inputs=self.workspace_addrs,
                                       outputs=(), enable_l2=False)
        return self.tik_instance


# 'pylint: disable=dangerous-default-value
@para_check.check_op_params(para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_FLOAT,
                            para_check.KERNEL_NAME)
def mem_set(sizes, dtypes=[], values_int=[], values_float=[], kernel_name="mem_set"):
    """
    clean memory of workspace list

    Parameters
    ----------
    sizes :  list
        sizes of workspaces
    dtypes : list
        dtypes of init values
    kernel_name : str
        kernel name, default value is "mem_set"

    Returns
    -------
    tik_instance
    """
    if not dtypes:
        dtypes = [1, ] * len(sizes)
        if not values_float:
            values_float = [0.0, ] * len(sizes)
    support_move_align = tbe_platform.api_check_support("tik.data_move_pad")
    for msize in sizes:
        if msize <= 0:
            expected_value = "greater than 0"
            real_value = "less than or equal to 0"
            error_manager_vector.raise_err_input_value_invalid("mem_set", "sizes of workspaces",
                                                               expected_value, real_value)
        if not support_move_align and msize % Constant.MEMORY_CLEAR_BLOCK_SIZE != 0:
            expected_value = "can be divided by 32"
            real_value = "can not be divided by 32"
            error_manager_vector.raise_err_input_value_invalid("mem_set", "sizes of workspaces",
                                                               expected_value, real_value)
    _dtypes = []
    for dst_type in dtypes:
        _type = _memset_dst_type_conversion(dst_type)
        if _type == "error":
            error_manager_vector.raise_err_input_value_invalid("mem_set", "values of dtypes",
                                                               "in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]", "out of range")
        else:
            _dtypes.append(_type)
    memery_set = MemSet(sizes, _dtypes, values_int, values_float)
    return memery_set.build(kernel_name)
