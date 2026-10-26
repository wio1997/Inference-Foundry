# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
dynamic reduce mean
"""
from impl.ops_legacy.util.platform_adapter import tvm
from impl.ops_legacy.util.platform_adapter import tbe
from impl.ops_legacy.util.platform_adapter import classify
from impl.ops_legacy.util.platform_adapter import para_check
from impl.ops_legacy.util.platform_adapter import shape_util
from impl.ops_legacy.util.platform_adapter import tbe_platform
from impl.ops_legacy.util.platform_adapter import tbe_context
from impl.ops_legacy.util.platform_adapter import OpImplMode
from impl.ops_legacy.util.platform_adapter import OpPatternMode
from impl.ops_legacy.util.platform_adapter import register_operator
from impl.ops_legacy.util.platform_adapter import register_operator_compute
from impl.ops_legacy.util.util_compute import get_cof
from impl.ops_legacy.util.util_compute import only_static_support
from impl.ops_legacy.util.util_common import check_op_impl_mode


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals
# 'pylint: disable=unused-argument,invalid-name
def get_calc_dtype(dtype_x, impl_mode):
    """
    get calc dtype
    """
    if dtype_x == "float32":
        calc_dtype = "float32"
    elif dtype_x == "float16":
        cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        if not tbe_platform.api_check_support("tbe.dsl.sum", "float32"):
            calc_dtype = "float16"
        elif cce_product in ("Ascend310", "Ascend310B", "AS31XM1") and impl_mode == OpImplMode.HIGH_PERFORMANCE:
            calc_dtype = "float16"
        else:
            calc_dtype = "float32"
    else:
        # int8 and uint8
        calc_dtype = "float16"
    return calc_dtype


@register_operator_compute("ReduceMean", op_mode="dynamic", support_fusion=only_static_support, support_bfp16=True)
def reduce_mean_compute(x,
                        axes,
                        y,
                        keepdims=None,
                        noop_with_empty_axes=True,
                        kernel_name="reduce_mean",
                        impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """reduce_mean compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    axes: int, list, tuple or NoneType
        the axes for reduce.
    y: dict
        the dict of output tensor.
    keepdims: bool or NoneType
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_mean_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    shape_x = shape_util.shape_to_list(x.shape)
    dtype_x = x.dtype
    reduce_elts = get_cof(axes, shape_x)
    calc_dtype = get_calc_dtype(dtype_x, impl_mode)

    if dtype_x != calc_dtype:
        data_input_tmp = tbe.cast_to(x, calc_dtype)
    else:
        data_input_tmp = x

    if isinstance(reduce_elts, float):
        if reduce_elts == 0:
            res = tbe.reduce_mean(data_input_tmp, axis=axes, keepdims=keepdims)
            if dtype_x != calc_dtype:
                if dtype_x in ("int8", "uint8"):
                    res = tbe.cast_to(res, dtype_x, False)
                else:
                    res = tbe.cast_to(res, dtype_x)
            return res
        
        cof = reduce_elts**(-1)
        cof = tvm.const(cof, dtype=calc_dtype)
    else:
        cof = tbe.var("cof", dtype=calc_dtype)
        if calc_dtype == "float16":
            tbe.var("cof_empty", dtype=calc_dtype)
        tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", calc_dtype)

    data_input_tmp = tbe.reduce_sum(data_input_tmp, axis=axes, keepdims=keepdims)
    res = tbe.vmuls(data_input_tmp, cof)

    if dtype_x != calc_dtype:
        if dtype_x in ("int8", "uint8"):
            res = tbe.cast_to(res, dtype_x, False)
        else:
            res = tbe.cast_to(res, dtype_x)

    return res


@register_operator("ReduceMean")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_mean(x,
                axes,
                y,
                keep_dims=False,
                noop_with_empty_axes=True,
                kernel_name="reduce_mean",
                impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    Reduce a tensor on a certa in axes based on mean.

    Parameters:
    ----------
    x : dict
        shape and dtype of input
    axes : dict
        shape and dtype of input
    y: dict
        shape and dtype of output
    keep_dims : bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    noop_with_empty_axes : bool, NoneType
        useless attr to avoid dynamic reduce_mean_d compile error.
    kernel_name : str
        cce kernel name, default value is reduce_mean_d

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    keepdims = False if keep_dims is None else keep_dims
    dtype_x = x.get("dtype")
    dtype_lower_x = dtype_x.lower()
    check_list_x = ("bfloat16", "float16", "float32", "int8", "uint8")
    para_check.check_dtype(dtype_lower_x, check_list_x)
    x["rel_pos_to_reduce"] = "before"

    dtype_axes = axes.get("dtype")
    dtype_lower_axes = dtype_axes.lower()
    check_list_axes = ("int32", "int64")
    para_check.check_dtype(dtype_lower_axes, check_list_axes, param_name="axes")
    axes["rel_pos_to_reduce"] = "axis"

    tbe_context.get_context().add_compile_info("axes_idx", 1)
    if "const_value" in axes.keys():
        axes["value"] = list(axes["const_value"])

    schedules = []
    tensors = []
    ins = classify([x, axes], OpPatternMode.REDUCE, {"keepdims": keepdims is True})

    for (_x, _axes) in ins:
        with tbe.compute():
            shape_x, shape_axes = shape_util.variable_shape([_x, _axes], op_mode="reduce")
            data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=dtype_lower_x)
            data_input_axes = tvm.placeholder(shape_axes, name="data_input_axes", dtype=dtype_lower_axes)
            axes_d = shape_util.axis_check(len(shape_x), _axes.get("value"))
            res = reduce_mean_compute(data_input_x,
                                      axes_d,
                                      y,
                                      keepdims,
                                      noop_with_empty_axes=noop_with_empty_axes,
                                      kernel_name=kernel_name,
                                      impl_mode=impl_mode)
            tensors.append([data_input_x, data_input_axes, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
