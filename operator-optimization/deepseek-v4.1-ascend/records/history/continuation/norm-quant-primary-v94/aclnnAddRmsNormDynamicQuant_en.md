# aclnnAddRmsNormDynamicQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:49:09.901Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: The RmsNorm operator is a commonly used normalization operation in large models. Compared with the LayerNorm operator, it removes the part of subtracting the mean. The DynamicQuant operator performs symmetric dynamic quantization on the input tensor. The AddRmsNormDynamicQuant operator fuses the Add operator before RmsNorm and one or two DynamicQuant operators that receive the RmsNorm normalized output, reducing data movement in and out.
- Formula:

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - If neither smoothScale1Optional nor smoothScale2Optional is input, the outputs y2Out and scale2Out are meaningless. The computation process is as follows:

  $$
   scale1Out=row\_max(abs(y))/127
  $$

  $$
   y1Out=round(y/scale1Out)
  $$

  - If only smoothScale1Optional is input, the outputs y2Out and scale2Out are meaningless. The computation process is as follows:

  $$
    input = y\cdot smoothScale1Optional
  $$

  $$
   scale1Out=row\_max(abs(input))/127
  $$

  $$
   y1Out=round(input/scale1Out)
  $$

  - If both smoothScale1Optional and smoothScale2Optional are input, all five outputs of the operator are valid outputs. The computation process is as follows:

  $$
    input1 = y\cdot smoothScale1Optional
  $$

  $$
    input2 = y\cdot smoothScale2Optional
  $$

  $$
   scale1Out=row\_max(abs(input1))/127
  $$

  $$
   scale2Out=row\_max(abs(input2))/127
  $$

  $$
   y1Out=round(input1/scale1Out)
  $$

  $$
   y2Out=round(input2/scale2Out)
  $$

  where row_max represents the maximum value of each row.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnAddRmsNormDynamicQuantGetWorkspaceSize` first to obtain the workspace size required for the computation and the executor that includes the operator computation process, and then call `aclnnAddRmsNormDynamicQuant` to perform the computation.

```Cpp
aclnnStatus aclnnAddRmsNormDynamicQuantGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *smoothScale1Optional,
  const aclTensor *smoothScale2Optional,
  double           epsilon,
  aclTensor       *y1Out,
  aclTensor       *y2Out,
  aclTensor       *xOut,
  aclTensor       *scale1Out,
  aclTensor       *scale2Out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNormDynamicQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormDynamicQuantGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (shape)</th>
        <th>Non-Contiguous Tensor</th>
      </tr></thead>
    <tbody>
    <tr>
      <td>x1 (aclTensor*)</td>
      <td>Input</td>
      <td>Source data tensor during the normalization process. Corresponds to `x1` in the formula.</td>
      <td>Supports empty tensor.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*)</td>
      <td>Input</td>
      <td>Source data tensor during the normalization process. Corresponds to `x2` in the formula.</td>
      <td><ul><li>Support empty tensor.</li><li>Shape and data type need to be consistent with `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Weight tensor in the normalization process. Corresponds to `gamma` in the formula.</td>
      <td><ul><li>Support empty tensor.</li><li>Data type needs to be consistent with `x1`.</li><li>Shape needs to be consistent with the last dimension of `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScale1Optional (aclTensor*)</td>
      <td>Input</td>
      <td>smoothScale tensor used to obtain `y1Out` during quantization. Corresponds to `smoothScale1Optional` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>Optional parameter. A null pointer can be passed in.</li><li>The shape and data type need to be consistent with `gamma`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScale2Optional (aclTensor*)</td>
      <td>Input</td>
      <td>smoothScale tensor used to obtain `y2Out` during quantization. It corresponds to `smoothScale2Optional` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Optional parameter. An empty pointer can be passed in. Must be used together with smoothScale1Optional.</li><li>Shape and data type need to remain consistent with `gamma`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Used to prevent division-by-zero errors, corresponding to `epsilon` in the formula.</td>
      <td>A small positive value, such as 1e-6, is recommended.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Quantized output Tensor, corresponding to `y1Out` in the formula.</td>
      <td><ul><li>Support empty tensors.</li><li>The shape needs to be consistent with input `x1`/`x2`, or must be two-dimensional, where the first dimension equals the product of all dimensions of `x1` except the last one, and the second dimension equals the last dimension of `x1`.</li></ul></td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Quantized output tensor, corresponding to `y2Out` in the formula.</td>
      <td><ul><li>Support empty tensors.</li><li>If smoothScale2Optional is absent, this output is meaningless.</li><li>If the `y2Out` output is meaningless, the shape is [1]; in other use cases, the shape needs to be consistent with `y1Out`.</li></ul></td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*)</td>
      <td>Output</td>
      <td>Sum of x1 and x2, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Shape and data type need to be consistent with input `x1`/`x2`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Output of the first quantization path, corresponding to `scale1Out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape needs to be consistent with the shape of input `x1` excluding the last dimension, or with the product of the dimensions of `x1` excluding the last dimension.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scale2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the output of the second quantization path, corresponding to `scale2Out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>This output is meaningless when smoothScale2Optional does not exist.</li><li>The shape needs to be consistent with `scale1Out`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be side-applied on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="A3,910b" id7 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    - The output parameters `y1Out` and `y2Out` only support INT8.
  <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If a required input, output, or attribute parameter is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of the input or output is not within the supported range.</td>
    </tr>
  </tbody></table>

## aclnnAddRmsNormDynamicQuant

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Device side-applied workspace memory address.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the Device side-applied workspace, obtained by calling the first-phase API aclnnAddRmsNormDynamicQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that includes the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. (For details, see [aclnn return code](../../../docs/en/context/aclnn_return_code.md))

## Constraints

- **Supported Type Description**

  Whether empty tensors are supported: empty input yields empty output.

- **Data Format Description**

  It is recommended to use the ND format for all input and output tensors. Other data formats are converted to the ND format by the framework by default for processing.

- **Supported Data Types by Product Model**

  <!-- npu="A3,910b" id8 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.

    | `x1` Data Type | `x2` Data Type | `gamma` Data Type | `smoothScale1Optional` Data Type | `smoothScale2Optional` Data Type | `y1Out` Data Type | `y2Out` Data Type | `xOut` Data Type | `scale1Out` Data Type | `scale2Out` Data Type |
    | ----------- | ----------- | -------------- | ----------------------------- | ----------------------------- | -------------- | -------------- | ------------------ | ------------------ | ------------------ |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16                       | FLOAT16                       | INT8           | INT8           | FLOAT16            | FLOAT32            | FLOAT32            |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16                      | BFLOAT16                      | INT8           | INT8           | BFLOAT16            | FLOAT32            | FLOAT32            |
  <!-- end id8 -->

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:

    | `x1` Data Type | `x2` Data Type | `gamma` Data Type | `smoothScale1Optional` Data Type | `smoothScale2Optional` Data Type | `y1Out` Data Type | `y2Out` Data Type | `xOut` Data Type | `scale1Out` Data Type | `scale2Out` Data Type |
    | ----------- | ----------- | -------------- | ----------------------------- | ----------------------------- | -------------- | -------------- | ------------------ | ------------------ | ------------------ |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16                       | FLOAT16                       | INT8           | INT8           | FLOAT16            | FLOAT32            | FLOAT32            |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16                      | BFLOAT16                      | INT8           | INT8           | BFLOAT16            | FLOAT32            | FLOAT32            |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16                       | FLOAT16                       | HIFLOAT8           | HIFLOAT8           | FLOAT16            | FLOAT32            | FLOAT32            |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16                      | BFLOAT16                      | HIFLOAT8           | HIFLOAT8           | BFLOAT16            | FLOAT32            | FLOAT32            |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16                       | FLOAT16                       | FLOAT8_E5M2           | FLOAT8_E5M2           | FLOAT16            | FLOAT32            | FLOAT32            |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16                      | BFLOAT16                      | FLOAT8_E5M2           | FLOAT8_E5M2           | BFLOAT16            | FLOAT32            | FLOAT32            |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16                       | FLOAT16                       | FLOAT8_E4M3FN           | FLOAT8_E4M3FN           | FLOAT16            | FLOAT32            | FLOAT32            |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16                      | BFLOAT16                      | FLOAT8_E4M3FN           | FLOAT8_E4M3FN           | BFLOAT16            | FLOAT32            | FLOAT32            |
  <!-- end id9 -->

- Deterministic computation:
  - aclnnAddRmsNormDynamicQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_dynamic_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate: resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId); aclFinalize(); return ret);

    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to the device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct input and output tensors as required by the API.
    std::vector<int64_t> xShape = {2, 8};
    std::vector<int64_t> gammaShape = {8};
    std::vector<int64_t> reduceShape = {2};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* smooth1DeviceAddr = nullptr;
    void* smooth2DeviceAddr = nullptr;

    void* y1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* scale1DeviceAddr = nullptr;
    void* scale2DeviceAddr = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* smooth1 = nullptr;
    aclTensor* smooth2 = nullptr;
    aclTensor* y1 = nullptr;
    aclTensor* y2 = nullptr;
    aclTensor* x = nullptr;
    aclTensor* scale1 = nullptr;
    aclTensor* scale2 = nullptr;

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t gammaShapeSize = GetShapeSize(gammaShape);
    int64_t reduceShapeSize = GetShapeSize(reduceShape);

    std::vector<short> x1HostData(xShapeSize, 0x3800);
    std::vector<short> x2HostData(xShapeSize, 0x3800);
    std::vector<short> gammaHostData(gammaShapeSize, 0x3e00);
    std::vector<short> smooth1HostData(gammaShapeSize, 0x3e00);
    std::vector<short> smooth2HostData(gammaShapeSize, 0x3e00);

    std::vector<short> y1HostData(xShapeSize, 0);
    std::vector<short> y2HostData(xShapeSize, 0);
    std::vector<short> xHostData(xShapeSize, 0);
    std::vector<short> scale1HostData(reduceShapeSize, 0);
    std::vector<short> scale2HostData(reduceShapeSize, 0);

    float epsilon = 1e-6;

    // Create the x1 aclTensor.
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x2 aclTensor.
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the smooth1 aclTensor.
    ret = CreateAclTensor(smooth1HostData, gammaShape, &smooth1DeviceAddr, aclDataType::ACL_FLOAT16, &smooth1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the smooth2 aclTensor.
    ret = CreateAclTensor(smooth2HostData, gammaShape, &smooth2DeviceAddr, aclDataType::ACL_FLOAT16, &smooth2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the y1 aclTensor.
    ret = CreateAclTensor(y1HostData, xShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the y2 aclTensor.
    ret = CreateAclTensor(y2HostData, xShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the outScale1 aclTensor.
    ret = CreateAclTensor(scale1HostData, reduceShape, &scale1DeviceAddr, aclDataType::ACL_FLOAT, &scale1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create outScale2 aclTensor.
    ret = CreateAclTensor(scale2HostData, reduceShape, &scale2DeviceAddr, aclDataType::ACL_FLOAT, &scale2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAddRmsNormDynamicQuant.
    ret = aclnnAddRmsNormDynamicQuantGetWorkspaceSize(
        x1, x2, gamma, smooth1, smooth2, epsilon, y1, y2, x, scale1, scale2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormDynamicQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAddRmsNormDynamicQuant.
    ret = aclnnAddRmsNormDynamicQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormDynamicQuant failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, and copy the result from the device to the host. Modify the code based on the specific API definition.
    auto size = GetShapeSize(xShape);
    std::vector<int8_t> y1Ret(size, 0);
    ret = aclrtMemcpy(
        y1Ret.data(), y1Ret.size() * sizeof(y1Ret[0]), y1DeviceAddr, size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, y1Ret[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(smooth1);
    aclDestroyTensor(smooth2);
    aclDestroyTensor(y1);
    aclDestroyTensor(y2);
    aclDestroyTensor(x);
    aclDestroyTensor(scale1);
    aclDestroyTensor(scale2);

    // 7. Release device resources. Modify the code based on the specific API definition.
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(smooth1DeviceAddr);
    aclrtFree(smooth2DeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(y2DeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(scale1DeviceAddr);
    aclrtFree(scale2DeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
