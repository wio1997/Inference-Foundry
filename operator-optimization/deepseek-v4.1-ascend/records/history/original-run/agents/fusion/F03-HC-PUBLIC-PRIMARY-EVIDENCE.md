# F03：HC 分块的公开上游证据与 169 待核验边界

## 公开原始资料提供的新信息

- vLLM-Ascend 的公开 [issue #19357](https://github.com/vllm-project/vllm-ascend/issues/19357) 报告：从 commit `fbb4d867d22095047473e3c7ca9db053f5f4c7b0` 移植到 xLLM 的 A3 `npu_hc_pre_v3`，Sinkhorn 的列求和结果按 `hcMultAlign` 间距存储，而对应的加 epsilon 元素长度使用了 `hcMult`；例如 HC=4、对齐=8 时，不同 batch 行的位置可能造成 `comb` 差异。报告者的真实 NPU 复现与验证是**移植算子**，不是本次 169 已运行二进制的复现。其测试中 full/chunk/token 执行在报告的条件下可出现数值差异，不能假设改变 token 分块后逐位结果不变。
- 同仓库 [PR #19401](https://github.com/vllm-project/vllm-ascend/pull/19401) 页面在本次查询时标记 **Open**。PR 描述把修正指向 `hc_pre_m_k_split_core.h` 中的四个列归一化位置，并称行归一化使用 packed 长度、compact tiling 本来使用对齐长度、A5 寄存器 kernel 不属于同一影响范围。该 PR 自述的测试结果是提交者声明，不等于本工作对 169 的验证；issue 显示 Closed 也不能推断 PR 已合并或当前镜像包含修复。
- issue 的信息能区分 **按 token 独立的数学意图**与**具体实现的分块位置敏感**：相关现象不等于证明数学上 token 相互依赖，但它说明即使理论上按 token 拆分，其既有实现也可能根据 rowFactor/位置给同样值不同结果。必须保留原始基线所要求的输出语义；选择修复行为会改变对照，需要单独明确正确性目标。

这些是公开特定源码版本的报告与提交信息。其容器名称与本项目镜像标签相同**不等于镜像 digest、patch、已加载算子二进制、tiling key 或历史运行分支相同**。尤其 PR 自述其受影响的是特定 `m_k_split` 实现而非所有路径；现有材料不证明目标 `[M,HC,H]` 在 169 走该路径，也不证明生产服务结果受影响。

## 准备好的最小只读核验

`f03_hc_inventory_inside.py` 仅枚举容器内固定 HC pre/post 源码子树与单一已知参考测试路径，限制目录数量、单目录入口数、文件数和字节，拒绝 symlink；不会 import 框架、运行模型或设备。`snapshot_f03_hc_inventory.py` 经既有 SSH wrapper 对冻结 169 baseline/stage 容器校验镜像 digest、源路径覆盖挂载与文件 SHA/内容，要求两容器结果一致，保存 `f03-hc-inventory.json`；缺失的路径列为 missing，禁止覆盖已有不同内容。容器中源码一致也**不证明实际运行的二进制由它构建**；需另有可验证构建关联与运行 tiling 信息。

预期仅在拿到源码后审阅：v3 / post 输入输出 dtype 与 shape、归约轴、不同 token 的数据依赖、每条 tiling 路径及其条件、涉及 epsilon 的对齐/索引，以及 post 是否支持输出在不同消费者可见。若源码缺失或二进制关联缺失，明确 UNKNOWN；不强行归因或新增硬件测试。
