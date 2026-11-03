# Stage 089：Ascend sampler patch 的实际静态注册链

ZCode 的 `operator-ceiling169-v2-089-sampler-patch.exit-code` 为 0，verified 回执的执行模型为 `deepseek-flash`。baseline、stage 两容器的镜像 ID 一致，文件路径没有覆盖挂载，以下三份快照的重算 SHA 与两容器结果一致：

| 文件 | SHA-256 |
|---|---|
| `sampler-patch-snapshot/patch.py` | `f037f710e0b7f80e73b94d9b93ce87aacddf58af2b3e6f5df53426717d95e7a3` |
| `sampler-patch-snapshot/registration.py` | `1b08bf00aca38554156c91527cce8bdf67ea875cd984c8dc56ff7ed26cbf08ee` |
| `sampler-patch-snapshot/adapt_patch.py` | `5f88fd83bcba539c3d24324ed104469914254989996342b27bb36a326e3f8458` |

## 静态数据流

- `adapt_patch.py:683-687`：当 `adapt_patch(is_global_patch=False)` 被调用时，导入 `vllm_ascend.patch.worker`。
- `registration.py:45`：该 worker 注册模块无条件导入 `patch_rejection_sampler`。
- `patch.py:1-9`：导入 `vllm_ascend.sample.rejection_sampler` 的 `rejection_sample` 等，并把 vLLM rejection sampler 模块的对应名称指向 Ascend 函数。

因此 Stage 088 看到的原始 vLLM 函数宽度不能直接视为 NPU worker 当前实现；如果 worker 注册路径已执行且后续未被覆盖，vLLM 模块内的同名调用会走 Ascend 实现。当前材料仍没有证明该 worker 实际何时执行注册、是否经过其他替换，也没有看到 Ascend 函数及 metadata 构造细节；不能直接断言 5+1 是这次全体输出的边界，更不能将它转换为实际性能上界。

已准备 `compare_sampler_impl_readonly.py`，核对并快照上述 patch 指向的 Ascend 函数文件以及 vLLM sampler 导入的 spec metadata 文件；由主控安排 ZCode 执行。还需证明配置 5 与 metadata 的对应关系、prefill 首 token 与其他输出路径、双 DP 调度事件之间的计数。上述约束满足之前，`oem-and-scheduling-symbolic.md` 中 `g` 为 UNKNOWN。

无新增 NPU 工作、无服务代码变更。
