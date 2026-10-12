# 记录与证据范围

旧 Run / patch / manifest 保留原件；实验仍放 `records/`，方案设计放 `studies/` 并引用证据，不复制性能账本。

## 每轮最小记录

1. 假设和合同：模型、负载、拓扑、精度、状态；预期回收哪段关键路径时间。
2. 来源：代码 commit/diff、problem/plan、成本出处；真实 tensor/路由的稳定位置与 hash，不提交权重、凭据或大 trace。
3. 干预：改变哪些决策，预测什么，什么结果否定假设，哪些保护场景不应退化。
4. 结果：正确性、实际路径、样本/噪声、有效工作量、证据类别和局限。
5. 决定：ACCEPTED / REJECTED / INCONCLUSIVE / INVALID / REUSED；下一步如何受结果影响。

已有身份和合同用不可变引用复用；未知字段显式 unknown，目标值不能替代观察。

## 证据不能越级

| 证据 | 允许结论 | 不允许结论 |
|---|---|---|
| synthetic/model estimate | 模型内合法性、候选排序 | 硬件下界、真实吞吐、性能 KEEP |
| CPU reference replay | callbacks 在该输入上的输出一致 | GLM 数值、KV/MTP 状态或 NPU 并发正确 |
| calibrated device fragment | 合同内机制与设备收益 | 未测试的完整服务收益 |
| matched model replay | 原模型中收益是否传递 | 正式动态产品/SLA 已验收 |
| matched repeated service E2E | 合同内功能、稳定容量与性能 | 未测试模型/拓扑/负载的通用收益 |

局部 ACCEPTED 不提升产品 Current；Current 只引用符合 [MISSION](MISSION.md) 的完整验收。已有局部收益不因缺产品验收而清空；累计收益须重测，不能相加。

## 性能与发布

正式性能和重型 profiling 分开；重复顺序和样本数须能区分噪声/漂移。比较实际完成的相同工作，设备完成时间与主机提交时间分开，多 rank 记录覆盖与时钟可比性。设备实验须满足当前授权。

发布可审阅代码与紧凑证据，保留失败及适用条件。旧记录的新归约写新文件引用原件；本地 demo/test 不消耗 GLM-RUN 编号，不写性能成功。
