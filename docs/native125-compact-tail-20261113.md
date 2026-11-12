> 后续结果：[125运行通过但性能不采用](native125-result-20261113.md)。本阶段仅1次加载，默认保留114；下文原型或计划描述为历史。

# FULL MTP 后四步缩行原型：125

本页记录2026年11月13日的C1产物；**尚无125的NPU正确性或性能结论**。保留默认源码114，固定强参考117。此前C4/C8单批优势34.28%/20.60%仍是历史基线，双档30%目标未完成。

## 改了什么

以114为基线，MTP第1步继续处理验证批次的24/48行；第2～5步改为TP8对齐的8行，仍生成5个草稿，保留FULL图及所有共有优化。模型、权重、量化、采样和底层算子不变。除了输入切片，还必须同步处理：

- 捕获及运行时的query、位置、slot和block table；3个metadata后端映射到2个物理缓存组。
- 每个草稿步骤的持久CP缓冲，避免后续8行metadata覆盖首步24/48行缓冲。
- 共享top-k索引的跨rank归属：原先每rank3/6行，后续每rank1行；沿用原有一次all-reduce，仅改变接收行的映射。
- SP/MoE执行上下文与8行输入保持一致，离开实例后恢复其上下文。

原型通过固定SDK源码哈希及3处AST结构核验，在DeepInfer拥有的proposer实例安装执行适配；不修改镜像内SDK文件，也不全局替换类。相关机制可以移植到其他框架，不宣称DeepInfer独占原理。

## 已完成与未完成

源码提交`7f3c015abac23ffd6d3285791b3754c2a37067cb`，父基线`c4ec8e000d04966574970802ae01db76e57a01b5`，仅4个文件变化。6项主机整数契约检查通过，覆盖输入宽度、捕获缓冲隔离、top-k分配和安装/关闭顺序；实际固定SDK函数只改变预定3处AST。

这不是CPU与NPU的模型精度对比，也不能证明NPU能执行或实际提速。候选第一次8die加载已经启动，结果待回收；阶段最多3次条件性加载，失败也计入。当前没有额外预热或额外测试请求。

## 如何复现原型

在本分支保留的114源码树上先检查并应用[隔离补丁](../validation/shared-features-20261111/native125-compact-tail.patch)：

```bash
git apply --check validation/shared-features-20261111/native125-compact-tail.patch
git apply validation/shared-features-20261111/native125-compact-tail.patch
python3 DeepInfer/tests/ascend/test_native_mtp_compact_tail.py
```

模型执行沿用[114的实际参数和环境](../validation/shared-features-20261111/native114.json)，仅增加`--native-compact-mtp-tail`。必须使用对应固定依赖、8die/TP8、GLM-5.3-Flash W8A8、MTP5/FULL/DSA-CP/SP路径；原型对不匹配源码拒绝启动。镜像本地ID不能冒充镜像仓库digest。设备使用须遵循操作者现场授权与协调机制。

固定AISBench为C4→C8，各32请求、1024输入/256输出、各2次既有预热；另有4项既定工程请求，总72请求。容器和模型启动参数除新开关与任务路径外，已与124实际manager的参数结构核对相同；124共有设置与114/117的对照证据仍保留。

## 采用条件及停止点

先以现有工程请求检查运行和状态，通过后在同一进程计分。只有同一候选C4/C8都具备30%通过前景，才允许各一次114/117新控制，核对机制净贡献及环境漂移。缺口仍在或工程失败就不采用，不增加参数网格或第四次加载。

MTP输入行调用量减少53.33%/66.67%只描述形状；主模型、通信固定开销和实际kernel效率仍在。净省多少时间尚未验证，通用质量未验收，当前没有125性能数据可宣传。

依据：[主机检查](../validation/shared-features-20261111/native125-host-checks.json)、[候选准备及负载](../validation/shared-features-20261111/native125-preparation.json)、[路线反思与固定预算](performance-direction-review-20261113.md)。
