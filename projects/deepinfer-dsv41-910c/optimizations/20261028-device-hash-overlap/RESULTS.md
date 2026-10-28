# 910C 推理优化测试记录

记录日期：2026-10-28

## 1. 测试范围

测试使用服务器 172、DeepSeek V4.1 Flash ModelSlim W8A8 检查点和 8 张物理 Ascend 910C 卡，对应 16 个逻辑设备。并行配置为 DP2 × TP8、EP16；每个 DP 处理 4 个并发请求。Engram 表保留在 CPU，两个 DP 共享 memfd 映射并由 NPU 读取；主干权重不使用 CPU prefetch。

每轮使用 AISBench 执行 8 个请求，并发 8；每请求输入 32768 token、输出 1024 token，采用贪心生成并忽略 EOS。每轮正式请求共输入 262144 token，实际 KV 命中 235904 token，命中率为 89.9902%。正式测试前执行前缀种子预热和 C8、输出 32 token 的预热。合并输出吞吐按两轮输出 token 总数除以两轮总耗时计算。

两项实验分别比较设备 hash 和共享专家多流开关。两项实验的服务版本和对照方式不同，不能将各自的组间差值直接相加。冻结 vLLM Ascend 记录未在本组实验期间重新调优和复测。

## 2. 版本边界

| 版本 | 用途 | 源码文件数 |
|---|---|---:|
| `allopt-mc2-cap4096-v1-20261028` | 既有冻结部署版本，保留原部署说明、启动脚本和源码归档 | 1436 |
| `allopt-engram-devicehash-v1-20261028` | 设备 hash 单因素实验 | 1437 |
| `allopt-shadow-overlap-v1-20261028` | 共享专家串行对照所用源码和服务 | 1440 |
| `allopt-shared-overlap-on-v1-20261028` | 使用同一 1440 文件源码、开启共享专家 decode 多流的候选服务 | 1440 |

后两个服务使用同一源码目录、镜像、权重和已预热编译缓存；共享专家多流开关在模型启动及图捕获前固定。旧部署文档和启动脚本对应 1436 文件版本，不用于直接启动上述新候选。

## 3. Engram 设备 hash

对照在同一服务进程和同一源码中进行，顺序为 CPU hash 第一轮、NPU hash 第一轮、NPU hash 第二轮、CPU hash 第二轮。两种模式的输入、DP/槽位、UVA 查表、图执行和其他配置一致。HCCL NPU 端口均使用 `auto`。

| 指标 | CPU hash | NPU hash |
|---|---:|---:|
| 第一轮输出吞吐，token/s | 553.29 | 569.79 |
| 第二轮输出吞吐，token/s | 551.41 | 566.73 |
| 两轮合并输出吞吐，token/s | 552.35 | 568.26 |
| 平均首 token 延迟，ms | 786.15 | 782.00 |
| 平均 TPOT，ms | 13.40 | 13.05 |
| 完整输出与冻结参考一致 | 16/16 | 16/16 |

在该组固定负载下，两轮合并吞吐提高 2.88%，两组对应轮次的差值均为正。四轮合计 32 个完整输出与冻结参考一致；每轮核验 16 个 rank 的实际 hash 和图执行回执，并观察到草稿拒绝。

新路径仍由 CPU 维护历史并构造原始窗口，在目标图入口前执行设备 hash 和 UVA 查表。已验证的完整模型执行路径使用 24-token 目标图；其他图形状只有单卡原语证据。本次两轮结果不能证明其他负载下的稳定收益，也不能作为全流程 ZOS 已完成的依据。

## 4. 共享专家 decode 多流

串行服务与候选服务使用相同的 1440 文件源码、权重、镜像和工作负载。候选服务在启动前设置 `DEEPINFER_910C_SHARED_EXPERT_OVERLAP_DECODE=1`，在 decode 图中将共享专家提交到辅助流，路由专家仍按原顺序执行融合 MC2，合并结果前进行流同步。预填充路径保持串行。

两组测试分别启动服务、捕获图并完成预热，各执行两轮。该过程引入服务重启和设备状态变化，因此该组不属于同进程 ABBA 对照。

| 指标 | 串行对照 | 多流候选 |
|---|---:|---:|
| 第一轮输出吞吐，token/s | 571.66 | 586.12 |
| 第二轮输出吞吐，token/s | 565.34 | 581.85 |
| 两轮合并输出吞吐，token/s | 568.48 | 583.97 |
| 平均首 token 延迟，ms | 779.50 | 788.85 |
| 平均 TPOT，ms | 13.00 | 12.60 |
| 完整输出与冻结参考一致 | 16/16 | 16/16 |

多流候选的两轮合并吞吐高 2.73%，平均首 token 延迟增加 9.35 ms。每组两轮均核验 16 个 rank、真实缓存命中和完整输出；该结果仅适用于本次固定负载。

设备 trace 中观察到不同流上的候选算子时间区间相交，但现有记录无法可靠关联到同一次 decode 图重放和同一层。设备端实际时间重叠尚未证实，吞吐差异不能据此归因于已证实的算子并行。

## 5. 源码与证据

源码补丁的应用与校验方法见[源码补丁与校验](PATCHES.md)。

设备 hash 版本相对旧部署版本新增 `engram_hash910c.py`，并修改目标图、模型和 worker/API 入口等文件。共享专家多流实验相对设备 hash 版本包含 7 个变化文件；证据包同时保留这些文件的前后版本和统一补丁。具体源码以各版本逐文件 SHA256 清单为准。

| 证据文件 | 内容 | SHA256 |
|---|---|---|
| `/data/tiankuan/wio/910c-deepinfer-172/delivery-optimizations/allopt-engram-devicehash-v1-20261028/engram-hash-evidence.tar.gz` | 设备 hash 四轮原始结果、逐 rank 回执、独立审计及 1437 文件源码清单 | `f5123d683ef332afa9f45b4870e993cecb994ca4fc3c25d3af12bd0f9603d58d` |
| `/data/tiankuan/wio/910c-deepinfer-172/delivery-optimizations/shared-overlap-ab-20261028/shared-overlap-ab-evidence.tar.gz` | 多流 A/B 原始结果、独立审计、转换记录、源码差异及版本清单 | `536e8727a46ebf8f3bda7d690f641677df3cc213df84779ef27eb11bc4dfac96` |
| `/data/tiankuan/wio/910c-deepinfer-172/serving/allopt-shared-overlap-on-v1-20261028/jobs/profile-overlap-decode-20261028/profile-rank0/bm-jn-zs-zone1-910C-64G-10-172_1210467_20261028122602266_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json` | 独立 decode 诊断的原始 trace；结论为证据不足 | `e13d39a6d16fb2b00361c46aecfec0ae0abd3fb9a851fe9550258e33288be451` |

原始 trace 单独保存在服务器诊断目录，未包含在 A/B 证据包中。

既有冻结部署的 `source.tar.gz` SHA256 为 `ec00bee91f16f1fa85d2ff0e917ace7c5815fbe3e07ec4674fd0dd6605b4378b`。上述新增证据包不包含模型权重；它们记录实验及源码差异，不能替代对候选服务启动和其他工作负载的验证。
