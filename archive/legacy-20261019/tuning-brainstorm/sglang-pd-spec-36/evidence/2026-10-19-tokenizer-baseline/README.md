# M001：动态分词的同条件 CPU 对照

实际安装版 `AsyncDynamicbatchTokenizer` 在本轮的 2×512 输入中，`max_batch_size=32` 的批完成时间明显高于 `1`，CPU 时间没有超过 A/A 波动的改善。3×32768 输入也未建立稳定收益。不能把动态批量分词当作低并发下必然加速的开关。

本轮比较已有参数，未修改 SGLang，也未部署生产优化。完整 PD＋MTP 服务仍为 **E0**，未建立本任务服务器基线。

## 固定条件

安装源码 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`，镜像 `sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263`。使用真实完整 SGLang tokenizer loader 和原类，13 个源码文件及 4 个模型元数据/tokenizer 文件前后校验。输入沿用[上一检查点的五份输入](../2026-10-19-tokenizer-inputs/input-fixtures/manifest.json)：固定文本、模板和预期 IDs；合计 99,328 tokens。它们是重复标记构造的合成输入，不能代表自然业务分布。

CPU 依赖视图沿用前轮：重用镜像的包文件，独立 venv 排除可选 `torch_npu` 和启动 `.pth`；Transformers 5.12.1、tokenizers 0.22.2、Jinja 3.1.6、Torch 2.10.0+cpu，Rust 并行关闭。四个文件只读挂载，无权重、设备或模型请求。每个新容器限制 1 CPU、4 GiB、64 pids、禁网、只读根、删除 capabilities，结束后按精确 ID/标签清理；已有服务 ID、启动时间和重启计数前后相同。

## 测量与保留的错误

每个 case 各 12 个 block，交织 A/A 与 A/B，两对及成员顺序按种子 7911 打乱。A0/A1 是独立同参数 actor（cap=1），B cap=32；窗口均 2 ms。每 actor 预热 3 批，样本间隔 0.1 s。所有请求在第一次 yield 前排队；计时覆盖队列、编码和 future 完成，记录 wall 与包含 worker 的 process CPU；完整值和 ID 哈希在区间外核对。48 样本/case/run，不估计 p99。

R001/R002 共 96 样本输出均正确，但最初探针假设 cgroup v2 根路径，所有 cgroup 采样实际为 `unavailable`。首次“未见限流”的解读已撤回，旧数据原样保留，只作正确性及描述性证据。R003/R004 在新文件、新 Run、新容器中重测，唯一测量修复是按 `/proc/self/cgroup` 和 mountinfo 发现 v1/v2 路径，并验证配额。本机实际为 v1 `cpu,cpuacct`，quota/period 都是 100,000 us；纳秒 `throttled_time` 保留原值，另换算 us。

不能把累计启动限流当计时区间限流；本轮使用每个区间前后的计数差。没有物理核绑定、隔离 CPU 或物理峰值校准。cgroup CPU usage 原计数未采集，早期分析中的缺省 0 **不代表使用量为零**；[最终分析](analysis-final.json)改用 null/availability 标志，真实 process CPU 原样保留。

## 配对结果

| 输入 | cap=1 A/B 子集 wall 中位数 | cap=32 wall 中位数 | 配对 A/B 几何均值与 95% 描述性 bootstrap 区间 | A/A 区间 | 解释 |
|---|---:|---:|---|---|---|
| 2×512 | 1.698 ms | 3.746 ms | 0.4595 [0.4441, 0.4764] | [0.9364, 1.0044] | cap32 更慢，超过本次 A/A 波动；48 个区间无 CFS 限流增量 |
| 3×32768 | 151.966 ms | 152.673 ms | 1.0282 [0.9849, 1.0777] | [0.9976, 1.1063] | 区间重叠，不接受改善；另有 3 个 A/A-A0 区间限流，总计 2,454 us |

比值为 A/另一方：大于 1 才表示另一方更快。对 12 个 block 的配对 log ratio 重采样 4,000 次后取均值并指数化；这是固定合成负载的描述性区间，不是跨机器或跨业务保证。没有删行或过滤离群值。

短输入 process CPU 比值 1.0231，区间 [0.9991, 1.0460]；A/A [0.9539, 1.0130]，未分离。长输入 CPU 比值 1.0454，[1.0016, 1.1002]，与 A/A [1.0001, 1.1105] 重叠；仅看 A/B 区间略高于 1 会误报改善。长输入所有 A/B 区间无 CFS 增量，但 A/A 噪声与额外限流仍需保留，整体性能结论 INCONCLUSIVE。

## 机制解释与下一实验

安装版循环在开始时队列非空才进入凑批。收集最后一个请求后，只要数量未达到 cap，还会继续等待下一项，直到 2 ms 窗口耗尽。短输入约 2.05 ms 的 wall 中位数差、CPU 接近，与该源码解释相容；尚未单独测量等待分量，不能把全部差值精确归因给它。

下一轮只比较 cap32 与 case 的请求数 N（2/3），窗口、输入、到达过程不变，检验“达到 cap 即停止收集”能否省掉尾部等待。capN 是已知封闭批次的条件对照，不能读取未来请求数，不能冒充通用在线策略。更低 cap 可能拆分更大的突发，减少摊销并影响吞吐；需保留长输入反证和后续不整批/错开到达保护。没有向当前服务应用参数。

## 可复核证据

复现时以此目录为 scratch root，保留 scripts/specs/sources/fixtures 的目录关系。将上一检查点 `input-fixtures` 下五个 JSON 复制到 `fixtures/M001-v1/`，使用本目录提供的原始 manifest（不要用上一检查点为展示而改写路径的 manifest），逐文件校验其中 SHA-256。四个模型文件由有访问权的环境提供，必须匹配 specs 中哈希。具体 controller 保留本次 SSH/socket/Pod/镜像位置；其他环境需先建立新合同和新冻结 spec，不能改旧输入后复用旧证据。原环境可执行 `python3 scripts/execute_frozen.py specs/L011-R003.json` 或对应 Run 命令；无需 ZCode 私有配置。所有采样值来自上述现存 Run，复现会生成新产物，不覆盖公开 receipts。

- [最终配对分析](analysis-final.json)、[原始分析盲区](analysis-first.json)、[修复后初版分析](analysis-repair.json)。
- [R001](results/L011-R001.json)、[R002](results/L011-R002.json)、[R003](results/L011-R003.json)、[R004](results/L011-R004.json)：全部计时行、ID 哈希与环境。
- [原协议](specs/L011-protocol.json)、[修复协议](specs/L011-protocol-v2.json)、scripts 中的可复现 controller/probe/analyzer、receipts 中的冻结输入与退出码。
- [ZCode 执行审计](zcode-audit.json)：每个冻结命令恰好一次，实际 session 的模型别名 `deepseek-flash`；后端发布版本 UNKNOWN。主控编写和审查代码，ZCode 执行冻结命令，CLI 成功不代替内层结果。
- [产物索引](artifact-index.json)列出公开内容和本地原始产物哈希。无私有配置、模型 I/O 或凭据。

本轮 192 个测量样本均保持完整输出与固定 ID 一致。该结果仅适用于 CPU 类的参数对照；HTTP 路径是否启用、模板/API 流、PD 传输、MTP 接受长度、TTFT/ITL 和 NPU 吞吐均未测。
