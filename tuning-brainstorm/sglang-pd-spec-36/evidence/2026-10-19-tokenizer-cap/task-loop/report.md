# M001：固定封闭批次的 cap 条件对照

对预先固定的两条 512-token 输入，同时排队且窗口仍为 2 ms 时，把现有 `max_batch_size=32` 改为 2，实际原类的批完成中位数从 **3.668 ms 降至 1.652 ms**。配对效应超过本轮 A/A 波动，CPU 时间没有明确下降。三条 32,768-token 输入改成 cap=3，尚未建立超出 A/A 波动的改善。

这是已有参数的条件对照，未修改 SGLang、未向服务应用配置。整体 Ascend PD＋MTP 服务仍为 **E0**，accepted_baseline 为 null。

## 为什么测它

[前轮基线](../../2026-10-19-tokenizer-baseline/README.md)显示短输入 cap32 比 cap1 多约 2 ms，CPU 效应与 A/A 重叠。源码在已收齐当前排队请求、数量尚未达到 cap 时继续等到窗口到期；达到 cap 则结束收集。本轮将 cap 固定为 2 或 3，检验同一个封闭批次是否可避免尾等待。没有读取未来到达数量：这两个 cap 在执行前按 fixture 合同写入协议。

## 固定合同与结果

沿用前轮完整 CPU tokenizer loader、真实 `AsyncDynamicbatchTokenizer`、精确文本/ID、13 个源码哈希及 4 个只读模型文件；安装 commit 为 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`。同镜像、独立 CPU venv 排除可选 torch_npu，禁用 Rust 并行。任务容器 1 CPU、4 GiB、禁网、无权重/NPU设备、只读根；计时区间前后采集实际 cgroup v1 计数，quota/period 均为 100,000 us，按精确 ID 清理，现有服务身份和启动状态前后不变。

每个 case 有 12 个 block、48 行测量；A0/A1 cap32 作 A/A，A0/B 作 A/B，B 分别 cap2/3。每 actor 预热 3 批，固定种子 7911 随机化两对和成员顺序、间隔 0.1 s。所有请求在第一次 yield 前排队，计时覆盖队列、分词与 future 完成；校验在区间外。所有 **96 行完整输出与固定 ID 哈希一致**。wall 是整批完成时间，不是服务 TTFT 或单个请求延迟。

| 输入 | A/B 中 A0 cap32 的 wall 中位数 | B 的 wall 中位数 | 配对 A/B 几何均值 [95% 描述性区间] | A/A 区间 | 结论 |
|---|---:|---:|---|---|---|
| 2×512，cap2 | 3.668 ms | 1.652 ms | 2.2305 [2.1693, 2.2806] | [0.9802, 1.0043] | 局部 wall 改善超过本次 A/A；48 区间无 CFS 限流增量 |
| 3×32768，cap3 | 145.443 ms | 142.634 ms | 1.0217 [1.0150, 1.0287] | [0.9966, 1.0166] | 区间仍重叠，未接受改善；5 区间有 CFS 限流增量 |

比值 A/B 大于 1 表示 B 更快。对 12 个 block 的配对 log ratio 重采样 4,000 次再指数化；保留全部行，不过滤限流或离群值，不估 p99。长输入 5 次区间限流共 827 us：A/A A0 232 us、A1 584 us；A/B A0 3 us、B 8 us。差值很小仍完整披露；A/A 区间重叠本身已不足以接受长输入收益。

短输入 process CPU 的 A/B 比值 0.9929，[0.9770, 1.0070]，A/A [0.9734, 1.0102]；长输入 1.0064，[1.0004, 1.0126]，A/A [0.9967, 1.0163]。两组 CPU 效应都与 A/A 重叠。因此短输入 wall 改善与省掉尾等待相容，不能称减少了同等比例的编码计算，也不能将 2.23 倍写成 NPU 或服务加速比。

## 边界、反证与下一步

固定 cap2 对更大突发会拆分批次；错开到达可能减少摊销。该结果只覆盖封闭、同时到达且同 kwargs 的两请求 fixture，不能证明未知在线负载下 cap2 最优。没有 HTTP/frontend 动态分词启用证据，没有 PD 传输、MTP 接受长度、TTFT/ITL 或 NPU 吞吐数据，没有物理峰值校准。

Loop 总体保留 INCONCLUSIVE：短输入的局部实测成立，长输入和一般到达流未建立收益。本轮采用同一 Run 内交织 A/A 与 A/B，taskctl 的优化 ACCEPTED 门槛要求单独 before/after Run 记录；没有为满足该门槛伪造独立测量或拆分成新实验。事实和作用域保存在 knowledge-delta 中，整体基线不提升。

下一轮可以实现仅处理当前队列已有请求的 drain 候选，保留 cap32 而不等待未来到达。先覆盖单请求、同时/错开到达、混合 kwargs、取消、异常恢复和队列顺序，再决定是否计时；更小批次可能牺牲 CPU 摊销，必须计入反例。它是待验证的新调度候选，本轮没有运行或部署。

## 复核

- [配对分析](../analysis.json)、[短输入全部样本](../results/L012-R001.json)、[长输入全部样本](../results/L012-R002.json)。
- [冻结协议](../specs/L012-protocol.json)、[原始输入 manifest](../fixtures/M001-v1/manifest.json)、scripts、specs、receipts 和容器安全/清理回执。
- [ZCode 精确 session 审计](../zcode-audit.json)：两个冻结命令各执行一次，模型元数据别名 deepseek-flash，后端发布版本 UNKNOWN。主控编写/分析，ZCode 执行冻结命令；不以 CLI 退出码代替结果。
- [产物索引](../artifact-index.json)保留公开及本地原始文件哈希。无私有配置、模型 I/O、权重或凭据。

复现方法与[前轮](../../2026-10-19-tokenizer-baseline/README.md)相同：以本目录为 scratch root，从输入检查点复制五份 JSON 到 fixtures/M001-v1，使用本目录原 manifest 校验；提供匹配哈希的四个模型文件和原合同环境，再执行 `python3 scripts/execute_frozen.py specs/L012-R001.json` 或 R002。其他机器须新建位置合同与 spec；不能修改旧输入后复用旧证据。
