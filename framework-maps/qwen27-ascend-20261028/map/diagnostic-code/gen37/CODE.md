# 双框架诊断代码入口（未编号草稿）

本次整合无新性能测量：这里只交付解释既有短诊断所需的代码、身份审计和 CPU 检查，不形成优化采用或新的服务复现结论。记录中的正式性能数值、波动与质量边界保持不变。

| 组件 | 改变了什么 | 可复核范围 | 入口 |
|---|---|---|---|
| vLLM gen81 | 在确切层 pre/post Event 附唯一 JSON 标签，避免缺失 capture-stream 映射时按 task 序号猜配 | raw Model/Stream/Task 与显式回放 flow；不把端点区间解释为独占 kernel | [代码与核验](../../../research/scripts/vllm-task-tag-diagnostic/README.md) |
| SGLang gen67 | 公共 Event 层端点、显式 requestID/RID 正式回放筛选；确切 graph 的 EOF 后清理 | 保留 startup、mixed 和未知；CPU stub 不证明真实设备同步或线程竞争 | [代码与核验](../../../research/scripts/sg-public-event-diagnostic/README.md) |

两份组件共用部分历史 helper，但各自保留完整 manifest 及原布局；不是两个独立实验。组件 README 中的“本包”指各自组件，历史代次事实不被本次整合改写。代码入口不包含完整服务启动器、SDK 借槽、权重、原始 profile、SO 或 transport。bootstrap-reference 只供查阅，不得直接加入 PYTHONPATH。完整源码范围与逐文件 SHA 见 [代码清单](CODE_INDEX.json)。

在仓外两个各自不存在的绝对输出目录运行：

```sh
python3 -I -B -S scripts/vllm-task-tag-diagnostic/check_cpu.py --output /absolute/new/vllm-cpu
python3 -I -B -S scripts/sg-public-event-diagnostic/check_cpu.py --output /absolute/new/sg-cpu
```

这些检查只需标准库；只验证原字节、静态任务依赖和 CPU fixtures。不会证明 aarch64/910B1 设备行为、两框架输出等价、纯框架因果或完整204完成。构建入口本次未执行，SO 字节等同与可移植构建未验证。真实服务/模型/设备工作仍需确切归属及 performance-workflow.lock。

当前相同 payload 的两份插桩短诊断文本不一致；整体 greedy/质量与差异根因未闭合。四 rank 不相加为墙钟；HCCL 等待和别名去重边界保留。基线 CANN9.2/9.0、卡组及 KV 混杂仍在；未采用优化。

本目录是 main 的候选记录，代码放 scripts；index 候选只在 main 事实确定后同源同步到 code。实际 R ID、模型登记、main/index 全仓检查、最终204地图及浏览器 QA 尚未完成。占号/提交前须刷新实时 AGENTS/README/R_TEMPLATE、账号权限及全部 refs；本次固定规则来自第29代快照，不能用来声称发布时规则已刷新。


## 保存输出谱系的CPU复核入口（gen37）

新增第三组件 [SG输出谱系CPU审计](../../../research/scripts/sg-output-lineage-audit/README.md)，三个第35代脚本保留原字节；新入口先验证SHA，再在不存在的新输出树运行14个身份反例，按需复核30份仓外固定输入。保存谱系JSON除verified_ns外完全相等，源码契约与两份台账全SHA相同。原始证据根不写入，逐件SHA见 [组件清单](../../../research/scripts/sg-output-lineage-audit/code-manifest.json)；仓外证据包位置见 [输入清单](../../../research/scripts/sg-output-lineage-audit/pinned-inputs.json)。

本组件不把观察或解码升级为质量/greedy等价；历史loaded tokenizer SHA、逐SSE消息token身份、vLLM生成ID与跨框架根因仍未知。32是逻辑输出；132是TP副本commit行。原接受native compact的关联不是独占kernel、全stream关键路径或可恢复收益。没有新服务/NPU/SDK、实际204接受或优化采用。原60份代码和22份CSV保持原字节。
