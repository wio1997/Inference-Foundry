# 双框架诊断代码入口（未编号草稿）

本次整合无新性能测量：这里只交付解释既有短诊断所需的代码、身份审计和 CPU 检查，不形成优化采用或新的服务复现结论。记录中的正式性能数值、波动与质量边界保持不变。

| 组件 | 改变了什么 | 可复核范围 | 入口 |
|---|---|---|---|
| vLLM gen81 | 在确切层 pre/post Event 附唯一 JSON 标签，避免缺失 capture-stream 映射时按 task 序号猜配 | raw Model/Stream/Task 与显式回放 flow；不把端点区间解释为独占 kernel | [代码与核验](scripts/vllm-task-tag-diagnostic/README.md) |
| SGLang gen67 | 公共 Event 层端点、显式 requestID/RID 正式回放筛选；确切 graph 的 EOF 后清理 | 保留 startup、mixed 和未知；CPU stub 不证明真实设备同步或线程竞争 | [代码与核验](scripts/sg-public-event-diagnostic/README.md) |

两份组件共用部分历史 helper，但各自保留完整 manifest 及原布局；不是两个独立实验。组件 README 中的“本包”指各自组件，历史代次事实不被本次整合改写。代码入口不包含完整服务启动器、SDK 借槽、权重、原始 profile、SO 或 transport。bootstrap-reference 只供查阅，不得直接加入 PYTHONPATH。完整源码范围与逐文件 SHA 见 [代码清单](CODE_INDEX.json)。

在仓外两个各自不存在的绝对输出目录运行：

```sh
python3 -I -B -S scripts/vllm-task-tag-diagnostic/check_cpu.py --output /absolute/new/vllm-cpu
python3 -I -B -S scripts/sg-public-event-diagnostic/check_cpu.py --output /absolute/new/sg-cpu
```

这些检查只需标准库；只验证原字节、静态任务依赖和 CPU fixtures。不会证明 aarch64/910B1 设备行为、两框架输出等价、纯框架因果或完整204完成。构建入口本次未执行，SO 字节等同与可移植构建未验证。真实服务/模型/设备工作仍需确切归属及 performance-workflow.lock。

当前相同 payload 的两份插桩短诊断文本不一致；整体 greedy/质量与差异根因未闭合。四 rank 不相加为墙钟；HCCL 等待和别名去重边界保留。基线 CANN9.2/9.0、卡组及 KV 混杂仍在；未采用优化。

本目录是 main 的候选记录，代码放 scripts；index 候选只在 main 事实确定后同源同步到 code。实际 R ID、模型登记、main/index 全仓检查、最终204地图及浏览器 QA 尚未完成。占号/提交前须刷新实时 AGENTS/README/R_TEMPLATE、账号权限及全部 refs；本次固定规则来自第29代快照，不能用来声称发布时规则已刷新。
