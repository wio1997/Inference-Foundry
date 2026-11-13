# C4主归因：在固定vLLM内隔离单步批处理策略

这是[机制报告](performance-mechanism-audit-20261113.md)§8的源码候选，不是新性能成绩。原HTTP cohort-only方案只能测附加准入，不能取消DeepInfer本身的同型步。主裁决改为在同一vLLM内改变批次组织，保留模型、SDK、算子数学、异步、MTP及缓存配置。

## 1. 唯一策略变量

两臂都使用同一`PhaseControlledAsyncScheduler`、同一helper及观测，`ATTRIBUTION_PHASE_POLICY=0/1`是唯一策略差异。关闭时原选择逻辑继续运行并允许混合；开启时，将prefill、普通decode、推测decode分开成步。后两者区分是因为MTP的spec/non-spec混合会影响KDA分组和图资格。

该干预的效应属于完整的“计算契约分别成步”策略，不能拆成PD同型与spec分组各自的独立收益；也不能直接等同于DeepInfer原24.13%的全部贡献。它检验已观察到的机制链能否在参考框架内改变端到端表现。

两臂共同增加的计划入口片段如下，尚非实际启动记录：

```bash
--scheduler-cls phase_scheduler.PhaseControlledAsyncScheduler
```

```text
ATTRIBUTION_PHASE_POLICY=0    # 关闭干预的控制臂
ATTRIBUTION_PHASE_POLICY=1    # 按计算契约分别成步
ATTRIBUTION_PHASE_STATS_DIR=<该臂已存在的绝对证据目录>
```

helper放入同一只读overlay的PYTHONPATH；其余启动参数沿固定138参考，包括8die、TP8/EP8、DP1/PP1、context2048、max-seqs8、1792 token预算、4GiB普通缓存、MTP5、图与共有优化。仅测C4，沿原4工程+2预热+32正式请求，不加prompt或额外质量门槛。模型数学和量化实现不变。

## 2. 在哪里生效

`phase_scheduler.py`继承固定SDK的`AsyncScheduler`；不改已安装SDK文件。根据固定core源文件SHA256，生成仅含附加门禁的`Scheduler.schedule`方法给这个子类使用。core、async及config三份源文件都有身份门禁；不匹配即拒绝加载。

门禁有两处：

1. RUNNING请求完成本步可执行token数计算后、`allocate_slots`之前，异型请求留在原队列，本步不被选中。
2. WAITING请求计算prefix命中及计划跨度后、`allocate_slots`和入队/进度更新之前，不合本步类型则停止本轮接纳。常规prefix查找可能已更新边界提示；没有事后撤销分配、吞掉输出或回滚位置计数。

一个step第一次成功选择的计算类型决定其后续选择。分类使用SDK主机侧进度、placeholder、spec计划和prefill标记；prefix末尾被原SDK补成spec宽度时按spec_decode分类。没有读取设备tensor、重算logits或改变接受/拒绝采样。

原cache分配、抢占代码、已选请求的打包和`_update_after_schedule`保持在原位置。诊断限定于无connector、无多模态、无抢占的固定文本负载；遇不覆盖的抢占即保留失败，不当成通过。未选RUNNING请求的后续重新接入，仍由原`prev_step_scheduled_req_ids`与`all_token_ids`协议处理。

分别成步不自动保证FULL图：宽度、初始状态及其它图资格仍由原SDK判断。必须测到实际路径，不能把策略标签当作worker图模式。

## 3. 观测与关闭

每个非空调度step记录最多2048条主机事件：请求ID哈希、计算类型、computed/prompt/scheduled/spec行数及单调时钟。空步仅累计计数。策略关闭时也执行同一观测；无逐step文件写入。

子类调用原`shutdown`后写`phase-policy-<pid>.json`；原shutdown失败则记录错误并继续抛出。真实关闭回调和实际SDK类尚未执行，不能预写“回收已通过”。缺闭合日志、事件丢弃、worker异常均使该轮归因无效。

## 4. 已有源码与主机证据

[检查结果](../validation/performance-attribution-20261113/phase-policy-source-checks.json)包含：

- 两个门禁都在对应分配之前；记录检查位于原异步进度更新之前。
- 从修改方法中去除新增私有节点，AST与原方法完全一致，原控制代码未被替换。
- SDK源文件变化会被拒绝。
- 主机元数据覆盖prefix命中、异步placeholder、waiting spec补宽、关闭臂保留混合、开启臂拒绝异型输出。
- 分类及观测不修改请求对象；拒绝把非主机标量转换为整数。

这些检查没有导入实际SDK、执行完整scheduler或运行模型；不代表真实KV生命周期、async在途状态、worker描述符或性能通过。[源码与版本清单](../validation/performance-attribution-20261113/phase-policy-preparation.json)给出helper和生成方法的SHA256。

## 5. 上机前与裁决条件

先取得中央池完整8die租约、固定两个臂的执行包并分配有界预算。当前未取得租约、未部署、未分配新NPU预算。只在该前提成立后做实际SDK导入和工程门槛，避免没有对照价值就加载模型。

必须核对：

1. 两臂实际都选择同一自定义AsyncScheduler，async仍开启；源码、helper、镜像、扩展、权重、参数及数据身份一致，策略值回执为0/1。
2. 控制臂确实执行原选择，开启臂门禁有实际效应；无未覆盖抢占、无状态/placeholder错误、无事件丢弃，关闭和资源释放完整。
3. 正式阶段的混合组成、target/draft图和普通调用必须对应客户端窗口；不能用全38请求计数冒充32正式请求，不能将policy类别当作设备图证据。
4. 32/32正式成功，32768输入/8192输出；保存原始输出及文本差异、SQLite时间戳、真实argv和依赖身份。输出异常的快不能接受。

若同型策略减少混合/普通执行，且C4端到端长间隔同步下降，支持这条机制的因果作用；收益使用新两臂的秒数，不能搬用旧参考分母。若只改路径不改耗时，否决其为主要关键路径收益的主张。若控制臂本来不混合、实际策略无效或SDK工程失败，该轮不能回答根因，保留失败，不改参数继续凑数字。

当前没有新速度数据。噪声未标定时不将小差异当作稳定改善；必要设备时间在非计分机制窗口采样，不能以算子累计耗时替代端到端关键路径。

## 6. 代码

- [批处理策略与源码变换](../validation/performance-attribution-20261113/phase_policy.py)
- [固定SDK AsyncScheduler子类](../validation/performance-attribution-20261113/phase_scheduler.py)
- [主机源码/状态契约核对](../validation/performance-attribution-20261113/verify_phase_policy.py)

核对脚本需要`fixed-sdk/scheduler.py`为清单中SHA256对应的实际SDK源文件。该文件从用户固定镜像读取，不在公开仓库另存一套SDK源码；使用方须从其有权限的相同镜像获取。代码仍是诊断候选，不能脱离固定SDK和原执行包当作已验证服务运行。
