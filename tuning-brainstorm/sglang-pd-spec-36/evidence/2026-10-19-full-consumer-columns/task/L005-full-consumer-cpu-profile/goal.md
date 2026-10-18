# Loop 目标：M035 CPU链补齐真实consumer及碎片波动归因

- Loop ID：`L005-full-consumer-cpu-profile`
- 模式：`baseline`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T17:10:43Z`

## 目标/假设

在页号元数据降为range后，描述符tuple构造与真实_transfer_data的zip/list化形成剩余CPU成本；thread CPU和GC记录能区分部分wall停顿

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`

## 目标用例

- descriptor-cpu-pp1

## 接受条件

保留L004原方法与range候选，以实际_transfer_data方法及no-op engine保持相同native调用输入；记录group、构造、转置的CPU/wall及GC，A/A和交错样本，解释碎片噪声并决定下一候选

## 否定条件

地址/长度/顺序/调用次数或错误语义不同即失败；无法区分调度/GC和有效工作则INCONCLUSIVE，不按中位数删离群点；不外推NPU和PD

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
