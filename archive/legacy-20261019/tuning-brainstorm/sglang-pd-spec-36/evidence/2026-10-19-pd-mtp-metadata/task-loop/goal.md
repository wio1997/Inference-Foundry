# Loop 目标：M031 PD推测metadata的实际CPU消费者差分

- Loop ID：`L016-pd-mtp-metadata-cpu`
- 模式：`correctness`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T20:15:42Z`

## 目标/假设

MetadataBuffers容量topk16、clone及DSA sentinel必须与实际eagle_disaggregation有效width和FutureMap交接一致；slot复用可能保留无效尾部，但是否可见取决于真实消费者。先验证实际CPU类/消费者，不以字段存在证明原生PD。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- lifecycle

## 接受条件

先收集实际eagle_disaggregation实现。用完整未修改模块CPU路径或明确受限的抽取，验证set/get完整字段、clone独立性、slot复用、topk16到配置1的有效数据、DSA有/无/全负标记和bootstraproom；观察到的反例先查可达性，不冒充NPU传输。

## 否定条件

模拟输出替代实际消费者；忽略旧slot尾部/None字段/请求顺序；将CPUuint64bootstrap路径当NPUint64；修改installed类或触碰未分配NPU/模型服务。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
