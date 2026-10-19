# Loop 目标：M017/M018 NPU长度镜像与staging真实消费者支持审计

- Loop ID：`L007-npu-metadata-staging-support`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T17:44:51Z`

## 目标/假设

实际NPU attention/graph消费者及内存能力声明能区分必要MTP长度、可选CPU镜像与CUDA-only异步路径，避免把未支持的删除或复用候选带入Ascend主线

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract

## 接受条件

固定安装源码与哈希，逐字段记录producer/consumer/gate/生命周期；读取可用的只读resolved配置作为观察线索，未知保留；支持性谓词仅CPU验证，NPU事件/拷贝与PD收益不外推

## 否定条件

缺少consumer闭合或只据默认开关/目录名推断NPU支持则INCONCLUSIVE；不把观察服务当本任务资源或基线，不启动设备工作

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
