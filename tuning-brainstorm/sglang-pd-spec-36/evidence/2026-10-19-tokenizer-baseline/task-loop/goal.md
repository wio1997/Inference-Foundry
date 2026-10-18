# Loop 目标：M001 动态批处理参数的CPU基线与噪声

- Loop ID：`L011-tokenizer-batching-cpu-baseline`
- 模式：`baseline`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T19:15:01Z`

## 目标/假设

相同2/3请求与固定2ms窗口下，max_batch_size改变CPU调用次数和队列等待，收益取决于实际编码成本；先测A/A和交织A/B，不用吞吐假设代替测量

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- tokenizer-cpu-c2
- tokenizer-cpu-c3

## 接受条件

使用M001-v1固定输入、同镜像CPU依赖视图和实际SGLang类，逐用例比较max_batch_size1与32；值/取消保护保持，clean wall与process CPU、到达时序及cgroup限流记录；效应需与A/A噪声分开，明确CPU-only范围

## 否定条件

改变文本/模板/请求数量或窗口制造收益；把包导入启动计时当稳态改善；输出差异、观测限流或噪声不支持结论；从CPU配置研究外推PD或NPU收益

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
