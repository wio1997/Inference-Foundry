# Loop 目标：M025 NPU DSA seed复用与必要索引工作

- Loop ID：`L017-npu-dsa-seed-work-contract`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T20:33:05Z`

## 目标/假设

实际模型index_share=True和DSA宽度2048只是前提；NPUindexer是否复用seed、跳过TopK或重算取决于真实draft/verify分支、context长度和batchfallback。512/32768案例必须分别建必要工作合同，不能套CUDA/HIP fusedTopK逻辑。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- c2-short
- c3-long

## 接受条件

固定commit追踪EAGLEconsumer的DSAseed到NPUindexer和自定义op调用，核对Indexer/KPool选择与短/长gate；给出可执行最小CPU/原生验证及工作量边界，严格区分source/CPU/native/perf。

## 否定条件

由index_share字段直接称NPU复用；将CUDA remap用于Ascend；关闭MTP或减少选中token换收益；未经同条件测量宣称跳过/加速或物理极限。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
