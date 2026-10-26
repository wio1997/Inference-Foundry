# Loop 目标：M001 tokenizer与模板身份及输入合同准备

- Loop ID：`L010-tokenizer-input-contract`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T18:44:18Z`

## 目标/假设

固定tokenizer文件、模板、后端版本和真实SGLang入口是冻结同条件PD输入的前提；CPU侧可以先建立该身份和差分，不能由模型名推断token流

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract

## 接受条件

固定安装入口与tokenizer/模板文件哈希、模型候选身份和依赖版本；检查实际加载及动态批处理调用；之后用任务CPU容器核验token流，不加载模型权重或NPU

## 否定条件

只按model名称或token数量认定输入相同；改变模板/特殊token而当作优化；以CPU分词收益代替PD端到端收益

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
