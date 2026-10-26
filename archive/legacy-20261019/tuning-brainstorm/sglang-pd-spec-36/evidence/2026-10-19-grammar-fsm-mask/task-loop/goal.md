# Loop 目标：M032 grammar 编译缓存、状态隔离与接受回滚

- Loop ID：`L032-grammar-cache-state`
- 模式：`correctness`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-19T04:28:42Z`

## 目标/假设

在固定源码和同一GLM5.3 tokenizer下，现有grammar缓存应复用不可变编译产物并给请求独立匹配状态；接受、回滚、EOS和批量mask更新应保持状态对应。能力谓词不替代真实FSM和Ascend支持证据。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- lifecycle
- tokenizer-cpu-c2
- tokenizer-cpu-c3

## 接受条件

追踪Ascend/PD/spec实际选择边界，使用可行的真实grammar库执行原样后端方法与独立正确性检查，保留不支持或失败；不将CPU编译/mask当原生采样或性能。

## 否定条件

缓存命中复用可变状态、请求间串扰、rollback/mask/EOS不一致，或后端/目标组合不可执行。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
