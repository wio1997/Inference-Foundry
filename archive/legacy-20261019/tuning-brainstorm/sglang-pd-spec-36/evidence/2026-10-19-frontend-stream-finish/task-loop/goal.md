# Loop 目标：M033 默认累计文本到 SSE 的通知与结束契约

- Loop ID：`L031-frontend-stream-finish`
- 模式：`correctness`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-19T03:55:40Z`

## 目标/假设

默认累计文本和可选增量模式在通知合并、finish及abort-shaped输出下，应向OpenAI流式层交付无丢失、无重复的文本/IDs，并有明确的请求表和abort派发边界。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- lifecycle
- tokenizer-cpu-c2
- tokenizer-cpu-c3

## 接受条件

原样完整方法在异步组件/可行的HTTP协议夹具中保持文本、IDs、finish、usage及通知对应，保留全部失败和未执行边界；原生PD+MTP未测时不作全链路接受。

## 否定条件

通知合并或模式桥接造成token/text重复丢失、finish错误，或完整路径因未满足依赖/前提无法执行。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
