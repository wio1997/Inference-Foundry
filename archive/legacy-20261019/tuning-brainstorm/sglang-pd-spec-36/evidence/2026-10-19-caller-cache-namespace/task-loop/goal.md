# Loop 目标：M036 请求到缓存的身份合同闭合

- Loop ID：`L035-caller-cache-namespace`
- 模式：`correctness`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-19T05:59:47Z`

## 目标/假设

不同adapter/调用方namespace在真实请求归一化、carrier、Req与UnifiedTreeCore组合后应仍隔离；流式复用应拒绝不满足身份稳定前提的请求，或找到明确上游限制。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- lifecycle
- c2-short
- c3-long

## 接受条件

实际执行原始请求归一化/registry-carrier/Req与完整原Core，验证受保护identity controls并确认或否定上轮局部反例的可达边界；修补仅在明确合同后作为离线候选评估。

## 否定条件

合法请求跨adapter/base返回相同缓存行或流式namespace改变仍复用旧记录且无上游拒绝；保留反例，不把其他通过检查当作修复。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
