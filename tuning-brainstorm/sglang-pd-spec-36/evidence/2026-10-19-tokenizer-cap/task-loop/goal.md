# Loop 目标：M001 固定封闭批次cap与尾等待对照

- Loop ID：`L012-tokenizer-cap-timeout-cpu`
- 模式：`optimization`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T19:34:50Z`

## 目标/假设

L011短输入cap32多约2ms但CPU无分离；固定2/3个同时到达请求，将cap32改为预先固定的2/3可能在相同实际批次下省掉尾等待。已有配置对照，不是全局在线策略。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- tokenizer-cpu-c2
- tokenizer-cpu-c3

## 接受条件

真实原类、相同输入/2ms窗口/CPU隔离/配对AAAB，capN值与32完整相同；short wall改善超过AA且无CFS混淆；long单独报告并保留反证。仅CPU局部条件接受，禁止更新整体serving baseline。

## 否定条件

改变文本、窗口、计时或事后选择行制造收益；值差异；CPU/墙钟效应与AA重叠；将已知N用于未知在线流或宣称PD/NPU改善。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
