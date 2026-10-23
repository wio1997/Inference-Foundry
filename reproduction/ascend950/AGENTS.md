# Ascend 950DT 复现材料

本目录面向 DeepSeek-V4.1-Flash、8×950DT、DP8/TP1/EP8/CP1。根目录历史 910B3、GLM 和 TP8 任务不适用于这里。

先读本目录 README、performance-20261021/STATUS.json 和实际启动脚本。保留原始测试合同、输入哈希和失败记录。部署补丁、原生框架优化、诊断探针与未完成实验要分别说明。不得把 profile 累计时间当作端到端加速收益。

仅操作当前任务明确指定的容器，不操作 dpinfer-build 或 deepseekv4-fp8-a5-2。不要提交凭证、SSH 连接配置、权重或原始请求响应。公共网站使用 HTTP/API/Web Search。
