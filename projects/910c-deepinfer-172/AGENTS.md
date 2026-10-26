# 172 native DeepInfer scope

- 本目录仅对应用户明确指定的172；设备16逻辑/8物理，DP2 TP8 EP16。
- 当前固定吞吐验收完成；不要为确认而重新跑基准、重启服务或加入无关测试。
- 先读README、STATUS、runtime和源码/镜像身份。668.35token/s是重复上下文固定profile的吞吐结果，TTFT仍明显慢于vLLM。
- 原vLLM数据不改写；原容器STOPPED并保留。任何恢复先备份其会被重定向覆盖的log。
- 不停别人任务，不改170；项目数据全部在wio。权重、容器根和驱动只读；仅本次172 privileged已获授权。
- 简单机械工作仍由本机ZCode已配置DeepSeek执行，主控负责设计和独立验收。公网页面不使用浏览器。
