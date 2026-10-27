# 172 native DeepInfer scope

- 本目录仅对应用户明确指定的172；设备16逻辑/8物理，DP2 TP8 EP16。
- 当前固定吞吐验收完成；不要为确认而重新跑基准、重启服务或加入无关测试。
- 先读README、STATUS、runtime和源码/镜像身份。668.35token/s是此前v8的重复上下文固定profile吞吐结果，TTFT仍明显慢于vLLM。
- 原vLLM数据不改写；原容器STOPPED并保留。任何恢复先备份其会被重定向覆盖的log。
- 不停别人任务，不改170；项目数据全部在wio。权重、容器根和驱动只读；仅本次172 privileged已获授权。
- 简单机械工作仍由本机ZCode已配置DeepSeek执行，主控负责设计和独立验收。公网页面不使用浏览器。

- 此前接受版本`prefill-wide-v12`：997.90token/s、TTFT1838.35ms。具体参数见STATUS指向的runtime，旧数值仅作历史；全部原始数据在wio。

- 当前接受v21：固定重复文本热身/C8/32k输入/1k输出/两轮为815.25ms、1312.32token/s，16/16完整输出一致；模型workerv20/APIv21，实际KV命中89.99%。额外精确文本输入ID缓存已热，新文本仍付分词成本，关闭缓存v20为1013.35ms。当前STATUS与20261027-gap-closure优先；不为交付重跑/重启，全部原始基线保留。
