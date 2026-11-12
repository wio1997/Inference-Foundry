# prefill MC2：明确缓冲下限与一次对称修正

参考107确实启动8rank，主模型与MTP加载后，在初始化的CANN `moe_distribute_dispatch_v2` 触发通信缓冲断言。8rank安装回执均为prefill MC2开启、容量1792、A3策略FUSED_OR_CAPACITY。正式AISBench请求为0，容器已停止，资源已释放。

算子明确要求 **1,152,385,024字节（1099MiB）**，实际只有419,430,400字节（400MiB）。因此目前结论是缓冲配置不足，不是精度失败，也不据此断言MC2不支持该模型。

源码 `5fd2cd7e9c671c4cc9ab42cf4dec42a5b302b97b` 规定该有界prefill配置两侧一致使用1280MiB，超过算子报告下限。原生启动入口、参考进程环境和加载前校验同时保持一致；旧decode-only配置仍为400MiB。该值来自实际错误下限，不是参数网格，也不减少模型负载、KV缓存或生成长度。

109在模型加载前因原卡组不可用而停止，110未启动。111参考重新认领时原物理卡组已恢复，实际仍使用同一8die映射；112必须与111一致。当前111运行中，112未启动，**修正配置尚未通过硬件验收，没有新增性能成绩**。实际后续结果以状态页的更新为准。

源码还确认fused MC2与shared-expert overlap互斥，因此保留相同overlap路径，不把两者都打开描述为可行配置，也不宣称已经搜索所有共有配置的最优点。

原6/12小时预算与独有双档30%目标不变。最近成功同组实测仍为102/104的+9.06%/+19.73%；它们不能替代新MC2配置的匹配对照。

[缓冲失败、退出和8rank证据](../validation/shared-features-20261111/reference-g107-failure-verification.json) · [共有开关兼容边界](../validation/shared-features-20261111/shared-mc2-compatibility-scope.json) · [此前路线纠正](prefill-route-correction-20261112.md)
