# 假设记录

## H001 - ACTIVE

已验证的TensorList graph observer可在隔离诊断入口中读取实际FusedExpertsResult计数、mask、topk和调用序号，避免capture常量及跨层共享buffer覆盖误读；不改变模型返回值。
