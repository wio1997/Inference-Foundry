# 原GLM-5.3：SDMA四机2P2D框架比较

最新资源：vLLM P30/31、D43/45；SGLang P34/35、D36/41。备用32/33/37，38/42排除。原有SGLang已按用户授权停止并保存；当前未启动新2P2D模型或压测。

先用同一原GLM53 W8A8权重和本地NVMe副本建立vLLM SDMA正确性参考及新基线，再同合同验证SGLang。2P共32die，2D共32die；新并行参数/缓存容量独立冻结，不能照搬旧4P EP64或把旧4P2D成绩当对照。

实施见 [PLAN](../../PLAN.md)，计量遵循 [BENCHMARK_CONTRACT](../../BENCHMARK_CONTRACT.md)，资源/资产看 [服务器表](../../SERVER_ALLOCATION.md) 与 [资产表](../../ASSET_READINESS.md)。本目录维护STATE、后续runs与原始证据索引。父审查精确launcher后经本机ZCode DeepSeek执行；不启动旧broker/定时，不改系统SDK/驱动/BIOS/时钟。
