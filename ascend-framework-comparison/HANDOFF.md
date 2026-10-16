# 当前状态：两套2P2D，旧SGLang已停止（2026-10-16）

先看 [服务器分工](SERVER_ALLOCATION.md)、[资源JSON](RESOURCES.json)、[资产表](ASSET_READINESS.md) 和 [停机证据](inventory/20261016/stop-and-reallocation.json)。当前状态覆盖上一提交的4P2D规划/P47冲突。

- 原GLM53 vLLM：P30/31，D43/45；SGLang：P34/35，D36/41。Flash仍为vLLM14P/15D、SGLang46P/47D。备用32/33/37，38和42排除。
- 通过父审查的本机ZCode DeepSeek精确launcher停止33个自有SGLang相关容器。全部容器对象/镜像/模型/配置/日志保留；43AISBench和K8s声明保留。父独立复核16台256die健康OK、无NPU进程，42核查在其排除前完成。
- 42SSH认证原先不可用。通过一个临时K8s管理Pod已验证宿主机root、磁盘与真实npu-smi，不是宕机。随后用户排除42，已撤除本次管理Pod，未改SSH/宿主机文件/原有作业。不要再将42加入新组或恢复该Pod。
- 新2P2D/Flash尚未部署、无新模型请求/性能成绩，所有定时任务保持暂停。禁止因“服务停止”触发旧自动恢复；这是用户授权的主动停用。
- 当前15台可用池的原GLM53本地NVMe权重清单已核；45只做清单/小文件SHA，没有新的索引头探针。Flash目标已查目录仍未发现Flash权重，新固定推理镜像也未导入。B103完整tar保留Mac；另一网络166Flash资产只有本地历史证据。
- 原模型P从64die变为32die，不能重放旧EP64/TP64。新TP/attentionDP/EP、KV/MemCache容量、图/MTP与SDMA实际链路需重新冻结并做正确性门控；备用节点不默认作为额外缓存资源。
- 本轮公共合同升v2只固定新资源/2P2D范围；C16/32/64四轮、独立74400前缀、实际80000/600、预热排除、自然缓存保持。旧4P2D不进入新2P2D框架收益计算。

接手入口：[Flash](tracks/flash-adaptation/START_PROMPT.md)、[原GLM53 2P2D](tracks/glm53-2p2d/START_PROMPT.md)。两个任务共享合同，不自动创建/消息对话。实际构建/部署/模型/性能仍须父审查固定SHA后本机ZCode DeepSeek执行一次。

## 历史材料边界

archive/20261016仍保存旧六节点4P2D日更27b、官方配套SDK1.3/MFLOCAL、MemCache640GiB、P1/D5、D图、buffer_only、SMG1.13和真实源码/配置/映射SHA。原自然暖C64的252成功/4超时/3成功缺缓存明细仍不合格；旧强制L3的448合格结果不是自然暖或新2P2D基线。旧记录不覆盖当前停机和重分配。

inventory/20261016/readiness.json保留上一轮15宿主机只读快照与当时P47冲突；最新以stop-and-reallocation.json为准。42后来只读发现的资产缺口不再是本批部署条件，因为用户已排除该节点。
