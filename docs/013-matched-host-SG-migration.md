# 同一组硬件迁移到 SGLang：交接、失败和修正

四档 vLLM G4 首次正式冷热以及同批次 P/D 十六 rank 模型设备 profile 完成后，原两台 GLM 主机拉取固定 SGLang A2 镜像并核对完整镜像ID和arm64架构；运行时tar的SHA为f98cfbd3e825a8f08431541fe342053a84639a84781b002253e070b9fad11a8a。复用此前已运行的 G33/G32 模型适配源码、权重和配置，将网络角色映射到同一对 GLM 主机。

可从固定底座重建此运行时。 [G32 六项适配、补丁和复现步骤](038-GLM53-G32-adaptation-and-reproduction.md)

交接前两端队列均为0，成功请求计数均513。仅向本任务原控制器写stop.request；核对容器退出、控制器消失、所有八卡无进程、所有者释放和锁可重获后，才启动新代。原容器、原始正式测量、原始profile与失败现场均保留，没有重发正式数据。

G164在权重后处理时失败：控制器创建了cache父目录，遗漏ASCEND_CACHE_PATH子目录，CANN初始化返回500001并明确报目录不存在。两端终态均failed_released、全部八卡进程清理。G181仅修正控制器的运行时目录创建，并在模型加载前从实际容器核对所有目录存在且可写；保留镜像、运行时、模型、TP8/EP8、BF16 KV、完整263296上下文、MTP3、P/D分块和D完整图配置。

- [固定镜像取得](../evidence/matched_SG_migration/G169_SG_stage_observe_2201_parsed.json)
- [安全交接 P](../evidence/matched_SG_migration/G172_matched_SG_start_2201_parsed.json)
- [原始失败 P](../evidence/matched_SG_migration/G180_matched_SG_brief_2201_parsed.json)
- [原始失败 D](../evidence/matched_SG_migration/G180_matched_SG_brief_2202_parsed.json)
- [修正后的实际目录门槛与加载状态 P](../evidence/matched_SG_migration/G187_matched_SG_brief_2201_parsed.json)
- [修正后的实际目录门槛与加载状态 D](../evidence/matched_SG_migration/G187_matched_SG_brief_2202_parsed.json)
- [修正控制器](../code/matched_SG_G181/node_controller.py)
- [真实生成与原生PD功能门槛源码](../code/matched_SG_migration/sg_G181_native_PD_functional_g182.py)

这批记录的末次观察仍在模型加载阶段，不能据此称服务就绪或开始正式计分。SGLang native max_running_requests=16，vLLM P/D native max_num_seqs=64/32；客户端均为C16，这个引擎内部槽位差异须在最终可比性说明中披露。充分独立预热、首轮四档冷热、匹配profile和优化差异地图仍未完成。
