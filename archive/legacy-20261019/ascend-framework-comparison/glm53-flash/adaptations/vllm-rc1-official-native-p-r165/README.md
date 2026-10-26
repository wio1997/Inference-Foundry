# Flash vLLM RC1：实际 R165 官方原生 P 路径

固定090镜像，original native KDA3文件/pool_worker；11个ROoverlay与R165实际导入一致。相较旧17-overlay候选撤3自写KDA、D-onlypoolworker和2图观察器；indexer用固定官方22730f46版本，较image只改INDEX32 valid page mask。P2TP8按新官方Pflags，HCCS+hccn.confRO，20GB/worker原生MMC。

实测两个API ready/16HTTP200/原生0+Meta0/idle8；27实际put/knownkey/full readback共181043200B精确，P1native externalhits1920、P实际MTP105draft/40accepted。保留strict SDMA原始扩展入口和同inodeSMEM别名；SDK二进制未变。

**整体验收仍FAIL**：7/16 strictnumeric失败、3个same-path repeatID漂移、缺shutdownqueue观察。R169进一步原生TP16/确定性/无MTP/无cache控制也有两例LP>.25，全部repeat稳定；不能归咎于唯一优化项，也不能说官方模型语义答案错误或不支持。native框架服务可运行与自有冻结数值门通过分别记录。

无D/PDpayload/GraphD/80K600/formal性能资格。qualification观察器会CPU复制/哈希，不得用于性能。后续须独立验证flag0、真实缓存预热和原生DP数据面；未知tier/perrequesthits保持null。原始保护输入/tensor/完整日志未公开，不重放历史一次性launcher。
