# 原 GLM-5.3 资产与模型身份

主权重：源1 `172.16.12.1:/data/wio/GLM53/GLM-5.3-w8a8`；本项目目标入口`/data/wio/models/GLM-5.3-w8a8`。预期model_type=`glm_moe_dsa`，architecture=`GlmMoeDsaForCausalLM`。193文件、764,240,829,290bytes。默认只挂载这个明确目录，禁止使用仅到`/data/wio/GLM53`的模糊目录或自动选首个GLM权重。

本项目vLLM候选为rc1 image090eeee6，SGLang候选B103 imagec5cf2deb；完整tag/ID/归档SHA看[RESOURCES](RESOURCES.json)。两框架具体节点的本模型资源证据看[RESOURCE_EVIDENCE](RESOURCE_EVIDENCE.json)，已过滤掉其他模型字段。资源到位只允许开始模型功能工作，不代表已有运行资格/性能。

主W8A8的193文件名/大小与config SHA已核，本次资源准备没有再次全量哈希764GB payload。

32/33补充的是**原GLM53 W8A8C8**：196文件764240832702bytes，源及两目标全部文件payload SHA通过，rc1已导入；[父独立验收](C8_RESOURCE_EVIDENCE.json)核实际imageID/本地文件map/config/量化说明与完整SHA清单；[源SHA](C8_SOURCE_SHA256.json)。路径`/data/wio/GLM53/GLM-5.3-w8a8c8`，别名`/data/wio/models/GLM-5.3-w8a8c8`。它未分配P/D，不自动并入2P2D，也不替换主W8A8。config.json可相同，量化说明/分片/模板不同；C8的88项attention/indexer INT8_DYNAMIC且部分层保留高精度，不能宣称全部缓存统一8bit或容量必然减半。它与主W8A8分别建run和精度基线。
