# 原 GLM-5.3 资产与模型身份

主权重：源1 `172.16.12.1:/data/wio/GLM53/GLM-5.3-w8a8`；本项目目标入口`/data/wio/models/GLM-5.3-w8a8`。预期model_type=`glm_moe_dsa`，architecture=`GlmMoeDsaForCausalLM`。193文件、764,240,829,290bytes。默认只挂载这个明确目录，禁止使用仅到`/data/wio/GLM53`的模糊目录或自动选首个GLM权重。

本项目vLLM候选为rc1 image090eeee6，SGLang候选B103 imagec5cf2deb；完整tag/ID/归档SHA看[RESOURCES](RESOURCES.json)。两框架具体节点的本模型资源证据看[RESOURCE_EVIDENCE](RESOURCE_EVIDENCE.json)，已过滤掉其他模型字段。资源到位只允许开始模型功能工作，不代表已有运行资格/性能。

主W8A8的193文件名/大小与config SHA已核，本次资源准备没有再次全量哈希764GB payload。

本项目只使用上述普通W8A8权重；两框架挂载同一模型目录并逐项核对量化说明、索引和模板。此前32/33的C8传输已移出本项目，已有文件不删除，不安排额外精度变体测试。
