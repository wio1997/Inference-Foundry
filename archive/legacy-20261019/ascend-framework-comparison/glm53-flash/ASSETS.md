# GLM-5.3-Flash 资产与模型身份

主权重：源1 `172.16.12.1:/data/wio/GLM53/GLM-5.3-Flash-w8a8`；本项目目标入口`/data/wio/models/GLM-5.3-Flash-w8a8`。预期model_type=`glm5_next`，architecture=`Glm5NextForConditionalGeneration`。97文件、332,970,562,139bytes。默认只挂载这个明确目录，禁止使用仅到`/data/wio/GLM53`的模糊目录或自动选首个GLM权重。

本项目vLLM候选为rc1 image090eeee6，SGLang候选B103 imagec5cf2deb；完整tag/ID/归档SHA看[RESOURCES](RESOURCES.json)。两框架具体节点的本模型资源证据看[RESOURCE_EVIDENCE](RESOURCE_EVIDENCE.json)，已过滤掉其他模型字段。资源到位只允许开始模型功能工作，不代表已有运行资格/性能。

Flash四台97文件payload SHA已逐文件通过，源manifest46a48be8f7cd6000d260f5829089651cdca36fd26dca78e2a391d35d3ea590e2；父独立复核文件/大小、config与本地NVMe，未第二次重读全部333GB。原模型764GB/C8、旧MLA层数和MemCache640GiB部署不能充当Flash权重或Flash混合状态验证。
