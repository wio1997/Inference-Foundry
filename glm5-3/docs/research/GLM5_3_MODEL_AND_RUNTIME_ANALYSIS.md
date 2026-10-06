# GLM-5.3模型与Runtime核验

当前目标为GLM-5.3 W8A8标准P/D分离；唯一权重路径是两机`/data/tiankuan/wio/GLM-5.3-w8a8`，新服务模型名`glm-53`。2026-10-06用户正在上传，两机只读确认是独立实际目录、并非旧权重路径或symlink；目录中已出现量化safetensors、quant_model_description.json、chat_template.jinja和generation_config.json。检查时尚无config.json，不能推断上传完成、加载成功或完整模型身份。

上传完成后按实际artifact定向核验：

| 事实 | 权威输入 | 当前结论 |
| --- | --- | --- |
| 架构、主层/专家、Full/Shared、MTP与上下文 | 新config.json及checkpoint index | unknown，不能继承旧模型数值 |
| W8A8导出与实际量化范围 | 新量化描述、index及实际loader | 目标W8A8；实际加载与cache dtype待核验 |
| tokenizer/template、特殊token、tool/thinking语义 | 新tokenizer文件与chat_template、原生renderer/parser | 待核验；不继承历史token计数/输出语义 |
| artifact完整性 | 上传完成确认，config/index/tokenizer、全部index引用分片 | 待核验；目录或分片数量本身不足 |
| 模型实现、设备布局、KV/MTP/Graph/PD合同 | 实际安装路径、版本、启动argv与受控功能证据 | 待核验；不回退旧模型 |
| fit、完整功能、E2E及收益 | 新模型同口径correctness与matched重复完整PD E2E | unknown；Current=None，无PERF_KEEP |

不在上传过程中扫描/读取/hash权重payload，不触发自动加载或benchmark。现有`glm52-single`与`glm52-pd/deploy`是资源标识，不说明目标仍为旧模型。恢复路径见[ENVIRONMENT_RECOVERY](../../ENVIRONMENT_RECOVERY.yaml)、[RECOVERY_INDEX](../../RECOVERY_INDEX.md)；研究顺序见[PLAN](../../PLAN.md)。

历史[旧模型分析](GLM5_2_MODEL_AND_RUNTIME_ANALYSIS.md)、原Run及来源hash保留，只在安装源码与相应机制条件仍匹配时作为线索，不替代新模型结构、支持与性能证据。
