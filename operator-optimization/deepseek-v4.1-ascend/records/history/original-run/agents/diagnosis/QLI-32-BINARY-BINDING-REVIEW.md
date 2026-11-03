# F12 的 32 组 Indexer：限定的二进制与属性绑定核查

沿用 `INDEXER-ATTENTION-EVENT-IDENTITY.md` 已选 EP3/EP8 × 两个 scope × 八个 index-source 角色层 32 组，使用其源文件中的 Indexer 精确 `startNs` 各自定位同卡 SQLite `TASK`，每个均唯一；然后以 `globalTaskId` 唯一查 `COMPUTE_TASK_INFO`，解码 `name`、`opType` 和 `attrInfo`。脚本先核验源文件及两数据库 SHA-256，以只读 SQLite 操作，不导入设备库、不执行 NPU 工作。脚本 `query_qli_32_binding_fields.py` SHA-256 `8221da5ae3ac39e4b5e6804461eb262c52d7c44e1a4720ec32f691c05b043f10`；结果 `qli-32-binding-fields.json` SHA-256 `e577e8491aa2f13803c9262dc36774f36c5ac7cf8c30489e5a9b41ccda7c32ca`，其中记录每条 task 和 DB 摘要。

32 条的名称和类型均为 `VllmAscendQuantLightningIndexerV2`，其 `attrInfo` 解码均为空字符串；`blockNum=24`、`mixBlockNum=48`、`gridDim=blockDim=36`、`modelId=4294967295`。在各自 DB 的 TASK 中，32 个 `globalTaskId` 各出现一次。这些仅是已导出字段，不是编译二进制摘要、tiling key、选中模板实例、逐行 sequence 或 tensor 数据。同名及这些数字相同不能证明实际采用被冻结的 arch22 源码或执行相同分支。空字符串只说明该导出字段没有可读属性值，不证明原始采集、其他记录或运行时没有这些属性。

本批样本不能使用另一轮 EP8 层 24、model 49 的 QLI `attr_N` 来推断自身 mask 或 candidate 模式；那里有属性字符串，且 attr_N 的版本契约和二进制来源仍未闭合。对于本批 32 组，现有静态源码所述可见性等价式只能作为**条件性候选**，无法证明每行相等或删除过滤可行。下一步若已保存原始资料中可找到 task→执行 binary hash/编译版本→源码，以及同 task 的模式和 L、Q、i、p 值，逐项核验；否则列为缺失，不额外启动 NPU 负载，也不以空字段推断其为关闭或默认值。
