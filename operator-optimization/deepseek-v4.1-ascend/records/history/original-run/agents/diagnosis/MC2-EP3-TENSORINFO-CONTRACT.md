# EP3 历史 tensor_info：解析契约与目标关联边界

针对 [db852 路由元数据](MC2-DB852-ROUTER-METADATA.md) 中 EP3 的保存采集，使用该 stage 容器已安装的 CANN 9.2.0 MindStudio `TensorAddInfoBean` 和解析器代码，只读 5 个 aging、1 个 unaging 原始 `tensor_info` 切片。已安装解析器代码的文件哈希和字段摘录由 `diagnostic-20261029/cann-tensor-bean-source.json`、`cann-tensor-parser-source.json`、`cann-tensor-base-bean.json` 保存；仍未证明切片生产者和安装解析器构建版本完全一致。解析字段包括 level、type、线程、时间戳、node ID、输入输出格式、数据类型和形状；每个 tensor 描述符由输入/输出类别、format、dtype 和形状组成。该解析器没有公开张量元素的解码字段，也没有直接提供 device globalTaskId。它按 timestamp、thread、node 合并记录；本次对 aging 和 unaging 分开应用相同聚合键。

全部 35,713 条 256 字节记录通过 magic、level、tensor 数量范围与 bean 解码检查，聚合得到 18,186 组；这些格式检查不证明采集完整或格式版本完全一致。与目标 MC2 已导出的全部 8 个输入和 2 个输出形状严格一致的有 40 组，level 均为 node、输出 dtype 均为 `DT_BF16;INT32`；目标的输出形状一致记录也有 40 组。形状和类型不是任务标识，因此不能从中挑出第 15 层 EP3 globalTaskId 2380，也不能得到任何专家计数的值。即使将这些形状用于筛选，也只能说明这些记录的解析结果与目标导出字段相容；目前没有同次绑定证明。文件内是否存在安装解析器未描述的其他编码或其他捕获渠道，不能由本检查排除。

复现 `python3 agents/diagnosis/audit_ep3_mc2_tensorinfo.py`，只在自有 stage 容器内进行 CPU 元数据解析，输出非敏感汇总 [ep3-mc2-tensorinfo-aggregate.json](ep3-mc2-tensorinfo-aggregate.json)。脚本 SHA-256 `2cb2f7b8144b46736443dcd17bb8300c793d948860d02bf91a1652de6e94cc06`，结果 SHA-256 `f58e6a2b04fb231b11bc0987f390fef8814bf43b981a38e8787ecdd111df3abd`；结果列出切片哈希与检验次数，不输出原始 node、时间戳、请求、权重或张量元素。

后续若需要将这些 40 组与具体调用关联，需核实 profiler 中 GE node/时间戳到目标 CANN launch 与 globalTaskId 的官方关联契约及同版本生产者；即便关联成立，现有描述符也不提供专家计数元素。历史计数是否另有已保存数据须以其他文件的独立格式证据为准，不能从本结果反推为零或不存在。
