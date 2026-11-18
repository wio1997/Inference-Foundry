# 有界CPU客户端与同字节数据复核

G503客户端已在目标D主机的数据分区启动，限制CPU184–187四核、32GiB、无NPU设备。旧G83容器核对精确ID、固定镜像和任务标签，并确认只有sleep进程后停止；原始数据完整保留，供新客户端只读挂载。CPU lease在同一规范flock下由控制器持有，模型P/D服务继续运行。

新客户端独立使用同一tokenizer重新解析24份数据，逐条断言32768/65536/131072/262144实际输入token数和1024输出配置；448个首128token前缀互不重复。父级重新对照原始G83全部数据SHA，集合完全相同。五轮warm16和formal32每档数据保持首次准备的原始字节，未重建或选择更有利样本。

这一验收只覆盖CPU输入资产和受限客户端。尚未执行AISBench，没有实际server-usage、命中、PD、质量或性能通过结论；服务API和完整图也不能替代这些门槛。48份两框架配置与绑定真实D缓存的采集源码已准备，正式发送仍受完整服务、功能、前缀和充分独立预热要求约束。

- [父级复核与原始归档SHA](../evidence/CPU_client_G503/parent_acceptance.json)
- [24份逐文件tokenizer复核结果](../evidence/CPU_client_G503/asset_revalidation.json)
- [数据manifest](../evidence/CPU_client_G503/dataset_manifest.json)
- [448前缀唯一性](../evidence/CPU_client_G503/cold_prefix_audit.json)
- [CPU/无NPU/只读挂载凭证](../evidence/CPU_client_G503/create_receipt.json)
- [实际CPU控制器](../evidence/CPU_client_G503/cpu_client_controller.py)
- [实际复核源码](../evidence/CPU_client_G503/validate_existing_assets.py)
- [原始G83数据manifest](../evidence/CPU_client_G83/dataset_manifest.json)
