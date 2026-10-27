# SG输出token谱系补证（未编号草稿）

本段只复核已保存的 cleanSG gen67：aarch64 / Ascend910B1、物理4–7卡TP4、CANN9.0.0 / torch_npu2.10.0、bucket1、512/C1/out32插桩诊断；无新模型/服务/NPU/SDK作业，无新增性能测量。原正式out1024数字、噪声范围与部署混杂保持不变。来源是第35代执行者核验，root验收仍待办。

四rank各32次正增量commit：1次prefill提交首token，31次decode提交其余token。输出ID序列在四rank与独立frontend记录中完全一致，逻辑输出只有32个。每rank的第33次commit为零增量tail，不能多算为输出。台账共132行是4个TP副本，不相加成token数或墙钟。

图回放按同rank/PID、明确RID和work_key.step=scheduler_commit.forward_iter关联，31个正输出decode与1个零输出tail/rank分别对应原root已接受的7936与256个层端点行。首token来自prefill；不能给它补造decode图或原生端点。端点不是独占kernel、全stream关键路径或可恢复收益，本次未重新审计原始native trace。

当前tokenizer全文件SHA为 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`。独立ByteLevel与tokenizers0.22.2原生解码均与保存SSE全量及32个累积内容前缀相等。历史进程实际加载tokenizer的不可变SHA收据未取得；每SSE消息未记录token IDs，前缀内容相等不升级为传输身份。当前 `<think>` 对应ID248068且special=false，解释解码保留，不解释模型生成原因。

vLLM保存记录缺少已生成token IDs；不能通过文本重编码替代，也不能推出首个不同采样token、logits原因、全矩阵质量或greedy等价。相同payload两份短诊断文本不一致的旧结论保留，优化未采用。HCCL等待与别名去重、跨CANN/卡组混杂与Q06 fallback不变。

- [完整132行逐rank台账](assets/sg-output-token-ledger.csv)：包含step、增量、累计长度、last_token_id、明确invocation和层端点行计数。
- [谱系审计](evidence/sg-output-lineage/sg-output-lineage-audit.json)、[安装源码契约](evidence/sg-output-lineage/source-contract.json)、[CPU验证](evidence/sg-output-lineage/validation.json)。14个x86_64 CPU身份/负向测试只验证审计器边界，不替代aarch64设备或性能结论。
- [逐文件SHA与仓外证据包](SG_OUTPUT_LINEAGE_INDEX.json)：仓外包保存审计脚本、原小件和完整回放关联输入；大profile、tokenizer文件、传输器和凭据不入库。

新版地图证据页提供四rank全部台账及10份小件SHA入口；这是独立预览，实际204、最后地图QA、分R、main/index发布与root整体接受仍待完成。临发布须刷新实时项目规则、分支头、全部占号及leafsys1权限，当前仅沿用第29代规则快照制作草稿。
