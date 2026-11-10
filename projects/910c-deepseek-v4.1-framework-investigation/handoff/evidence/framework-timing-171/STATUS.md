# 状态

介绍：将输入准备的未知区间继续拆到具体函数，同时建立可迁移的“请求→EngineCore→worker函数→设备算子”账本。

到了哪里：本次有界诊断采集完成，2组×4条镜像请求、32K/1K；完整请求轻量计时，加一个16卡各12迭代的CPU/NPU详细窗。准备97个函数定义标记，实际加载95个，18个进程产生计时记录。新增性能优化实现0。

结果：原服务已恢复health200、8-token请求成功，11份适配源码SHA一致，原Config无诊断开关，临时实例停止。16份trace已离线导出。函数/设备分析只用本次导出，不再启动第二次模型采集。详细数值以REPORT.md、view-data.json和CSV为准。

本轮轻量请求15.967999s / 513.026输出token/s；详细profile17.632761s。off轮20.163047s仍有初版探针控制轮询，不作干净基线；既有真正无探针中位16.090768s另列。不能用单轮接近证明零扰动，也没有已兑现吞吐改善。

明确缺口：本次API前处理逐函数、自动跨进程request/batch因果关联、DFC内部阶段、off-CPU具体原因。标记函数自身仍可包含未标记的Python/库调用/等待；不能伪称没有未知或全为可省。

下一条待做：依据新同窗具体函数空档贡献，沿相同数据交付的公共metadata/输入frame大范围估净价；复用原融合搜索的逐位、代次与栅栏证据。只有会改变开发去留的关键缺口才补证。SGLang/DeepInfer需按真实版本绑定入口与SHA后验证，本轮没有在这两个框架运行。

证据：launch.json指向171原始数据目录；raw-inventory.json含16份trace、kernel CSV和19份span文件的字节与SHA；receipt.json、restored-smoke.json为恢复凭据。captured-hook/保存实测探针初版，tools/framework-timing保存修复off轮询后的通用工具，未进行第二次NPU复测。

失败记录保留：初次离线导出助手早于receipt创建而退出，已重启等待助手；初次函数解析未兼容trace JSON数组格式，修复后只重算已导出数据，不重跑请求。export-attempt1-state.json、export-state.json、原日志和analysis-state.json分别保留历史失败与最终CPU分析结果。
