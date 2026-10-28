# 172 官方 DSpark 当前状态

- 指定贪心32K/1K/C8两轮：置信度调度版v15正确且达冻结vLLM吞吐/TTFT门槛，16完整输出一致；结果仅限该负载，不宣称稳定或通用领先。当前资料见projects/910c-deepinfer-172/STATUS.json及optimizations/20261028-dspark。
- 当前172服务已按用户要求恢复，16逻辑设备DP2TP8EP16；保留现场服务和冻结结果，不为交付额外重跑或重启。旧目录现场描述仅对应旧版本。
- 必须保持学习式草稿、完整目标验证和官方置信度策略；提示查找不能替代。此前promptlookup63/verify64完成结论撤回，历史数据不删除。
- 所有远端项目数据仅在/data/tiankuan/wio；目标仅172。冻结vLLM、旧实验和旧优化目录不覆盖；不停止外部任务，不发送170消息。
- 简单执行交给本机ZCode配置的DeepSeek；主控负责设计、实现和独立验收。本次172 DeepInfer既有privileged授权保留；权重与驱动只读，只有Engram CPU卸载、主干prefetch关闭。
- 凭证只由当前Mac钥匙串直接送入SSH，不打印、不落盘或提交；公共网站使用HTTP/API/Web Search，不使用浏览器。
