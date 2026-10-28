# DeepInfer 910C 部署分支

- 部署入口为 projects/deepinfer-dsv41-910c/README.md，源码以该目录的归档及 SHA256 清单为准。
- 本分支仅记录服务器172；远端写入限于 /data/tiankuan/wio。保留冻结 vLLM 与全部历史实验，不覆盖旧结果，不停止其他项目进程。
- 使用学习式 DSpark、完整目标验证和置信度调度；提示查找不能替代 DSpark。仅 Engram CPU 卸载，主干 CPU prefetch 关闭。
- 本次172 DeepInfer既有 privileged 授权继续适用；权重、驱动只读。认证直接进入SSH，不打印、不落盘或提交。
- 简单机械执行按用户要求由本机 ZCode 配置的 DeepSeek 执行；设计与验收独立核对。
- 公共网页通过 HTTP、API、Web Search 或连接器读取，不使用浏览器。
- 所有上游优化的迁移与同期 vLLM 对照尚未完成，不以固定负载两轮结果宣称通用领先。
