# 复现边界

原始冻结命令包含本机绝对解释器路径、唯一输出目录和已锁依赖哈希。它们是实际执行记录，不承诺在其他平台原样一键运行。复现实验应建立新执行根和新Run/spec，审阅并重新冻结路径与依赖，保留已有FAIL/INVALID；不要覆盖旧输出或为通过检查修改原始记录。

所需源文件、脚本、spec、候选及其review均已按冻结相对路径保存。依赖二进制不分发，使用environment锁和msgspec020-files.json核验；跨平台重新安装须作为新环境，不能冒充原同条件Run。tokenizer两文件由[历史身份记录](../2026-10-19-detokenize-finish/tokenizer-asset-identity.json)恢复并匹配两个SHA256，不下载权重。

默认pickle与可选msgpack分开验证；msgspec0.19失败未删除，0.20仅为隔离对照。候选只应用于夹具中的原生成器方法，未应用任何运行服务。完整模型、PD、scheduler和KV释放不能由本机协议测试替代。
