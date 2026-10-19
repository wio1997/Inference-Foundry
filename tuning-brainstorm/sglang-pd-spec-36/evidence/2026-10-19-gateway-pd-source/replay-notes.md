# 复现边界

原冻结命令使用本机路径和一次性输出目录，不得重复执行覆盖旧Run。代码与回执提供输入哈希；2.94MB完整上游Git tree清单保留本地，由controller-review中的固定公开API入口恢复，不在本目录重复分发。raw source-acquisition的tree_sha复制的是commit/tree-ish字段，真实根tree由source-review另行记录；原始回执未改写。

复现时获取固定commit/逐blob核验，建立新环境、新源清单和新Run重新冻结。当前未安装Rust，未解析Cargo依赖；不要将官方下载尝试写成软件身份已验证。源码已完整恢复，不需再次下载408文件。未注册的test_L033_gateway_crate.py只是私有草稿，不属于本轮执行证据。模型、tokenizer、权重、Rust/dependency二进制和ZCode完整模型I/O均不分发。
