# SDK Python 运行层源码

保留实际私有 SDK 运行归档中的 Python 源文件，包括 MemFabric 的 RTLD_LOCAL loader，避免仅保存版本号而丢失本地适配。来源包 SHA 与成员 SHA 见 PROVENANCE.json。

这是文本源码备份，不是可独立安装的 SDK；native so、wheel 依赖和编译源码包位置见上层 ASSETS.json。旧 sdk-mf-local 是 fallback，不要覆盖官方1.3 overlay；恢复时先核对进程实际映射库和 Python 导入路径。不得将这些模块直接铺到系统 SDK，也不应在 Mac 导入测试。
