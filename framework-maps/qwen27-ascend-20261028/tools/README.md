# 公开归档工具

在归档目录执行：

```sh
python3 tools/verify_archive.py
```

若本机有 Node，可额外检查地图 JavaScript 语法：

```sh
python3 tools/verify_archive.py --node node
```

校验器检查导出清单中的文件字节/SHA、内嵌 gzip CPU 分片与外置文件一致性、页面引用的分片 SHA，以及原生事件入口关闭状态。它不访问网络，不启动模型或服务，不测量性能，不证明完整关键路径或输出质量。

`export_public.py` 是生成本公开派生副本的导出器，需要调用者提供已封存的研究目录和地图目录，并指定一个不存在的新输出目录：

```sh
python3 tools/export_public.py --research-source SOURCE_RESEARCH --map-source SOURCE_MAP --destination NEW_PUBLIC_ARCHIVE
```

它不包含来源服务器、账号凭据或自动下载原始资料的逻辑。源研究的原始大文件未随公开归档分发。导出器生成研究与地图文件；顶层说明和本工具说明属于本次归档装配。
