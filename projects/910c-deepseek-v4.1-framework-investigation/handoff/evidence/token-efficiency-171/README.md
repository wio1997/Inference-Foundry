# 有效Token与Padding账本

入口：[全局图 → 有效Token与Padding](../balanced-profile-171/index.html) · [报告](REPORT.md) · [断点](STATUS.md)。机器读取 [view-data.json](view-data.json)，逐波下载 [draft-dp-waves.csv](draft-dp-waves.csv)，代码与边界见 [source-evidence.json](source-evidence.json)。

新批次统计请求usage与全局计数器；原C8单独还原454个DP波组的草稿body SP行域。语义单位与采样不混用，当前全模型padding比例为null。重新生成仅使用已存资料：

```sh
python3 audit/token-efficiency-171/build_data.py
python3 audit/balanced-profile-171/build_global_view.py
node audit/balanced-profile-171/verify_global_view.mjs
```

构建输入是已有framework-timing-171/balanced-reference-1.json、execution-search/dspark-three-body-frontier.json及既有caller/source审阅。源码片段自包含；不需要为读报告连接服务器。原始输入SHA保留在JSON。本轮无新profile、模型请求、服务或远端动作。
