# 171 空档操作与数据字典

从[全局图的“空档操作与数据”](../balanced-profile-171/index.html)进入；人读 [REPORT.md](REPORT.md)，新对话读 [STATUS.md](STATUS.md) 后按需取 [view-data.json](view-data.json) 的单项记录。

28 项操作按 8 阶段排列，区分准备工作、跨轮反馈、真实设备数据读写、未核准人口的条件分支。每项含数据位置/语义形状、读取/生成/用途、实际消费者、提前依赖、优化判断、源路径/行段/源码片段/SHA，以及已有函数总账引用。`field_time_ms=null` 表示没有字段独立计时；引用同一函数的项不得重复加价。

计时使用 function-profile-20261110T090101Z 的16卡各11周期细函数诊断；source SHA 与同轮 catalog 或 loop046 当前源码审阅一致。本轮完全在本机复用资料，新 NPU/模型请求/profile/服务动作0。没有重新核验171 live环境。

```sh
python3 audit/gap-dataflow-171/build_data.py
python3 audit/balanced-profile-171/build_global_view.py
node audit/balanced-profile-171/verify_global_view.mjs
```

build_data.py 需要同级 framework-timing-171/function-analysis.json、已冻结源和 execution-search/common-device-source-review.json。无模型依赖，不连接服务器。索引器、压缩器两份复用源在本目录 source/；其它源引用已有证据包。HTML 内嵌所有展示数据及源码片段，可独立打开。
