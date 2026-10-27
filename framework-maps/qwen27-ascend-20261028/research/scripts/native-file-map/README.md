# 原生地图离线读取与浏览器核验

本轮无性能测量；这是交付可用性修复。原页面在 file: 下 fetch 原生 gzip 被 Chromium 拒绝。新目录为每个完整 gzip 生成本地脚本封装，浏览器只加载所选 rank；压缩与解压 SHA、字节数、身份校验保留，解码后释放封装字符串，HTTP 仍走原 fetch。无需放宽浏览器安全设置。

原 gzip、JSON、性能数字和旧地图保持原字节。输出是整个目录，必须保留 native-rank-details 下的 gzip 与 .js；不是包含全部原生数据的单 HTML。双击 index.html 或通过 HTTP 打开同一目录。

```sh
python3 -B build_native_file_map.py --source SOURCE_MAP --output NEW_MAP
python3 -B browser_qa.py --map NEW_MAP --output NEW_QA
```

默认严格要求真实204身份及已存在的 composition receipt。浏览器检查需要当前环境中可用的 playwright 与 Chromium；检查 file/HTTP 两种模式、每框架真实rank与最大解压分片、六类数据计数、筛选、SHA/身份门槛和旧41CPU/4SG/候选卡片。不会运行模型或设备测试。

本代已对真实138874072字节 JSON rank 验证 file/HTTP，六类数据与290772个 task groups 源计数正常；损坏实际 gzip 封装在 SHA 检查时拒绝，未显示 native table。上述单rank兼容检查不构成完整204地图QA、推理收益或质量等价。

--diagnostic-single-rank 仅用于明确单rank诊断，产出显式 not204 收据。正式调用不要使用此参数。root整体接受、quality/greedy和native外围完成语义均未被本组件改变。
