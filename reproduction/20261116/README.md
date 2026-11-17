# 当前G133/G134源码、依赖、参数与测量归档

[完整复现步骤与两边参数](../../docs/005-current-comparison/README.md) · [分支首页](../../README.md)

本目录覆盖2026-11-16测量所用SG适配与native配置，归档日2026-11-17。268文本运行文件（267原字节相同，1历史收据身份字段脱敏）、19二进制原测SHA与重建输入、23挂载、16AISBench配置和528请求证据。

- 源码：[runtime](runtime/) / [SHA映射](runtime-manifest.json)。
- 引擎/Docker/env：[deployment](deployment/)。
- 二进制依赖：[原测SHA](binary-manifest.json) / [CPU构建源码及固定镜像输入](build/)。
- 压测：[16配置](configs/) / [同一八档输入manifest](dataset-manifest.json)。
- 结果：[22轮阶段](measurement/rounds.json) / [主比较](measurement/comparison.json) / [验证](publication-validation.json)。

本次发布未执行新NPU任务；新构建/便携启动器仍需在目标机验收，不将离线校验称为重新测得性能。
