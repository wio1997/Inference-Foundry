# 上游适配复核：2026-11-18 14:14 UTC

本次复核固定在官方main commit `d89bc5b57477d89ab1411264599b153b6e34e78d`。GitHub最新release为[v0.5.23](https://github.com/sgl-project/sglang/releases/tag/v0.5.23)，发布时间2026-11-02T22:14:10Z；这不等于现有A2镜像已包含该release或具备本模型的完整NPU路径。[查询记录和源文件SHA](../evidence/upstream_refresh_G780/report.json)及[源码检查范围](../evidence/upstream_refresh_G780/inspection.json)保留了时间、版本和证据边界。

- [固定版本nightly workflow](https://github.com/sgl-project/sglang/blob/d89bc5b57477d89ab1411264599b153b6e34e78d/.github/workflows/release-docker-npu-nightly.yml)的device_type矩阵仍为a3、950。
- [固定版本NPU DSV4 backend](https://github.com/sgl-project/sglang/blob/d89bc5b57477d89ab1411264599b153b6e34e78d/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2507)的所查forward入口仍只接受0/4/128压缩比；当前检查点使用1/2。
- [PR47660](https://github.com/sgl-project/sglang/pull/47660)仍open、未合并，目标为Ascend A5。其描述涉及V4.1低压缩比、双层indexer和量化行格式；本次没有把它当作A2可运行或质量/速度通过的证据。

main相比09:26UTC的e9b0d0c0已变化；三个复核文件内容也都有变化。完整tree中存在通用V4.1模型、稀疏路径和多平台测试文件，[相关路径索引](../evidence/upstream_refresh_G780/relevant_tree_paths.json)只描述源码存在范围。上述证据尚未建立完整A2/910B原生支持，也没有证明新代码可直接移植到固定CANN9.1/TorchNPU post6镜像。未重新扫描全部镜像tag，本记录不更新“所有registry中最新A2标签”的结论。

继续沿用用户固定的既有daily A2镜像7da963869ee5，现有适配版本和历史失败保持原始证据。本轮只做上游只读复核，没有拉取/替换镜像、启动SGLang服务或提交NPU验证。完整1319题同口径质量、四档C16/1024首冷/重放矩阵、双方all-rank P/D profile及优化收益仍待完成。
