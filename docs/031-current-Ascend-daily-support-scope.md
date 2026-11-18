# 当前Ascend daily与1017 W4A8的适用范围

2026-11-18核对官方源与镜像注册表。上游[SGLang v0.5.21说明](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)写明停止发布A2/910B的release/nightly；[安装页](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/getting-started/installation)列出A3与950的daily/validated build。Ascend main固定到e9b0d0c0a3cc033a653501231b2de6af9ffec7db，nightly矩阵实际只有A3和950、amd64/arm64。最新A3/950标签不是910B适配通过证据。

Quay注册表第一100个按发布时间返回的标签里，最近A2日期构建仍为`cann9.1.0-910b-20261103`，manifest list `sha256:7bbe2e8e02affae5139710a692d345b6820de063d7ceb460da02b310114530a2`，arm64子manifest为82532c070427448997a338f53ed0a1a01a57b27c8f77afc12cd6360ce2eac44e；两机现用SG image ID7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec，现场RepoDigest相同。没有发现更新的官方A2 ready-to-run V4.1镜像。

这一结论不止依据模型名称：当前main原生NPU attention仍在forward拒绝非0/4/128的压缩比例，没有V4.1 C1/C2低比例分支；[PR47660](https://github.com/sgl-project/sglang/pull/47660)仍open/unmerged且面向A5，不能作为A2可用证明。上游GPU支持V4.1、A3/950有最新daily或PR内有算子均不等于910B/W4A8/完整PD已通过。

因此保留现有dailyA2及已经形成的适配证据，G603 P/D501功能诊断继续自然完成；不换成A3/950镜像或撤销必要适配来宣称开箱支持。vLLM按用户另行更新为v0.32.0rc1，两分支分别保留实际环境、配置和失败。新vLLM实际部署/完整质量/四档AISBench/双方profile与差异地图均待完成。

- [官方nightly实际矩阵](../code/upstream_daily_review_G612/.github/workflows/release-docker-npu-nightly.yml)
- [当前原生NPU后端源码](../code/upstream_daily_review_G612/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)
- [注册表查询原始100标签](../evidence/image_refresh_G612/sglang_registry_tags_first100.json)
