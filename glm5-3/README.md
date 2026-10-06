# GLM-5.3 Extreme

在现有正确算子与两台服务器内，为GLM-5.3 W8A8构建功能完整的标准P/D分离推理框架，寻找可消除的执行浪费并提高真实完整E2E性能。目标合同见[MISSION](MISSION.md)。

目标权重：`/data/tiankuan/wio/GLM-5.3-w8a8`；新服务模型名：`glm-53`。2026-10-06用户正在上传，完成并核验前不加载或测试。现有`glm52-single`容器、`glm52-pd/deploy`目录是实际资源名；历史Run保留当时版本，不构成5.3的Current/性能基线。当前方案见[PLAN](PLAN.md#当前执行方案标准pd优化)。

全新Agent从[START_NEW_CHAT](docs/START_NEW_CHAT.md#恢复顺序)最小恢复顺序开始，按[AGENTS](AGENTS.md)短规则进入性能分析；可直接复制该页的短启动指令。

166本地工作目录固定为`/data/tiankuan/wio/Inference-Foundry/glm5-3`；Agent直接读取同一checkout内的`../AGENTS.md`、`AGENTS.md`、`MISSION.md`、`PLAN.md`及恢复入口。GitHub用于同步/版本核验，不用逐页访问网页恢复资料，也不另复制一套规则。详见[本地入口](docs/START_NEW_CHAT.md#166本地工作入口)。

稳定环境字段按需查[ENVIRONMENT_RECOVERY](ENVIRONMENT_RECOVERY.yaml)；路径与发现方法只查[RECOVERY_INDEX](RECOVERY_INDEX.md)，现役/候选代码版本只查[CURRENT_PRODUCT_MAP](CURRENT_PRODUCT_MAP.md)；事实、性能与证据沿其权威链接读取。其他资料从[按需研究索引](docs/README.md)进入，不默认通读。
