# 169 / 9382：核数与频率的一手资料补查

核查范围：本任务已保存 `hardware_sources_audit.md`、`int8_ceiling_certificate.md`、`evidence/actual-sku-and-cp.json` 和 `evidence/profile-device-capacity-info.json`；2026-10-29 用 Web 与直接 HTTP 读取公开资料，未访问设备和运行 NPU。

## 24 核的额外来源

昇腾官方 [RecSDK 固定提交的通信实现](https://gitee.com/ascend/RecSDK/blob/8251e16519db9132e33d33685d9d7a50607e1567/src/lccl/src/lcal_comm.cpp) `GetChipName` 使用 `rtGetSocVersion` 的返回字符串匹配 `Ascend910_9382`，`GetCoreNum` 对该枚举返回常量 24（源文件第 39、68、85、110–112 行）。通过直接 HTTP 获取的源文件保存为 `official-recsdk-lcal.cpp`，SHA-256 `343a7164c3e3c4bff4f2d5d580a3a8cf107eec71f7285922793a68f1829ce26f`。它与已保存 9382 SDK ini 的 `cube_core_cnt=24` 及历史 profiler 16 条 `ai_core_num=24` 一致，是额外的软件实现证据。

**边界**：这段代码是某版本 RecSDK 用于通信逻辑的静态型号映射，并非硬件规格证明；它不能证明本机每个逻辑 device 是完整 9382、历史进程真实执行了该 SDK、24 是所有可用计算资源的总上限，也未证明对应 profiler 字段在 169 的颗粒度。`rtGetSocVersion` 是可供之后核对的接口线索，当前不启动进程或调用设备。

## 频率的不同接口语义

[CANN driver 的公开头文件索引](https://gitcode.com/cann/driver/blob/master/pkg_inc/ascend_hal_base.h) 描述 `halGetDeviceInfo` 中 `MODULE_TYPE_AICORE/INFO_TYPE_FREQUE` 为 AI Core **rated frequency**；另在 `halGetDeviceInfoByBuff` 的 `MODULE_TYPE_AICORE/INFO_TYPE_CURRENT_FREQ` 描述当前频率。这提醒我们至少要区分额定频率和当前频率。上述内容来自官方 GitCode 搜索索引；直接 HTTP 返回了 HTML 客户端页面，未获得 raw 头文件、固定提交或与 169 已安装驱动版本的绑定，不能当成针对历史运行接口的版本内实证。

更关键的是：尚未证明本次 profiler 的 `aic_frequency=1800` 或配置 `cube_freq=1800` 来源于 `INFO_TYPE_FREQUE`，不能把两者的语义等同。即便证明是 rated，也不自动证明任何合法工作状态都不超过它。因此 `24 × 1.8 GHz × 8192 INT8 MAC/cycle = 353.8944 TMAC/s` 仍仅是 `int8_ceiling_certificate.md` 所列全部条件成立时的名义模型，**不是实证物理硬上限**。

## 缺口与优先级

针对只允许 INT8 Cube 稠密乘加的受限算子界，最低限度还须：

1. 用该次任务固化的资产/BOM 和运行时配置将全部参与 logical device 映射到精确芯片、芯片切分和可用 Cube 池；确认每个池实际和最大资源数。
2. 对对应 SKU、驱动和固件版本获取厂商 ISA/物理技术规格，明确 INT8×INT8→INT32 的单位周期最大发射/执行容量、是否支持更高瞬时突发、其多个指令能否流水叠加，以及适用的全部电源/频率状态的硬上限。不能仅由 profiler 估算器或 Mmad 单指令模型推出。
3. 对被比较的工作，证明真正必要的运算和允许实现域；若不限制只走 Cube，则需覆盖可并发参与的 Vector 或其他计算资源。对短时段需要计入起始缓存、在途工作和突发信用。

这些还不足以导出整模型 token/s 界：需要该负载有效 token 的不可绕过工作量/依赖与资源归属；如再用通信或访存收紧，才另外需要必要流量、拓扑和物理方向容量。资源能并行时以相应约束最大值组合；依赖已证明不可重叠才累加。尚缺硬规格时留作符号，不借其它产品额定值填空，也不把当前观测速度当上限。
