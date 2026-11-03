# Stage 086：169 容器可见的服务器型号核验

## 执行及结果

`operator-ceiling169-v2-086-dmi.exit-code` 为 **0**；`operator-ceiling169-v2-086-dmi.verified.json` 记录执行模型为 `deepseek-flash`，查询输出位于 `stage-dmi-readonly.json`。该输出所带的 `stage_dmi_readonly.py` SHA-256 与本地脚本重算一致。脚本限定读取五个非序列号 DMI 字段，未调用 NPU、未读取 UUID，使用已有 stage 容器。

容器能够读到以下字段：

| DMI 字段 | 返回值 | 可支持的判断 |
|---|---|---|
| `sys_vendor` | `iSoftStone` | 固件报告的系统厂商名称。 |
| `product_name` | `ChaoqiangA800T A3` | 固件报告的产品型号字符串。 |
| `product_version` | `1.0` | 固件报告的产品版本字符串。 |
| `board_vendor` | `Huawei` | 固件报告的主板厂商名称。 |
| `board_name` | `BC83AMDBI02-7280Z` | 固件报告的主板型号字符串。 |

这些字段是**容器可见 DMI**，本次没有验证它们必然来自裸金属而非虚拟化环境，也没有获取资产清单、完整 BOM 或实际芯片规格。`board_name` 是主板字段，不能直接当作 NPU 芯片型号。

## 与公开规格如何关联

[软通华方产品目录](https://tongfangpc.com/zh-cn/htmls/chanpin/index.html) 的搜索索引中存在“软通华方超强 A800T A3”。它与 DMI 中 `ChaoqiangA800T A3` 在拼写/音译上吻合，是识别整机产品的有力线索，但当前尚没有厂商文件把该 DMI 字符串、版本 1.0 和精确产品/BOM 作一一映射。检索结果保存为 `isoftstone-catalog-search.txt`。

此前查到的华为 **Atlas 800T A3** 是另一个厂商页面。即使名称相似、两者均可能使用昇腾硬件，也不能因此认定软通华方的同名系列与该华为型号的出厂芯片、时钟、互联和峰值完全相同。因此华为页面的 **12.0 POPS、3.2 TB/s、双向 784 GB/s** 暂不能用作 169 的物理硬上限或有方向通信 cut 上限。当前没有找到与 DMI 型号和版本准确对应、含精度/频率/链路范围的软通华方一手规格；实际 NPU INT8 和带宽硬上限维持 **UNKNOWN**。

下一步查该 OEM 产品的技术规格或已保存的资产/BOM，以及系统是否为裸金属；核对板卡型号、数量、每卡 chip/资源布局、dtype 与计数口径后再评价厂商额定值。只有必要工作与对应硬容量都得到证明，才能把额定容量用于算子或端到端吞吐上界。

## 复核

执行 `python3 agents/bounds/stage_dmi_audit.py` 可校验退出码、ZCode 回执、脚本 SHA 和索引快照并生成 `stage-dmi-audit.json`。它只读取已有本地文件，不连接服务器或操作 NPU；不能验证 DMI 是否真实反映物理硬件，也不能证明厂商规格适用。
