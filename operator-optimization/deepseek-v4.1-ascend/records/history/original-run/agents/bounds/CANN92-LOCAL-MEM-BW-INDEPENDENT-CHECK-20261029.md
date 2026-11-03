# 独立复核：CANN 本版带宽接口的二进制关系与文档版本

2026-10-29，只读检查已有反汇编与 169 stage 容器内安装头文件，未运行 NPU 负载。

1. 在已记录 SHA-256 为 `62fa56581595618a8f696a79e332195de95771d5a84cb8b9f81b3dd6d970c5fd` 的静态归档中，`platform_ascendc.cpp.o` 的 `GetCoreMemBw` 指令 `0x144` 从输入引用加载 32 位枚举，`0x14c` 保存到栈，`0x168` 将该栈地址作为 x1，`0x16c` 传递原输出地址 x2；`0x170` 的 `R_AARCH64_CALL26` 重定位指向 `fe::PlatFormInfos::GetLocalMemBw`。因此该归档内的调用链和原整数传递成立；它并不能单独证明任一历史运行链接了该归档、最终解析到哪个共享库或资源对象的来源。反汇编全文见 `cann92-getcoremembw-archive-20261029.txt`。
2. 当前安装的 `include/tiling/platform/platform_ascendc.h` 第 47–58 行中 `CoreMemType::L2=4`、`HBM=6`；`include/platform/platform_infos_def.h` 第 23 行 `fe::LocalMemType::L2=4`、`HBM=6`。同版 `GetLocalMemBw` 对 4 / 6 的分支及字符串映射见 `cann92-getlocalmembw-disasm-20261029.txt`。两枚举**仅就 L2/HBM 已验证相同**：前者还有 `FB=7, BT=8`，后者 `RESERVED=7`，不可泛称全部枚举一致。
3. 公开 [Ascend C GetCoreMemBw 页面](https://asc.gitcode.com/api/Utils-API/platform_info/PlatformAscendC/GetCoreMemBw.html) 称输出 Byte/cycle；直接 HTTP 取到的文档 footer 标记构建版本 `master`、提交 `21bad166c6b39ffeb2388a5fa47394968bfa612d`，并说明其来自持续开发的 master 分支，不能据此认定是本机 CANN 9.2 的定版文档。[CANN 9.0 官方接口文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1035.html) 也规定 Byte/cycle，但它是另一版本。当前同安装头文件接口声明未标注单位，机器码仅读取并解析字符串、没有单位信息。对本次 9.2 安装应用 Byte/cycle 应标记为跨版本 API 契约一致性假设，直到取得与该安装构建匹配的文档或其他权威证据；更不能据此证明物理最大值。

复核入口（不会访问 NPU）：

```sh
docker exec wio-vllm-v41-repro169-stage-20261028 sh -c 'sed -n "47,59p;146p" /usr/local/Ascend/cann-9.2.0/aarch64-linux/include/tiling/platform/platform_ascendc.h; sed -n "23p" /usr/local/Ascend/cann-9.2.0/aarch64-linux/include/platform/platform_infos_def.h'
curl -L --max-time 15 -s https://asc.gitcode.com/api/Utils-API/platform_info/PlatformAscendC/GetCoreMemBw.html | grep -oE '.{0,100}(document-build-info|21bad166c6b39ffeb2388a5fa47394968bfa612d|持续开发版本|Byte/cycle).{0,150}'
```

文档 master 会变化，提交号只是上述取证时点的版本；应保存该 HTML 或提取记录复核，不能假设将来页面不变。
