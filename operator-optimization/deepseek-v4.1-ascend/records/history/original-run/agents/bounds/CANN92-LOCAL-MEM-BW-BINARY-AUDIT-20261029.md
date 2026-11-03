# CANN 9.2 存储带宽平台字段：二进制只读审计

## 能确认到哪一步

2026-10-29，检查 169 个人 stage 容器已安装的 CANN 9.2 `aarch64-linux/lib64/libplatform.so`。其 `fe::PlatFormInfos::GetLocalMemBw` 实际二进制中的 `LocalMemType` 值 6（该版本头文件定义 HBM）分支，以 `AICoreMemoryRates` 为资源类别、`ddr_rate` 为键调用 `GetPlatformResWithLock`；值 4（L2）分支使用同一类别、`l2_rate` 键。返回字符串非空时经 `strtoll` 十进制解析并写入输出；空字符串时输出零。该函数没有读取 `ddr_read_rate` 或 `ddr_write_rate` 来分别返回单向值。**这是已经安装的库中此函数的行为，不是其物理带宽可靠性证明。**

新增证据具体位置：函数从 `0x30288` 开始；比较值 4/6 在 `0x302cc` / `0x302d4`；HBM 的两字符串地址 `0x51fe0` / `0x51798` 和调用 `0x30324`；L2 对应 `0x51fe0` / `0x517c8` 和调用 `0x303e8`；数字解析 `0x303f4`。精简原始反汇编与字符串见 [cann92-getlocalmembw-disasm-20261029.txt](cann92-getlocalmembw-disasm-20261029.txt)，SHA-256 `d99a2acbd2fd158bad70de02750c19e60bb6567f4dc58120b8e2f8126bbba93e`。库自身 SHA-256 `324f595d468896e855a10a836c06a4cab90fba6ec8c53124827bd266d65bd597`；当前 stage 和基线容器中的该文件 SHA 相同，两个容器的 Docker image ID 均为 `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`。这不证明历史 profile 的任务实际调用了这段函数，或其 runtime 映射及数据没有变化。

已有 `evidence/actual-sku-and-cp.json` 保存的 `Ascend910_9382.ini` 中 `ddr_rate=32`；**仅当**调用时此资源对象确实由该文件且未经覆盖填入，`GetLocalMemBw(HBM)` 此路径才按上述逻辑取到对应值。具体值源、runtime 所选配置及实际调用尚未验证。已安装头文件定义 `fe::LocalMemType` 为 HBM=6、L2=4。继续检查同安装中的静态归档 `lib64/libtiling_api.a`，其 `platform_ascendc.cpp.o` 的 `PlatformAscendC::GetCoreMemBw` 在 `0x144` 读入内存类型整数，并在 `0x170` 调用 `fe::PlatFormInfos::GetLocalMemBw`（反汇编 relocation 符号明确）；在此调用路径中 L2 和 HBM 的枚举值分别相同；两套枚举并非整体相同，不能推广到其他类型。原始证据 [cann92-getcoremembw-archive-20261029.txt](cann92-getcoremembw-archive-20261029.txt) SHA-256 `07353e0edbccb1aec35df6597672328a9df5951f1b71823d0b79f32524c1ea21`，归档文件 SHA-256 `62fa56581595618a8f696a79e332195de95771d5a84cb8b9f81b3dd6d970c5fd`，当前 stage 和基线容器各自归档的 SHA 一致。公开的 [GetCoreMemBw 开发版文档](https://asc.gitcode.com/api/Utils-API/platform_info/PlatformAscendC/GetCoreMemBw.html) 将该接口返回值描述为 Byte/cycle，但该页面对应开发版 master 而非已证的本机 9.2 定版；相邻 CANN 版本文档有同样表述，不能单凭签名把单位无条件套给本机。结合本版静态归档与同版动态库，可以在**调用实际链接此二者且资源对象来源成立**的条件下，把 `ddr_rate` 视为该平台接口返回的 HBM 参数；单位对 9.2 的适用性、时钟域、读写方向、容量粒度及物理硬界语义均未闭合。没有证明历史 profile 的工作实际链接或调用这些文件。

## 对理论性能界的影响

本证据补上了内部资源取值过程，未关闭既有缺口：历史配置来源、时钟域及频率上限、返回值是每核还是整个设备、读写方向、共享路径与突发包络均未证明。因此 **`P_INT8,max` 和目标 HBM cut 的 `B_cut,max` 仍为 UNKNOWN，不能将 32 × 1800 MHz（或再乘核数）作为不可超越物理带宽，也没有新的数值吞吐上界**。符号界的必要工作量及该带宽是否覆盖整个时间窗仍要另证。

下一步应查资源对象加载/覆盖路径和历史 profile 使用的链接文件，再查精确板卡 SKU 的时钟/方向/共享和突发硬规范。找不到资料时保留 UNKNOWN，不能以实测最大值代替硬上限。

## 可复现与限制

所有指令均只读，CPU 执行，无 NPU workload 或推理实现修改。命令从已有 `ssh169.sh` 进入用户 stage 容器执行：

```sh
LIB=/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libplatform.so
sha256sum "$LIB"
nm -D -C "$LIB" | grep GetLocalMemBw
objdump -d -C --start-address=0x30288 --stop-address=0x30414 "$LIB"
objdump -s --start-address=0x51fd0 --stop-address=0x52010 "$LIB"
objdump -s --start-address=0x51780 --stop-address=0x517e8 "$LIB"
grep -n 'enum class LocalMemType' /usr/local/Ascend/cann-9.2.0/aarch64-linux/include/platform/platform_infos_def.h
ARCHIVE=/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libtiling_api.a
sha256sum "$ARCHIVE"
nm -A -C "$ARCHIVE" | grep -E 'GetCoreMemBw|GetLocalMemBw'
objdump -drC "$ARCHIVE" | sed -n '/^[0-9a-f]* <platform_ascendc::PlatformAscendC::GetCoreMemBw(/,/^$/p'
```

不同版本的地址可能变化；用符号和字符串交叉检验，不能直接把地址套到别的二进制。既有材料及完整硬件口径参见 `RECAPTURE-HARD-CAPACITY-20261029.md`、`hardware_sources_audit.md`。
