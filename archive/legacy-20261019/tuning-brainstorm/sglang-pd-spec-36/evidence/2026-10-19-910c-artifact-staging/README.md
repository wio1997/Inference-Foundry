# 910C 制品准备、后端支持与中断恢复

**46节点的原模型复制已完整校验；44节点使用新资源预留，正在重验旧文件并续传。完整PD＋MTP服务尚未启动，整体仍为E0，accepted_baseline=null。** 本轮所有测试运行在910C服务器；Mac只承担连接、静态审查、记录和提交。

## 已核实的制品

- 44/46镜像ID均对齐为`sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263`（[导入回执](runs/L039-R011-image-import/result.json)）。44导入前核验了完整导出归档SHA256、image配置ID与无RepoTags条件，没有改变既有tag。镜像源码HEAD为`ce0a8fea5f4245839d174b2b8c55692be1810a7f`。实际DockerRootDir为`/docker_data/data`，导入前在那里确认可用空间；早期`/var/lib/docker`观察没有用作真实容量依据。
- 原模型来自37节点本地ext4目录；186个索引权重分片764037264096字节，另有7份元数据。[初始193文件合同及镜像归档清单](runs/L039-R009-transfer-launch/remote-result.json)。普通W8A8目录有182分片，量化描述/索引/模板不同，没有代替W8A8C8。
- 模型经37→44/46私网直传。源只读挂载、文件stat固定，源与目的各算SHA256，目的fsync后原子改名；mTLS只接受任务临时客户端身份和指定地址。源CPU2/内存1GiB、目的CPU2/内存2GiB，均无NPU设备；源聚合限速512MiB/s。这些是制品搬运条件，没有当成服务通信带宽校准。
- **46最终193文件/186分片/764089835336字节，全部文件名、大小、源/目的SHA匹配，进程Exit0。** [R022完整文件哈希清单](runs/L039-R022-transfer-verification/stdout.json)、[R029复核](runs/L039-R029-transfer-verification/stdout.json)。完成复制不代表权重加载或模型输出已正确。

## 44节点中断与恢复

R019的kubelet查询超时，R020确认44为ReadyUnknown、unreachable；没有由查询超时推断任务退出或重启。R022恢复可读时，原复制容器已Exit255，只完成161文件（含镜像）；源端保留两条SSLEOF错误，原始记录未覆盖。[节点与复制观察](runs/L039-R020-network-state/metadata.json)、[退出状态](runs/L039-R022-transfer-verification/stdout.json)。现有证据不能确定主机故障根因。

R024在创建续传容器前被资源预留检查拒绝。**勘误：JUMP第6行失败是原44P Pod已Failed，不是节点再次ReadyUnknown。** R025确认节点ReadyTrue且相同Pod UID已Failed；R027读取到UnexpectedAdmissionError/no healthy devices的既有失败原因，随后重新核验当前空闲和可分配资源，创建新非抢占Pod `sglang-mech36-prefill-44-l039r027`，UID`afbb3660-1ce7-4df2-9970-874b5ece4461`，Running/Ready、16资源；原Failed Pod及复制容器均保留。[新资源回执](runs/L039-R027-reservations/stdout.json)。46原预留不变，没有停止其他任务服务。

R028创建新目录和新CPU容器`if-mech36-transfer-dst-l039r028-44`。只将原已完成的160个模型文件硬链接到新目录，**逐个重新读盘算SHA并对照原记录/源端SHA后才信任**，然后复制缺失的128163549096字节。旧文件、进度和部分文件不覆盖。[启动回执](runs/L039-R028-resume-launch/remote-result.json)。R029实际观察到28/160个复用文件、88921106144字节重新校验通过；其余仍在后台处理，尚无新下载，也没有宣布44模型完成。[精确活动句柄](notes/L039-transfer-handles.json)。仅在全量193文件合同验证后才可加载模型；超时后先检查相同句柄。

## 实际服务镜像的参数与输入支持

R012只调用prepare_server_args，结果中名为resolved的字段仍属解析阶段，不能当最终运行值。R013对P/D分别执行resolve_once并确认完成标记，同时构造target/draft ModelConfig。[规范化结果](runs/L039-R013-args/result.json)、[准备意图](specs/L039-prepared-runtime-intent.json)。**原始NEXTN规范化为EAGLE，3 steps / topk1 / draft4；这里不是独立EAGLE3算法。** target/draft分别为GlmMoeDsaForCausalLM/GlmMoeDsaForCausalLMNextN，modelslim；准备TP16/EP16/DP1/PP1/CP1、BF16及BF16 KV、Ascend传输、page128。这些是元数据与参数解析结果，实际rank容量、KV分配和graph尚未运行。

原M001-v1已经使用原C8模板SHA`3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c`；`69bb3a...`是另一普通W8A8目录模板，并不是M001冻结输入。R026在46的实际服务镜像、实际SGLang tokenizer loader复现全部五组输入：512/512/32768/32768/32768，渲染文本、tokenIDs、ID哈希、vocab范围和普通encode回环全部一致。[结果](runs/L039-R026-inputs/result.json)、[原冻结输入](fixtures/M001-v1/manifest.json)。输入未改版。

R021无驱动库时在导入链缺libascend_hal，R023加入只读驱动库后因零可见NPU被SGLang原启动检查拒绝，均在tokenizer加载前失败、零fixture检查，临时容器已清理。R026只暴露一个已预留NPU以通过原检查；tokenization在CPU执行，未初始化NPU context，没有权重或服务计算。尚未冻结/验证输出策略、EOS/取消/重试、容量和性能，不能把五输入匹配写成原生推理正确性。

## 原始Ascend传输后端支持

所有探针容器均运行于46，结束后清理；没有改生产源码。

| Run | 实际阶段 | 结果 |
| --- | --- | --- |
| R015 | 先导入MemFabric，再TorchNPU lazy init | aclInit507899，尚未构造引擎 |
| R016 | 先导入Torch模块，NPU context仍在MemFabric之后创建 | 同阶段507899 |
| R017 | 先建立TorchNPU context，再导入MemFabric | context成功；原引擎因空store URL报basic_string错误 |
| R018 | context先建；MemFabric common init与任务独立loopback store均返回0 | 原AscendTransferEngine按SDMA初始化，真实4MiB NPU内存register/deregister成功 |

[R018实际结果](runs/L039-R018-memfabric/result.json)、[隔离与清理](runs/L039-R018-memfabric/remote-result.json)，R015–R017原始失败亦保留在本目录。安装版本为MemFabric1.3.0。官方[Python API](https://github.com/Ascend/memfabric_hybrid/blob/master/doc/pythonAPI.md)说明common init和create_config_store接口；实际返回值以安装版探针为准。这里的导入先后观察只适用于这些探针，不构成通用根因或生产修复结论。Loopback store已随探针删除；下一步使用新的任务私网共享store验证44→46真实搬运，不继承其他任务的store。

## 证据边界与下一步

22条新冻结命令R008–R029分别派发一次，输入哈希未变；外层CLI0与内层失败分开审计。[执行器元数据审计](notes/L039-staging-zcode-audit-v2.json)、[主控勘误和审查](notes/L039-staging-controller-review.json)。完整模型I/O、TLS私钥、权重/镜像二进制和其他任务挂载清单不在公开产物中。

接下来在44完成文件重验/补传期间做任务专属共享store和真实两节点传输正确性；两端模型完整、资源仍有效后，再冻结实际服务合同并启动16rank P/D＋MTP。当前没有TTFT、ITL、接受长度、有效token/s、HBM/互联可达上界或同条件基线，36机制记录没有晋升。源端两条旧TLS错误、P44旧进程255、Failed资源对象及输入导入失败都作为证据保留。只回收已核实归属且不再被活动任务需要的对象。
