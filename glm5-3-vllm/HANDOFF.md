# 当前交接 — 2026-10-17 18:51Z

当前在做首 token 发布依赖的自定义机制原型。官方部署已经完成；MTP、现有重叠、graph 的2.25×等结果归入官方基线完善，不当作自定义机制原创收益。自定义worker v1/v2没有无条件收益，未采用；v3/v4不自动继续。

32/33只操作本任务实例/holder，全 SDMA，模型19:21Z硬截止，holder19:31Z到期，不续租。不改算子、模型精度、量化。P11保留，原样官方MooncakeConnectorV2和proxy保留。既有 draft PR2不合并。不主动向其他聊天发消息。

当前候选D12已launch：CID `52c938a6f6d489b9256a8d75c73c46e4de451782b00469bbda89cc0fcc42fcf4`，root `decode-node33-v12`，scheduler只读patch SHA `4f42984e953b1d126bc2f9e4d35dc5c47c6d5623136a5b97df99a9f96b72812e`。相对D10只多scheduler发布机制；worker仍同a4a7文件/gate0。P11 CID `d82b591171b15560f4524fcf9a380537be82c10ee1a6a77df495d01f77d5cffc` 未动。D10全测结束、无active/queued后精确关闭，lifecycle已存。

候选在原生聚合KV完成通知后，提前发布P metadata中的last_token_id；D工作、请求history、async占位、停止和资源回收照常运行，首个原生D输出核对并只抑制一个重复token。不改GPU/host buffer生命周期。仅greedy文本ignore_eos/min0、无stop/logprobs/penalties等的窄范围；其他走原生。若不一致则该请求error并否定候选。它可能降低TTFT，但没有删除D计算，first-to-second gap和平均TPOT可能增加，必须联合报告。

证据：known-first-token-v11-v1 8/8一致，P已知到D首返回约70–82ms（同客户端monotonic；不是可直接声称的收益）；扩展24保护fixture在C1/C4共48/48首token一致，8种首token ID。固定镜像native Request/SamplingParams下CPU-only eligibility及8个fallback/failure检查通过，未把它当模型验证。源码和契约见 `runtime-patches/p-first-publish-v1/`、`evidence/20261017-official-pd-optimization/p-first-publication-loop-contract-v1.json`。

正在运行 `/tmp/glm53-run-p-first-v12.py`，session4474：D12启动后等待Ready，stock proxy smoke3waves共60请求（P11历史profile显式已停止gate）。先看smoke、实际published/verified/mismatch计数，再排48 eligible varied请求、24 C4 native/PD fallback保护、ignore_eos max1/cancel复用及1↔4转换、两轮formal。余量允许则同文件gate0恢复作更近控制；不因版本号自动扩大实验。所有模型仍19:21Z停止。

P11/D10完整基线已结束：正式240/240；24 C1/C4 native/PD各24/24；16reuse pairs+8 transitions；同arrival925/925；持续两轮2115/2115。相对P8/D10 eager，两轮TTFT p50几何均值0.86373、TPOT0.98028、波次吞吐点估计1.08411。持续256约208.67tok/s、1024约165.04tok/s；不得把波次点估计直接当持续容量。原12题仍11/12，既有短few-shot2+3=输出4保留，不宣称完整精度验收。

P11/D10实际profile均已API stop；CPU rank0导出完成，P:113059events/3236SDMA、D:496640events/18184SDMA，矩阵LOCAL/SIO/HCCS，无RDMA命名任务，范围仅rank0窗口；旧全32rank地图复用。P11全部测试203文件已Mac SHA归档，receipt node32-prefill-graph-v11-raw-backup-v1.json。rank0 raw+trace已在D12加载空窗完成Mac逐文件SHA核验。D10首次cat下载截断虽rc0，已保留partial并改4MiB逐块SHA校验，完整archive重新通过；不混入正式计时。

较早环境、命令、holder UID和所有历史Run见 [历史交接](analysis/handoff-history-through-p11-20261017.md)。本文件保持当前候选、证据和下一最小实验，不要求重读全部历史。
