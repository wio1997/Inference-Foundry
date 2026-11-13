# 实测138/139/140：实际命令与配置附录

本附录来自已经运行的reproduction清单，不是141预备命令。容器、8rank完整argv、环境、有效特性和原始计时端点以文末JSON为权威；下方抽出服务命令便于阅读。

## vLLM138服务参数

GLM53_KDA_CAPTURE=1通过固定插件启用兼容与观测；MTP5路径使用共同MTP observer，不是非MTP的旧KDA捕获shim。完整插件及额外helper还需按固定源码准备。

```bash
python3 -u -m vllm.entrypoints.cli.main serve /model --host 127.0.0.1 --port 18080 --served-model-name glm53flash --tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 1 --enable-expert-parallel --dtype bfloat16 --seed 0 --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 1792 --enable-chunked-prefill --enable-prefix-caching --block-size 128 --mamba-block-size 128 --mamba-cache-mode align --additional-config '{"enable_dsa_cp": true, "enable_flashcomm1": false, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": true, "enable_super_kernel": true}, "enable_prefill_mc2": true}' --async-scheduling --skip-mm-profiling --kv-cache-memory-bytes 4294967296 --cpu-offload-gb 0 --offload-group-size 0 --offload-backend auto --disable-log-stats --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6, 12, 24, 48]}' --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 5, "enforce_eager": false}' --quantization ascend
```

## DeepInfer139 rank0模型参数

其它rank采用相同参数，只改变rank/local-device/go-file等身份值；由DeepInfer原生引擎和Worker驱动。入口包装随本次实验清单提供。

```bash
python3 -u /task/main-session/batch-g139-shared-stride/overlay/postscore_profile_entry.py --model /model --rank 0 --local-device 0 --init-method tcp://127.0.0.1:29795 --output-dir /task/main-session/batch-g139-shared-stride/native-run/communication/native-tp8 --go-file /task/main-session/batch-g139-shared-stride/native-run/communication/native-tp8/rank0/go --world-size 8 --weight-offload none --max-context 2048 --max-step 1792 --max-new-tokens 256 --compute-backend sdk-batch --max-sequences 8 --cache-bytes 4294967296 --serve --host 127.0.0.1 --port 18080 --served-model-name glm53flash --serve-max-requests 72 --graph-decode --prefix-caching --async-scheduling --enable-expert-parallel --shared-expert-overlap --mtp-tokens 5 --mtp-indexed-loading --mtp-draft-graph --dsa-context-parallel --static-kernel --super-kernel --native-mtp-logprob-gather --native-prefill-state --native-admission-cohort --native-prefix-admission --prefill-mc2
```

140的模型源码和argv与139一致，实际入口文件内容改为状态复用旁路包装。不能只比较argv而忽略入口源码SHA256。

## AISBench正式命令

在既有固定环境中，每臂依次运行C4、C8。以下是从实际命令抽出的AISBench层；完整docker资源、挂载与环境见JSON。

```bash
/task/aisbench-tp8/venv/bin/ais_bench /task/aisbench-tp8/aisbench_c4.py -m perf --num-warmups 2 --work-dir <该臂C4结果目录>
/task/aisbench-tp8/venv/bin/ais_bench /task/aisbench-tp8/aisbench_c8.py -m perf --num-warmups 2 --work-dir <该臂C8结果目录>
```

提供原样的`aisbench_c4.py`、`aisbench_c8.py`及`frozen-1024-32.jsonl`。`/task/...`是测量时容器内布局，复现时需按挂载规则准备，不能直接当成本机下载路径。

## 复现入口与身份

- [参考138完整命令和环境](../validation/shared-features-20261111/reference138-reproduction.json)
- [开启139完整命令和环境](../validation/shared-features-20261111/native139-reproduction.json)
- [关闭140完整命令和环境](../validation/shared-features-20261111/native140-reproduction.json)
- [原始C4配置](../validation/performance-attribution-20261113/aisbench_c4.py)
- [原始C8配置](../validation/performance-attribution-20261113/aisbench_c8.py)
- [原始固定数据32条](../validation/performance-attribution-20261113/frozen-1024-32.jsonl)
- [数据/配置哈希、客户端版本和六份结果输入核对](../validation/performance-attribution-20261113/reproduction-inputs-manifest.json)

权重、tokenizer、已编译NPU扩展和SDK不随合成数据分发，身份按固定清单核对。三臂请求均使用temperature0、top_p1、seed0与ignore_eos，C8沿用C4状态。

## 新增复现边界：流式响应粒度不同

相同stream=True及256输出token不代表SSE分帧一致。正式原始时间点记录数为参考每请求46–48个、原生每请求259个；原生sdk_http按原生token回调输出可解码文本增量。时间点不能当成token或模型forward次数；吞吐使用实际输出token数8192及整批客户端起止时间。新增[客户端时间审计](../validation/performance-attribution-20261113/client-cadence-audit.json)完整披露此差异与阈值口径，未用帧数冒充计算量。
