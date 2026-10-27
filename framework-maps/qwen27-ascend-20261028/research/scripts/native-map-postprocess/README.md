# 原生分片地图后处理代码

本组件保存已实际部署的地图合成入口及冻结模板。合成器逐份核验压缩与解压后的 SHA、字节数、身份和行数守恒，默认要求完整 204 个身份、错误数组为空，保留旧地图数组与 unknown。单 rank 上限为 256 MiB，与实际分片器一致；不会读取完整多 GB bundle。renderer 生成 HTTP 页面，portable 只内嵌 41 份 CPU lifecycle 数据；原生分片仍需按需读取，不能将其称为全原生数据的离线单文件。

本代只增加归档入口；六份历史代码/模板均保留原字节，没有修改正在运行的 controller。源码绑定见 [清单](source-bindings.json)。实际运行的单 rank 诊断验证位于 EVIDENCE 中，不能代表最终 204 地图。

在本目录运行标准库检查：

```sh
python3 -B -m unittest discover -s . -p 'test_compose*.py' -v
```

七项用例覆盖保留旧数据、身份冲突、损坏 rank/unattributed、256 MiB 声明、超限拒绝与默认 204 拒绝不足身份。CPU fixtures 不代表设备行为或推理性能；未重新运行 NPU、服务、SDK 或性能矩阵。

合成与渲染需要完整、经过验收的 manifest、index、各 rank gzip、既有 combined-map.json 和 41 CPU shards。示意入口如下，输入目录由最终收据给出：

```sh
python3 -B compose_sharded_map.py --combined-map INPUT/combined-map.json --index INPUT/native-rank-details/index.json --completion-manifest INPUT/manifest.json --output-dir OUTPUT
python3 -B renderer.py --input OUTPUT/combined-map.json --output OUTPUT/index-HTTP.html --template template.html
python3 -B build_portable_map.py --source OUTPUT/index-HTTP.html --output OUTPUT/index.html
```

最后一步前须按原数据 SHA 将 41 份 CPU shards 放入 OUTPUT/cpu-lifecycle-details。原生结果的浏览器实际选择/SHA核对仍是独立门槛。gen91 四 rank 候选窗口只是显式 marker 包含关系；外围完成语义、独占 kernel、全 stream 关键路径未获证明，不能升级为采用结论。质量/greedy 等价、最终 204、发布及 root 整体验收尚未完成。
