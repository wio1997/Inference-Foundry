# 源码补丁与校验

两份补丁依次对应 1436 → 1437 → 1440 文件。操作前复制冻结源码至独立目录，不对运行中的服务目录应用补丁。旧版 `artifacts/source.tar.gz` 和 `config/source-SHA256.json` 保持原样。

以下示例在 172 宿主执行，`D` 为本项目交付目录，`S` 为新建的源码副本。目标目录必须不存在：

```bash
D=/data/tiankuan/wio/910c-deepinfer-172/delivery/deepinfer-dsv41-910c
U="$D/optimizations/20261028-device-hash-overlap"
S=/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-overlap-review
test ! -e "$S" || exit 1
mkdir "$S"
tar -xzf "$D/artifacts/source.tar.gz" -C "$S"
python3 "$U/verify-source.py" "$S" "$D/config/source-SHA256.json"
patch --dry-run -d "$S" -p1 -i "$U/01-device-hash.patch"
patch -d "$S" -p1 -i "$U/01-device-hash.patch"
python3 "$U/verify-source.py" "$S" "$U/device-hash-SHA256.json"
patch --dry-run -d "$S" -p1 -i "$U/02-shared-expert.patch"
patch -d "$S" -p1 -i "$U/02-shared-expert.patch"
python3 "$U/verify-source.py" "$S" "$U/shared-expert-SHA256.json"
```

补丁和源码清单仅用于恢复、审阅和核验实验源码。上述命令不会创建容器、启动服务或复现测试；实验结果及限制见 [RESULTS.md](RESULTS.md)。
