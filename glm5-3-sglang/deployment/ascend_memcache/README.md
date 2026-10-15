# Ascend MemCache 现场适配工具

使用方法、固定版本和验证边界见 [部署文档](../../ASCEND_MEMCACHE_PD_DEPLOYMENT.md)。

- `patches/mc-apply-patch.py`：针对固定SGLang镜像的精确锚点安装，内部执行 `mc-store-index-delta.py`，保留回滚备份。
- `patches/ascend_memcache_src/`：SGLang PR26043的后端源文件，固定源 head `08246a9e20fef1d75b4ee6c3d9646c415f215a54`；遵循 [上游许可证](https://github.com/sgl-project/sglang/blob/main/LICENSE)。
- `fix_ascend_dsa_budget.py`：固定源码SHA的Ascend MLA DSA index显存预算修复，保留GPU/HiSparse分支并保存备份。
- `fix_sonames.sh`：容器内wheel动态库解析适配。
- `start_meta_service.py`：由JSON配置启动独立MetaService。
- `pool_provider.py`：独立DDR提供者，SIGTERM/SIGINT时显式close；SDK1.1.5。
- `peer_metrics.py`：按当前engine日志寻找Prometheus mmap目录，提供只读metrics。

不包含权重、镜像、wheel二进制、现场地址和凭据。patch锚点检查失败时先核对目标镜像及源代码，不能把其他SGLang版本视为已兼容。
