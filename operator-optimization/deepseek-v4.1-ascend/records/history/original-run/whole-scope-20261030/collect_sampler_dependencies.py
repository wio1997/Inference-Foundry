#!/usr/bin/env python3
"""Supplement missing Ascend sampler/registration/head source dependencies.

Uses the same checked, read-only archive transport as the primary snapshot.
"""
from pathlib import Path
import importlib.util
import json

path=Path(__file__).with_name('collect_frozen_sources.py')
spec=importlib.util.spec_from_file_location('source_snapshot_transport',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source=module.REMOTE
begin=source.index('selection={')
end=source.index("manifest={'scope':",begin)
selection={
    'vllm-ascend':['vllm_ascend/sample','vllm_ascend/ops','vllm_ascend/patch',
                   'vllm_ascend/models/__init__.py','vllm_ascend/platform.py',
                   'vllm_ascend/__init__.py','vllm_ascend/envs.py'],
    'vllm':['vllm/models/qwen3_dspark.py','vllm/model_executor/custom_op.py',
            'vllm/model_executor/layers/pluggable.py','vllm/model_executor/models/registry.py',
            'vllm/platforms/__init__.py']}
module.REMOTE=source[:begin]+'selection='+repr(selection)+'\n'+source[end:]
# Isolate all outputs; never overwrite the primary source snapshot/receipt.
work=Path(__file__).resolve().parent/'sampler-source-supplement'
work.mkdir(exist_ok=False)
module.__file__=str(work/'collect_frozen_sources.py')
module.main()
