#!/usr/bin/env python3
"""Read actual imported frontend/config paths in the pinned owned stage tree."""
from pathlib import Path
import importlib.util

path=Path(__file__).with_name('collect_frozen_sources.py')
spec=importlib.util.spec_from_file_location('source_transport',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source=module.REMOTE
begin=source.index('selection={');end=source.index("manifest={'scope':",begin)
selection={'vllm-ascend':[], 'vllm':[
    'vllm/entrypoints/generate/base/protocol.py',
    'vllm/entrypoints/generate/base/serving.py',
    'vllm/entrypoints/serve/engine/protocol.py',
    'vllm/config/multimodal.py',
    'vllm/transformers_utils/config.py',
    'vllm/transformers_utils/configs/deepseek_v41.py',
    'vllm/transformers_utils/configs/__init__.py',
]}
module.REMOTE=source[:begin]+'selection='+repr(selection)+'\n'+source[end:]
work=Path(__file__).resolve().parent/'initialization-source-supplement'
work.mkdir(exist_ok=False);module.__file__=str(work/'collect_frozen_sources.py');module.main()
